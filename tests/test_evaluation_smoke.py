"""Smoke test untuk baseline Linear SVM dan evaluasi komparatif Test Set (Fase 5)."""

import json
from pathlib import Path

import pytest

from src.dataset.mock import generate_mock_ground_truth
from src.dataset.split import create_stratified_split
from src.models.baseline_svm import (
    BaselineSVMClassifier,
    train_baseline_for_aspect,
)
from src.training.evaluate import (
    evaluate_test_set_comparison,
    format_comparison_table,
)
from src.training.trainer import train_aspect_model


def test_baseline_svm_fit_predict_save_load(tmp_path: Path):
    """Verifikasi fit, predict, save, dan load pada BaselineSVMClassifier."""
    texts = [
        "Great combat and tight gameplay mechanics!",
        "Awful story and terrible character dialogue.",
        "Average graphics, runs okay, not much to say.",
    ]
    labels = [1, 2, 0]

    model = BaselineSVMClassifier(random_state=42)
    model.fit(texts, labels)

    preds = model.predict(["Awesome mechanics", "Hate this storyline"])
    assert len(preds) == 2
    assert set(preds).issubset({0, 1, 2})

    save_path = tmp_path / "svm_test" / "model.joblib"
    model.save(save_path)
    assert save_path.exists()

    loaded_model = BaselineSVMClassifier(random_state=42)
    loaded_model.load(save_path)
    loaded_preds = loaded_model.predict(["Awesome mechanics", "Hate this storyline"])
    assert loaded_preds == preds


def test_train_baseline_for_aspect(tmp_path: Path):
    """Verifikasi training baseline SVM untuk satu aspek menghasilkan model.joblib dan summary."""
    gt_csv = tmp_path / "ground_truth.csv"
    split_json = tmp_path / "split_indices.json"
    models_dir = tmp_path / "models"

    generate_mock_ground_truth(output_path=gt_csv, n_samples=30, seed=42)
    create_stratified_split(ground_truth_path=gt_csv, output_split_path=split_json, seed=42)

    metrics = train_baseline_for_aspect(
        aspect_name="gameplay",
        ground_truth_path=gt_csv,
        split_indices_path=split_json,
        models_dir=models_dir,
    )

    assert "macro_f1" in metrics
    assert "active_macro_f1" in metrics
    assert (models_dir / "svm_gameplay" / "model.joblib").exists()
    assert (models_dir / "svm_gameplay" / "training_summary.json").exists()


def test_baseline_rejects_invalid_aspect_labels_before_training(tmp_path: Path):
    import pandas as pd

    gt_csv = tmp_path / "invalid_ground_truth.csv"
    split_json = tmp_path / "split_indices.json"
    df = generate_mock_ground_truth(output_path=gt_csv, n_samples=30, seed=42)
    df.loc[0, "gameplay"] = 3
    df.to_csv(gt_csv, index=False)
    review_ids = df["review_id"].astype(str).tolist()
    split_json.write_text(
        json.dumps(
            {
                "train_ids": review_ids[:20],
                "val_ids": review_ids[20:25],
                "test_ids": review_ids[25:],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="integer non-null"):
        train_baseline_for_aspect(
            aspect_name="gameplay",
            ground_truth_path=gt_csv,
            split_indices_path=split_json,
            models_dir=tmp_path / "models",
        )


def test_evaluate_test_set_comparison(tmp_path: Path):
    """Verifikasi evaluasi komparatif apple-to-apple antara Linear SVM vs BERT pada Test Set identik."""
    gt_csv = tmp_path / "ground_truth.csv"
    split_json = tmp_path / "split_indices.json"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    # 1. Setup mock data dan split 70/15/15
    generate_mock_ground_truth(output_path=gt_csv, n_samples=32, seed=42)
    create_stratified_split(ground_truth_path=gt_csv, output_split_path=split_json, seed=42)

    # 2. Latih Linear SVM gameplay
    train_baseline_for_aspect(
        aspect_name="gameplay",
        ground_truth_path=gt_csv,
        split_indices_path=split_json,
        models_dir=models_dir,
    )

    # 3. Latih BERT gameplay 1 epoch pada CPU
    train_aspect_model(
        aspect_name="gameplay",
        ground_truth_path=gt_csv,
        split_indices_path=split_json,
        models_dir=models_dir,
        epochs=1,
        batch_size=8,
        lr=2e-5,
        device="cpu",
    )

    # 4. Jalankan evaluasi komparatif pada test set
    comparison = evaluate_test_set_comparison(
        ground_truth_path=gt_csv,
        split_indices_path=split_json,
        models_dir=models_dir,
        reports_dir=reports_dir,
        device="cpu",
        aspects=["gameplay"],
    )

    # 5. Verifikasi berkas laporan evaluasi
    report_file = reports_dir / "evaluation_comparison.json"
    assert report_file.exists(), f"Laporan {report_file} tidak ditemukan!"

    with open(report_file, "r", encoding="utf-8") as f:
        saved_report = json.load(f)

    assert "gameplay" in saved_report
    gp_data = saved_report["gameplay"]

    assert "svm" in gp_data and "bert" in gp_data and "delta" in gp_data
    assert "macro_f1" in gp_data["svm"] and "active_macro_f1" in gp_data["svm"]
    assert "macro_f1" in gp_data["bert"] and "active_macro_f1" in gp_data["bert"]
    assert "macro_f1" in gp_data["delta"] and "active_macro_f1" in gp_data["delta"]

    # 6. Verifikasi format tabel ASCII
    table_str = format_comparison_table(saved_report)
    assert "GAMEPLAY" in table_str
    assert "Linear SVM" in table_str
    assert "BERT (Proposed)" in table_str
    assert "Delta (BERT - SVM)" in table_str


def test_baseline_rejects_overlapping_split_indices(tmp_path: Path):
    """Verifikasi bahwa baseline SVM menolak split_indices dengan ID tumpang tindih."""
    gt_csv = tmp_path / "gt.csv"
    split_json = tmp_path / "split_indices.json"
    generate_mock_ground_truth(output_path=gt_csv, n_samples=20, seed=42)

    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_ids": ["rev_0", "rev_1"],
                "val_ids": ["rev_2"],
                "test_ids": ["rev_1", "rev_3"],
            },
            f,
        )

    with pytest.raises(ValueError, match="Data leakage terdeteksi"):
        train_baseline_for_aspect(
            aspect_name="gameplay",
            ground_truth_path=gt_csv,
            split_indices_path=split_json,
            models_dir=tmp_path / "models",
        )

