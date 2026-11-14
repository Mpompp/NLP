"""Smoke test untuk Bulk Inference Engine, Game Scorecard (ASI), dan Master CLI run.py."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from run import build_parser, get_ground_truth_path
import src.inference.predict as predict_module
from src.inference.predict import predict_unseen_corpus
from src.models.baseline_svm import BaselineSVMClassifier
from src.inference.scorecard import (
    determine_diagnostic_flag,
    format_scorecard_summary,
    generate_game_scorecard,
)
from src.models.bert_classifier import CustomBERTClassifier


def test_determine_diagnostic_flag():
    """Verifikasi aturan penentuan flag diagnostik otomatis."""
    # 1. Data kurang / None / NaN
    assert determine_diagnostic_flag(None, 80.0, 80.0) == "Data Kurang"
    assert determine_diagnostic_flag(80.0, float("nan"), 80.0) == "Data Kurang"

    # 2. Alert Stabilitas Teknis Kritis (< 40.0)
    assert determine_diagnostic_flag(90.0, 90.0, 39.9) == "ALERT: Stabilitas Teknis Kritis (Crash/Bug)"

    # 3. Pujian Kreatif & Mekanik Sangat Tinggi (gameplay >= 85 dan story >= 85)
    assert determine_diagnostic_flag(85.0, 85.0, 80.0) == "Pujian Kreatif & Mekanik Sangat Tinggi"
    assert determine_diagnostic_flag(95.0, 88.0, 70.0) == "Pujian Kreatif & Mekanik Sangat Tinggi"

    # 4. Keluhan Mekanik / Kontrol Dominan (gameplay < 50.0)
    assert determine_diagnostic_flag(49.9, 70.0, 70.0) == "Keluhan Mekanik/Kontrol Dominan"

    # 5. Karakteristik Seimbang
    assert determine_diagnostic_flag(75.0, 70.0, 70.0) == "Karakteristik Seimbang"


def test_generate_game_scorecard_calculation(tmp_path: Path):
    """Verifikasi presisi perhitungan matematis Aspect Satisfaction Index (ASI) dan penanganan threshold."""
    pred_csv = tmp_path / "predictions.csv"
    ref_tsv = tmp_path / "appids.tsv"
    out_csv = tmp_path / "scorecards.csv"

    # Buat file TSV referensi gim
    ref_tsv.write_text("appid\tjudul\n100\tGame Perfect\n200\tGame Broken\n300\tGame Small\n", encoding="utf-8")

    # Buat dataset prediksi:
    # Game 100: 10 ulasan (Gameplay: 9 pos (1), 1 neg (2) -> ASI = 90.0%)
    #                       (Story: 9 pos (1), 1 neg (2) -> ASI = 90.0%)
    #                       (Technical: 8 pos (1), 2 neg (2) -> ASI = 80.0%)
    rows = []
    for _ in range(8):
        rows.append({"review_id": f"rev_100_{len(rows)}", "app_id": 100, "pred_gameplay": 1, "pred_story": 1, "pred_technical": 1})
    rows.append({"review_id": f"rev_100_{len(rows)}", "app_id": 100, "pred_gameplay": 1, "pred_story": 1, "pred_technical": 2})
    rows.append({"review_id": f"rev_100_{len(rows)}", "app_id": 100, "pred_gameplay": 2, "pred_story": 2, "pred_technical": 2})

    # Game 200: 10 ulasan (Technical: 2 pos (1), 8 neg (2) -> ASI = 20.0% -> Alert Teknis Kritis)
    for _ in range(2):
        rows.append({"review_id": f"rev_200_{len(rows)}", "app_id": 200, "pred_gameplay": 1, "pred_story": 1, "pred_technical": 1})
    for _ in range(8):
        rows.append({"review_id": f"rev_200_{len(rows)}", "app_id": 200, "pred_gameplay": 1, "pred_story": 1, "pred_technical": 2})

    # Game 300: hanya 3 ulasan polar -> di bawah threshold min_reviews_threshold=5 -> "Data Kurang"
    for _ in range(3):
        rows.append({"review_id": f"rev_300_{len(rows)}", "app_id": 300, "pred_gameplay": 1, "pred_story": 1, "pred_technical": 1})

    pd.DataFrame(rows).to_csv(pred_csv, index=False)

    df_scorecard = generate_game_scorecard(
        predictions_path=pred_csv,
        appids_ref_path=ref_tsv,
        output_path=out_csv,
        min_reviews_threshold=5,
    )

    assert len(df_scorecard) == 3
    assert out_csv.exists()

    row_100 = df_scorecard[df_scorecard["app_id"] == 100].iloc[0]
    assert row_100["game_name"] == "Game Perfect"
    assert pytest.approx(row_100["gameplay_asi"], rel=1e-4) == 90.0
    assert pytest.approx(row_100["story_asi"], rel=1e-4) == 90.0
    assert pytest.approx(row_100["technical_asi"], rel=1e-4) == 80.0
    assert row_100["diagnostic_flag"] == "Pujian Kreatif & Mekanik Sangat Tinggi"

    row_200 = df_scorecard[df_scorecard["app_id"] == 200].iloc[0]
    assert row_200["game_name"] == "Game Broken"
    assert pytest.approx(row_200["technical_asi"], rel=1e-4) == 20.0
    assert row_200["diagnostic_flag"] == "ALERT: Stabilitas Teknis Kritis (Crash/Bug)"

    row_300 = df_scorecard[df_scorecard["app_id"] == 300].iloc[0]
    assert row_300["game_name"] == "Game Small"
    assert pd.isna(row_300["gameplay_asi"])
    assert row_300["diagnostic_flag"] == "Data Kurang"

    # Verifikasi format tabel konsol
    summary_str = format_scorecard_summary(df_scorecard)
    assert "Game Perfect" in summary_str
    assert "Game Broken" in summary_str
    assert "Pujian Kreatif" in summary_str

    # Verifikasi bahwa top_n=1 tetap menampilkan gim dengan ALERT kritis (Game Broken)
    summary_top1 = format_scorecard_summary(df_scorecard, top_n=1)
    assert "Game Perfect" in summary_top1
    assert "Game Broken" in summary_top1
    assert "ALERT: Stabilitas Teknis Kritis (Crash/Bug)" in summary_top1
    assert "Game Small" not in summary_top1


def test_cli_parser_and_subcommands():
    """Verifikasi bahwa parser CLI mendefinisikan seluruh 9 sub-command pipeline dengan benar."""
    parser = build_parser()
    expected_commands = {
        "scrape",
        "clean",
        "sample",
        "split",
        "train",
        "baseline",
        "evaluate",
        "predict",
        "scorecard",
    }

    # Ambil list sub-command dari parser
    registered_commands = set(parser._subparsers._actions[1].choices.keys())

    assert expected_commands.issubset(registered_commands), f"Ada sub-command yang hilang: {expected_commands - registered_commands}"

    # Jalankan subprocess python run.py --help
    res = subprocess.run([sys.executable, "run.py", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Steam Aspect-Category Sentiment Analysis (ACSA) Master CLI" in res.stdout
    for cmd in expected_commands:
        assert cmd in res.stdout

    evaluate_args = parser.parse_args(["evaluate", "--ground-truth", "labels.csv"])
    assert evaluate_args.command == "evaluate"
    assert evaluate_args.ground_truth == "labels.csv"
    assert evaluate_args.aspect == "all"
    assert parser.parse_args(["evaluate", "--aspect", "gameplay", "--mock"]).aspect == "gameplay"
    assert parser.parse_args(["evaluate", "--mock"]).mock is True


def test_ground_truth_requires_real_file_or_explicit_mock(tmp_path: Path, monkeypatch):
    import run

    annotation_dir = tmp_path / "annotation"
    annotation_dir.mkdir()
    monkeypatch.setattr(run, "get_config", lambda: {"paths": {"annotation_dir": str(annotation_dir)}})

    with pytest.raises(FileNotFoundError, match="Berkas ground truth tidak ditemukan"):
        get_ground_truth_path(custom_path=str(tmp_path / "missing.csv"))

    with pytest.raises(FileNotFoundError, match="File ground truth riil belum ditemukan"):
        get_ground_truth_path()

    mock_path = annotation_dir / "ground_truth_test.csv"
    mock_path.write_text("review_id\n", encoding="utf-8")
    assert get_ground_truth_path(mock=True) == mock_path
    assert not (annotation_dir / "ground_truth.csv").exists()


def test_predict_requires_split_indices(tmp_path: Path):
    corpus_csv = tmp_path / "corpus_clean.csv"
    pd.DataFrame([{"review_id": "r1", "app_id": 1, "clean_text": "review"}]).to_csv(corpus_csv, index=False)
    missing_split = tmp_path / "missing_split_indices.json"
    expected_error = "File split_indices.json wajib ada untuk menyaring ulasan unseen agar tidak terjadi kebocoran data ground truth!"

    for split_path in (None, missing_split):
        with pytest.raises(FileNotFoundError) as error:
            predict_unseen_corpus(corpus_path=corpus_csv, split_indices_path=split_path)
        assert str(error.value) == expected_error


def test_predict_rejects_invalid_split_indices(tmp_path: Path):
    corpus_csv = tmp_path / "corpus_clean.csv"
    split_json = tmp_path / "split_indices.json"
    pd.DataFrame([{"review_id": "r1", "app_id": 1, "clean_text": "review"}]).to_csv(corpus_csv, index=False)
    split_json.write_text(json.dumps({"train_ids": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="split_indices.json tidak valid"):
        predict_unseen_corpus(corpus_path=corpus_csv, split_indices_path=split_json)


def test_predict_preflights_all_checkpoints_before_inference(tmp_path: Path, monkeypatch):
    corpus_csv = tmp_path / "corpus_clean.csv"
    split_json = tmp_path / "split_indices.json"
    models_dir = tmp_path / "models"
    pd.DataFrame([{"review_id": "r1", "app_id": 1, "clean_text": "review"}]).to_csv(corpus_csv, index=False)
    split_json.write_text(json.dumps({"train_ids": ["t1"], "val_ids": ["v1"], "test_ids": ["te1"]}), encoding="utf-8")
    for aspect in ("gameplay", "story"):
        checkpoint = models_dir / f"bert_{aspect}" / "best_model.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()

    def fail_on_inference(*args, **kwargs):
        pytest.fail("Tokenizer or model inference started before checkpoint preflight completed")

    monkeypatch.setattr(predict_module.AutoTokenizer, "from_pretrained", fail_on_inference)
    monkeypatch.setattr(predict_module, "CustomBERTClassifier", fail_on_inference)
    with pytest.raises(FileNotFoundError, match="aspek 'technical'"):
        predict_unseen_corpus(
            corpus_path=corpus_csv,
            split_indices_path=split_json,
            models_dir=models_dir,
            device="cpu",
        )


def test_svm_predict_accepts_numpy_and_series():
    model = BaselineSVMClassifier().fit(
        ["Great combat and tight gameplay", "Awful story and terrible dialogue", "Average graphics and sound"],
        [1, 2, 0],
    )
    assert model.predict(np.array([])) == []
    assert model.predict(pd.Series([], dtype=str)) == []
    assert len(model.predict(np.array(["Awesome mechanics", "Hate this storyline"]))) == 2
    assert len(model.predict(pd.Series(["Awesome mechanics", "Hate this storyline"]))) == 2


def test_predict_unseen_corpus_pipeline(tmp_path: Path):
    """Verifikasi inferensi massal 3 model BERT dengan filter eksklusi data train/val/test."""
    corpus_csv = tmp_path / "corpus_clean.csv"
    split_json = tmp_path / "split_indices.json"
    models_dir = tmp_path / "models"
    out_csv = tmp_path / "predictions.csv"

    # 1. Buat data corpus bersih (6 ulasan: 3 split ids, 3 unseen)
    reviews = [
        {"review_id": "rev_train_1", "app_id": 500, "clean_text": "Good combat and graphics", "voted_up": True},
        {"review_id": "rev_train_2", "app_id": 500, "clean_text": "Boring dialogue and bad controls", "voted_up": False},
        {"review_id": "rev_test_1", "app_id": 500, "clean_text": "Test partition review", "voted_up": True},
        {"review_id": "rev_unseen_1", "app_id": 500, "clean_text": "Awesome game with great gameplay", "voted_up": True},
        {"review_id": "rev_unseen_2", "app_id": 600, "clean_text": "Crashes on startup, terrible bug", "voted_up": False},
        {"review_id": "rev_unseen_3", "app_id": 600, "clean_text": "Decent soundtrack and visual style", "voted_up": True},
    ]
    pd.DataFrame(reviews).to_csv(corpus_csv, index=False)

    # 2. Buat file split_indices.json yang mengecualikan ulasan train, val, test
    split_data = {
        "train_ids": ["rev_train_1"],
        "val_ids": ["rev_train_2"],
        "test_ids": ["rev_test_1"],
    }
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(split_data, f)

    # 3. Buat mock checkpoint model untuk 3 aspek
    dummy_model = CustomBERTClassifier()
    for aspect in ("gameplay", "story", "technical"):
        aspect_dir = models_dir / f"bert_{aspect}"
        aspect_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dummy_model.state_dict(), aspect_dir / "best_model.pt")

    # 4. Jalankan predict_unseen_corpus
    pred_df = predict_unseen_corpus(
        corpus_path=corpus_csv,
        split_indices_path=split_json,
        models_dir=models_dir,
        output_path=out_csv,
        batch_size=2,
        device="cpu",
    )

    # 5. Verifikasi bahwa tepat 3 ulasan unseen yang diprediksi
    assert len(pred_df) == 3
    assert set(pred_df["review_id"]) == {"rev_unseen_1", "rev_unseen_2", "rev_unseen_3"}
    assert list(pred_df.columns) == [
        "review_id",
        "app_id",
        "clean_text",
        "pred_gameplay",
        "pred_story",
        "pred_technical",
    ]
    for aspect in ("gameplay", "story", "technical"):
        assert set(pred_df[f"pred_{aspect}"]).issubset({0, 1, 2})
    assert out_csv.exists()


def test_generate_game_scorecard_zero_relevant_reviews(tmp_path: Path):
    """Verifikasi bahwa data dengan hanya ulasan kelas 0 (none) dan min_reviews_threshold=0 tidak memicu ZeroDivisionError."""
    pred_csv = tmp_path / "predictions_zero.csv"
    ref_tsv = tmp_path / "appids_zero.tsv"
    out_csv = tmp_path / "scorecards_zero.csv"

    ref_tsv.write_text("appid\tjudul\n999\tGame Neutral\n", encoding="utf-8")

    # Seluruh ulasan hanya berlabel 0 (none)
    rows = [
        {"review_id": f"rev_zero_{i}", "app_id": 999, "pred_gameplay": 0, "pred_story": 0, "pred_technical": 0}
        for i in range(5)
    ]
    pd.DataFrame(rows).to_csv(pred_csv, index=False)

    df_scorecard = generate_game_scorecard(
        predictions_path=pred_csv,
        appids_ref_path=ref_tsv,
        output_path=out_csv,
        min_reviews_threshold=0,
    )

    assert len(df_scorecard) == 1
    row = df_scorecard.iloc[0]
    assert row["game_name"] == "Game Neutral"
    assert pd.isna(row["gameplay_asi"])
    assert pd.isna(row["story_asi"])
    assert pd.isna(row["technical_asi"])
    assert row["diagnostic_flag"] == "Data Kurang"


def test_predict_rejects_empty_split_partition(tmp_path: Path):
    """Verifikasi bahwa predict_unseen_corpus menolak split_indices dengan partisi kosong."""
    corpus_csv = tmp_path / "corpus_clean.csv"
    split_json = tmp_path / "split_indices.json"
    pd.DataFrame([{"review_id": "r1", "app_id": 1, "clean_text": "review"}]).to_csv(corpus_csv, index=False)
    split_json.write_text(json.dumps({"train_ids": ["t1"], "val_ids": [], "test_ids": ["te1"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="tidak boleh kosong"):
        predict_unseen_corpus(corpus_path=corpus_csv, split_indices_path=split_json)


def test_generate_game_scorecard_rejects_null_app_id(tmp_path: Path):
    """Verifikasi bahwa generate_game_scorecard menolak ulasan prediksi dengan app_id bernilai null/NaN."""
    pred_csv = tmp_path / "predictions.csv"
    ref_tsv = tmp_path / "ref.tsv"
    out_csv = tmp_path / "scorecard.csv"

    ref_tsv.write_text("appid\tjudul\n100\tGame One\n", encoding="utf-8")
    rows = [
        {"review_id": "r1", "app_id": 100, "pred_gameplay": 1, "pred_story": 1, "pred_technical": 1},
        {"review_id": "r2", "app_id": None, "pred_gameplay": 1, "pred_story": 1, "pred_technical": 1},
    ]
    pd.DataFrame(rows).to_csv(pred_csv, index=False)

    with pytest.raises(ValueError, match="app_id bernilai null/NaN"):
        generate_game_scorecard(
            predictions_path=pred_csv,
            appids_ref_path=ref_tsv,
            output_path=out_csv,
        )


def test_generate_game_scorecard_handles_nan_title(tmp_path: Path):
    """Verifikasi bahwa referensi game dengan judul kosong/'nan' menggunakan fallback App ID."""
    pred_csv = tmp_path / "predictions.csv"
    ref_tsv = tmp_path / "ref.tsv"
    out_csv = tmp_path / "scorecard.csv"

    ref_tsv.write_text("appid\tjudul\n100\t\n200\tnan\n", encoding="utf-8")
    rows = [
        {"review_id": "r1", "app_id": 100, "pred_gameplay": 1, "pred_story": 1, "pred_technical": 1},
        {"review_id": "r2", "app_id": 200, "pred_gameplay": 1, "pred_story": 1, "pred_technical": 1},
    ]
    pd.DataFrame(rows).to_csv(pred_csv, index=False)

    scorecard_df = generate_game_scorecard(
        predictions_path=pred_csv,
        appids_ref_path=ref_tsv,
        output_path=out_csv,
        min_reviews_threshold=1,
    )

    names = dict(zip(scorecard_df["app_id"], scorecard_df["game_name"]))
    assert names[100] == "App ID 100"
    assert names[200] == "App ID 200"

