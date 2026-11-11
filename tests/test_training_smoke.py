"""Smoke tests untuk CustomBERTClassifier, metrics, dan manual PyTorch training loop."""

import json
from pathlib import Path

import pytest
import torch

import src.training.trainer as trainer
from src.dataset.mock import generate_mock_ground_truth
from src.dataset.split import create_stratified_split
from src.models.bert_classifier import CustomBERTClassifier
from src.training.metrics import calculate_metrics
from src.training.trainer import (
    compute_dynamic_class_weights,
    evaluate_epoch,
    train_all_aspects,
    train_aspect_model,
)


def test_metrics_calculation():
    """Verifikasi perhitungan metrik Macro-F1 dan Active Macro-F1."""
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]

    metrics = calculate_metrics(y_true, y_pred)

    assert "macro_f1" in metrics
    assert "active_macro_f1" in metrics
    assert "f1_per_class" in metrics
    assert "confusion_matrix" in metrics

    # Pastikan Active Macro-F1 hanya merata-ratakan kelas 1 (pos) dan 2 (neg)
    f1_pos = metrics["f1_per_class"]["pos"]
    f1_neg = metrics["f1_per_class"]["neg"]
    expected_active_f1 = (f1_pos + f1_neg) / 2.0
    assert pytest.approx(metrics["active_macro_f1"], rel=1e-4) == expected_active_f1

    # Format confusion matrix 3x3
    cm = metrics["confusion_matrix"]
    assert len(cm) == 3
    assert len(cm[0]) == 3


def test_metrics_defensive_flattening():
    """Verifikasi bahwa input 2D (misal dari torch.unsqueeze atau column vector) diratakan aman."""
    y_true_2d = torch.tensor([[0], [1], [2], [1]])
    y_pred_2d = torch.tensor([[0], [2], [2], [1]])

    metrics = calculate_metrics(y_true_2d, y_pred_2d)
    assert isinstance(metrics["macro_f1"], float)
    assert isinstance(metrics["active_macro_f1"], float)


def test_dynamic_class_weights():
    """Verifikasi bobot kelas dinamis memberikan penalti lebih tinggi pada kelas minoritas."""
    # Kelas 0 dominan (80%), kelas 1 (10%), kelas 2 (10%)
    labels = [0] * 80 + [1] * 10 + [2] * 10
    weights = compute_dynamic_class_weights(labels, device="cpu")

    assert weights.shape == torch.Size([3])
    # Bobot kelas 0 harus jauh lebih kecil dari kelas 1 dan 2
    assert weights[0] < weights[1]
    assert weights[0] < weights[2]
    # Rata-rata bobot ternormalisasi mendekati 1.0
    assert pytest.approx(weights.mean().item(), rel=1e-4) == 1.0


def test_bert_classifier_forward():
    """Verifikasi forward pass CustomBERTClassifier menghasilkan logits berdimensi (batch_size, 3)."""
    model = CustomBERTClassifier()
    model.eval()

    dummy_input_ids = torch.randint(0, 1000, (2, 16), dtype=torch.long)
    dummy_attention_mask = torch.ones((2, 16), dtype=torch.long)

    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)

    assert logits.shape == torch.Size([2, 3])
    assert not torch.isnan(logits).any()


def test_training_smoke_and_checkpoint_reload(tmp_path: Path):
    """Smoke test training loop 1 epoch pada CPU, verifikasi checkpoint & reload inferensi."""
    gt_csv = tmp_path / "ground_truth.csv"
    split_json = tmp_path / "split_indices.json"
    models_dir = tmp_path / "models"

    # 1. Setup mock data & split
    generate_mock_ground_truth(output_path=gt_csv, n_samples=32, seed=42)
    create_stratified_split(ground_truth_path=gt_csv, output_split_path=split_json, seed=42)

    # 2. Jalankan training model gameplay 1 epoch pada CPU
    summary = train_aspect_model(
        aspect_name="gameplay",
        ground_truth_path=gt_csv,
        split_indices_path=split_json,
        models_dir=models_dir,
        epochs=1,
        batch_size=8,
        lr=2e-5,
        device="cpu",
    )

    # 3. Verifikasi berkas keluaran
    checkpoint_file = models_dir / "bert_gameplay" / "best_model.pt"
    summary_file = models_dir / "bert_gameplay" / "training_summary.json"

    assert checkpoint_file.exists(), f"Checkpoint {checkpoint_file} tidak ditemukan!"
    assert summary_file.exists(), f"Summary {summary_file} tidak ditemukan!"

    with open(summary_file, "r", encoding="utf-8") as f:
        saved_summary = json.load(f)

    assert saved_summary["aspect"] == "gameplay"
    assert "best_active_f1" in saved_summary
    assert isinstance(saved_summary["best_active_f1"], float)
    assert saved_summary["best_active_f1"] >= 0.0

    # 4. Verifikasi reload checkpoint dan inferensi
    reloaded_model = CustomBERTClassifier()
    state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    reloaded_model.load_state_dict(state_dict)
    reloaded_model.eval()

    test_input = torch.randint(0, 1000, (1, 128), dtype=torch.long)
    test_mask = torch.ones((1, 128), dtype=torch.long)

    with torch.no_grad():
        test_logits = reloaded_model(test_input, test_mask)

    assert test_logits.shape == torch.Size([1, 3])
    predicted_class = torch.argmax(test_logits, dim=1).item()
    assert predicted_class in {0, 1, 2}


def test_train_all_aspects_cleans_memory_after_training_failure(monkeypatch, tmp_path: Path):
    events = []

    def fail_on_second_aspect(*, aspect_name, **kwargs):
        events.append(("train", aspect_name))
        if aspect_name == "story":
            raise RuntimeError("training failed")
        return {"aspect": aspect_name}

    monkeypatch.setattr(trainer, "train_aspect_model", fail_on_second_aspect)
    monkeypatch.setattr(trainer.gc, "collect", lambda: events.append(("gc", None)))
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        trainer.torch.cuda, "empty_cache", lambda: events.append(("cache", None))
    )

    with pytest.raises(RuntimeError, match="training failed"):
        trainer.train_all_aspects(
            ground_truth_path=tmp_path / "ground_truth.csv",
            split_indices_path=tmp_path / "split_indices.json",
            device="cpu",
        )

    assert events == [
        ("train", "gameplay"),
        ("gc", None),
        ("cache", None),
        ("train", "story"),
        ("gc", None),
        ("cache", None),
    ]


def test_train_all_aspects_order_and_memory_cleanup(monkeypatch, tmp_path: Path):
    """Verify orchestration order and cleanup between aspects without training BERT."""
    events = []

    def fake_train_aspect_model(*, aspect_name, **kwargs):
        assert kwargs["device"] == "cpu"
        events.append(("train", aspect_name))
        return {"aspect": aspect_name}

    monkeypatch.setattr(trainer, "train_aspect_model", fake_train_aspect_model)
    monkeypatch.setattr(trainer.gc, "collect", lambda: events.append(("gc", None)))
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        trainer.torch.cuda, "empty_cache", lambda: events.append(("cache", None))
    )

    summaries = trainer.train_all_aspects(
        ground_truth_path=tmp_path / "ground_truth.csv",
        split_indices_path=tmp_path / "split_indices.json",
        device="cpu",
    )

    assert events == [
        ("train", "gameplay"),
        ("gc", None),
        ("cache", None),
        ("train", "story"),
        ("gc", None),
        ("cache", None),
        ("train", "technical"),
        ("gc", None),
        ("cache", None),
    ]
    assert summaries == {
        "gameplay": {"aspect": "gameplay"},
        "story": {"aspect": "story"},
        "technical": {"aspect": "technical"},
    }


def test_training_rejects_non_positive_epochs(tmp_path: Path):
    """Verifikasi bahwa train_aspect_model dan train_all_aspects menolak epochs <= 0."""
    gt_csv = tmp_path / "gt.csv"
    split_json = tmp_path / "split.json"
    generate_mock_ground_truth(output_path=gt_csv, n_samples=20, seed=42)
    create_stratified_split(gt_csv, split_json, seed=42)

    with pytest.raises(ValueError, match="epochs harus bernilai integer positif > 0"):
        train_aspect_model(
            aspect_name="gameplay",
            ground_truth_path=gt_csv,
            split_indices_path=split_json,
            models_dir=tmp_path / "models",
            epochs=0,
        )

    with pytest.raises(ValueError, match="epochs harus bernilai integer positif > 0"):
        train_all_aspects(
            ground_truth_path=gt_csv,
            split_indices_path=split_json,
            models_dir=tmp_path / "models",
            epochs=-1,
        )


def test_evaluate_epoch_sample_weighted_loss():
    """Verifikasi bahwa evaluate_epoch membobotkan loss berdasarkan jumlah sampel per batch."""
    from torch.utils.data import DataLoader, Dataset
    import torch.nn as nn

    class DummyModel(nn.Module):
        def forward(self, input_ids, attention_mask=None):
            return input_ids

    class DummyDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    items = [
        {"input_ids": torch.tensor([5.0, 0.0, 0.0]), "attention_mask": torch.ones(3), "labels": torch.tensor(0)},
        {"input_ids": torch.tensor([5.0, 0.0, 0.0]), "attention_mask": torch.ones(3), "labels": torch.tensor(0)},
        {"input_ids": torch.tensor([0.0, 5.0, 0.0]), "attention_mask": torch.ones(3), "labels": torch.tensor(0)},
    ]
    loader = DataLoader(DummyDataset(items), batch_size=2, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    model = DummyModel()

    avg_loss, _ = evaluate_epoch(model, loader, criterion, torch.device("cpu"))
    l1 = criterion(torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), torch.tensor([0, 0])).item()
    l2 = criterion(torch.tensor([[0.0, 5.0, 0.0]]), torch.tensor([0])).item()
    expected_weighted = (l1 * 2 + l2 * 1) / 3.0
    assert pytest.approx(avg_loss, rel=1e-5) == expected_weighted

