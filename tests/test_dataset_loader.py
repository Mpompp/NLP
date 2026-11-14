"""Tests untuk verifikasi zero-leakage split, PyTorch Dataset, dan DataLoaders per aspek."""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from src.dataset.loader import SteamAspectDataset, get_dataloaders
from src.dataset.mock import generate_mock_ground_truth
from src.dataset.split import create_stratified_split


@pytest.fixture(scope="module")
def shared_tokenizer():
    """Fixture tokenizer yang di-load sekali untuk menghemat waktu komputasi."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def test_mock_generation_and_zero_leakage_split(tmp_path: Path):
    """Verifikasi pembuatan mock ground truth dan garansi partisi zero-leakage 70/15/15."""
    gt_csv = tmp_path / "ground_truth_test.csv"
    split_json = tmp_path / "split_indices.json"

    # 1. Generate 100 baris mock ground truth
    df = generate_mock_ground_truth(output_path=gt_csv, n_samples=100, seed=42)
    assert len(df) == 100
    assert list(df.columns) == ["review_id", "app_id", "clean_text", "gameplay", "story", "technical"]

    # 2. Lakukan partisi terstratifikasi
    splits = create_stratified_split(
        ground_truth_path=gt_csv,
        output_split_path=split_json,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_ids = set(splits["train_ids"])
    val_ids = set(splits["val_ids"])
    test_ids = set(splits["test_ids"])

    # 3. Assert zero-leakage
    assert len(train_ids & val_ids) == 0, "Ditemukan data leakage Train vs Val!"
    assert len(train_ids & test_ids) == 0, "Ditemukan data leakage Train vs Test!"
    assert len(val_ids & test_ids) == 0, "Ditemukan data leakage Val vs Test!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == 100

    # 4. Proporsi split
    assert len(train_ids) == 70
    assert len(val_ids) == 15
    assert len(test_ids) == 15


def test_dataloaders_shapes_and_aspects(tmp_path: Path, shared_tokenizer):
    """Verifikasi shape tensor, tipe data, dan independensi 3 aspek (gameplay, story, technical)."""
    gt_csv = tmp_path / "ground_truth.csv"
    split_json = tmp_path / "split.json"

    generate_mock_ground_truth(output_path=gt_csv, n_samples=100, seed=42)
    create_stratified_split(ground_truth_path=gt_csv, output_split_path=split_json, seed=42)

    batch_size = 16
    max_length = 128

    for aspect in ["gameplay", "story", "technical"]:
        loaders = get_dataloaders(
            ground_truth_path=gt_csv,
            split_indices_path=split_json,
            aspect_name=aspect,
            batch_size=batch_size,
            max_length=max_length,
            tokenizer=shared_tokenizer,
        )

        assert "train" in loaders and "val" in loaders and "test" in loaders

        train_loader = loaders["train"]
        first_batch = next(iter(train_loader))

        # Shape tensor verifications
        assert first_batch["input_ids"].shape == (batch_size, max_length)
        assert first_batch["attention_mask"].shape == (batch_size, max_length)
        assert first_batch["labels"].shape == (batch_size,)

        # Dtype verifications
        assert first_batch["input_ids"].dtype == torch.int64
        assert first_batch["attention_mask"].dtype == torch.int64
        assert first_batch["labels"].dtype == torch.int64

        # Value domain verification: {0, 1, 2}
        label_values = set(first_batch["labels"].tolist())
        assert label_values.issubset({0, 1, 2}), f"Label tidak valid ditemukan: {label_values}"


def test_extreme_long_text_tokenization(shared_tokenizer):
    """Verifikasi bahwa teks ulasan super panjang (> 5.000 karakter) berhasil dipotong aman."""
    super_long_text = "This is a detailed analysis of gameplay mechanics. " * 150  # ~7.650 karakter
    assert len(super_long_text) > 5000

    dataset = SteamAspectDataset(
        texts=[super_long_text],
        labels=[2],
        tokenizer=shared_tokenizer,
        max_length=128,
    )

    item = dataset[0]

    assert item["input_ids"].shape == torch.Size([128])
    assert item["attention_mask"].shape == torch.Size([128])
    assert item["labels"].item() == 2
    assert item["input_ids"].dtype == torch.int64
    assert item["attention_mask"].dtype == torch.int64


def test_invalid_aspect_name_raises_value_error(tmp_path: Path):
    """Verifikasi bahwa nama aspek yang tidak valid melempar ValueError."""
    gt_csv = tmp_path / "gt.csv"
    split_json = tmp_path / "sp.json"
    generate_mock_ground_truth(output_path=gt_csv, n_samples=20, seed=42)
    create_stratified_split(gt_csv, split_json, seed=42)

    with pytest.raises(ValueError, match="tidak valid"):
        get_dataloaders(gt_csv, split_json, aspect_name="audio")


def test_split_index_mismatch_raises_value_error(tmp_path: Path):
    """Verifikasi bahwa jika indeks split tidak cocok dengan CSV, dilempar ValueError."""
    import json
    gt_csv = tmp_path / "gt.csv"
    split_json = tmp_path / "sp.json"
    generate_mock_ground_truth(output_path=gt_csv, n_samples=20, seed=42)

    # Simpan indeks split yang berisi ID asing
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump({"train_ids": ["non_existent_1"], "val_ids": ["non_existent_2"], "test_ids": ["non_existent_3"]}, f)

    with pytest.raises(ValueError, match="Split index mismatch"):
        get_dataloaders(gt_csv, split_json, aspect_name="gameplay")


def test_split_rejects_missing_label_column_with_row_indices(tmp_path: Path):
    import pandas as pd

    gt_csv = tmp_path / "missing_label.csv"
    split_json = tmp_path / "split.json"
    df = generate_mock_ground_truth(output_path=gt_csv, n_samples=20, seed=42)
    df.drop(columns="story").to_csv(gt_csv, index=False)

    with pytest.raises(ValueError) as exc_info:
        create_stratified_split(gt_csv, split_json, seed=42)

    assert "story" in str(exc_info.value)
    assert "indeks baris terdampak: [0, 1" in str(exc_info.value)


def test_split_rejects_boolean_labels_with_row_indices(tmp_path: Path):
    import pandas as pd

    gt_csv = tmp_path / "boolean_labels.csv"
    split_json = tmp_path / "split.json"
    rows = [
        {
            "review_id": f"review_{i}",
            "app_id": 1,
            "clean_text": "review text",
            "gameplay": bool(i % 2),
            "story": bool((i // 2) % 2),
            "technical": bool((i // 4) % 2),
        }
        for i in range(20)
    ]
    pd.DataFrame(rows).to_csv(gt_csv, index=False)

    with pytest.raises(ValueError, match="integer non-null") as exc_info:
        create_stratified_split(gt_csv, split_json, seed=42)

    assert "gameplay: [0, 1" in str(exc_info.value)


def test_split_rejects_null_and_invalid_labels_with_row_indices(tmp_path: Path):
    gt_csv = tmp_path / "invalid_labels.csv"
    split_json = tmp_path / "split.json"
    df = generate_mock_ground_truth(output_path=gt_csv, n_samples=20, seed=42)
    df.loc[0, "gameplay"] = None
    df.loc[1, "gameplay"] = 3
    df.to_csv(gt_csv, index=False)

    with pytest.raises(ValueError) as exc_info:
        create_stratified_split(gt_csv, split_json, seed=42)

    assert "gameplay: [0, 1]" in str(exc_info.value)
    assert "integer non-null" in str(exc_info.value)


@pytest.mark.parametrize("invalid_label", [1.5, "1.5", "not-an-integer", 3, -1])
def test_split_rejects_non_integer_and_out_of_range_labels_with_row_indices(
    tmp_path: Path, invalid_label
):
    gt_csv = tmp_path / "invalid_label.csv"
    split_json = tmp_path / "split.json"
    df = generate_mock_ground_truth(output_path=gt_csv, n_samples=20, seed=42)
    df["gameplay"] = df["gameplay"].astype(object)
    df.loc[0, "gameplay"] = invalid_label
    df.to_csv(gt_csv, index=False)

    with pytest.raises(ValueError) as exc_info:
        create_stratified_split(gt_csv, split_json, seed=42)

    assert "gameplay: [0]" in str(exc_info.value)
    assert "integer non-null" in str(exc_info.value)


def test_dataloaders_reject_invalid_aspect_labels_before_tokenization(tmp_path: Path):
    import pandas as pd
    import json

    gt_csv = tmp_path / "invalid_loader_labels.csv"
    split_json = tmp_path / "split.json"
    df = generate_mock_ground_truth(output_path=gt_csv, n_samples=30, seed=42)
    df.loc[0, "gameplay"] = 3
    df.to_csv(gt_csv, index=False)
    row_ids = df["review_id"].astype(str).tolist()
    split_json.write_text(
        json.dumps(
            {
                "train_ids": row_ids[:20],
                "val_ids": row_ids[20:25],
                "test_ids": row_ids[25:],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="integer non-null"):
        get_dataloaders(
            ground_truth_path=gt_csv,
            split_indices_path=split_json,
            aspect_name="gameplay",
            tokenizer=object(),
        )


def test_stratified_split_rare_class_fallback(tmp_path: Path):
    """Verifikasi bahwa dataset dengan kelas single-occurrence (< 2 sampel) tidak crash dan aman terbagi."""
    import pandas as pd
    gt_csv = tmp_path / "rare_gt.csv"
    split_json = tmp_path / "rare_sp.json"

    # 10 baris dengan 1 baris berlabel unik yang mustahil dipasangkan
    rows = []
    for i in range(9):
        rows.append({"review_id": f"rev_{i}", "app_id": 100, "clean_text": "Good game to play", "gameplay": 0, "story": 0, "technical": 0})
    rows.append({"review_id": "rev_rare", "app_id": 100, "clean_text": "Rare bug encountered", "gameplay": 2, "story": 1, "technical": 2})

    pd.DataFrame(rows).to_csv(gt_csv, index=False)

    # Harus sukses tanpa melempar ValueError dari train_test_split
    splits = create_stratified_split(gt_csv, split_json, seed=42)
    assert len(splits["train_ids"]) + len(splits["val_ids"]) + len(splits["test_ids"]) == 10


def test_split_rejects_overlapping_split_indices(tmp_path: Path):
    """Verifikasi bahwa loader menolak split_indices dengan ID yang tumpang tindih."""
    gt_csv = tmp_path / "gt.csv"
    split_json = tmp_path / "split_indices.json"
    generate_mock_ground_truth(output_path=gt_csv, n_samples=20, seed=42)

    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_ids": ["rev_0", "rev_1"],
                "val_ids": ["rev_1", "rev_2"],
                "test_ids": ["rev_3"],
            },
            f,
        )

    with pytest.raises(ValueError, match="Data leakage terdeteksi"):
        get_dataloaders(gt_csv, split_json, aspect_name="gameplay")

