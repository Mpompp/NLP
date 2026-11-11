"""PyTorch Dataset and DataLoader modular per-aspect for Steam ACSA."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetLoader")

VALID_ASPECTS = ("gameplay", "story", "technical")


def validate_aspect_labels(labels: pd.Series, aspect_name: str) -> List[int]:
    numeric_labels = pd.to_numeric(labels, errors="coerce")
    boolean_labels = labels.astype(str).str.strip().str.lower().isin(("true", "false"))
    invalid = (
        boolean_labels
        | numeric_labels.isna()
        | ~numeric_labels.isin((0, 1, 2))
        | (numeric_labels % 1 != 0)
    )
    if invalid.any():
        raise ValueError(
            f"Label aspek '{aspect_name}' harus berupa integer non-null dalam {{0,1,2}}; "
            f"indeks baris tidak valid: {labels.index[invalid].tolist()}"
        )
    return numeric_labels.astype(int).tolist()


class SteamAspectDataset(Dataset):
    """PyTorch Dataset untuk Aspect-Category Sentiment Analysis (ACSA) satu aspek."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.texts = list(texts)
        self.labels = [int(lbl) for lbl in labels]
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

        if len(self.texts) != len(self.labels):
            raise ValueError(
                f"Jumlah teks ({len(self.texts)}) dan label ({len(self.labels)}) tidak sama!"
            )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0).long(),
            "attention_mask": encoding["attention_mask"].squeeze(0).long(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_dataloaders(
    ground_truth_path: str | Path,
    split_indices_path: str | Path,
    aspect_name: str = "gameplay",
    batch_size: int = 16,
    max_length: int = 128,
    tokenizer: Optional[Any] = None,
) -> Dict[str, DataLoader]:
    """Menyiapkan train, val, dan test DataLoaders untuk satu aspek tertentu.

    Args:
        ground_truth_path: Path berkas CSV ground truth.
        split_indices_path: Path berkas JSON indeks split.
        aspect_name: Nama aspek target ('gameplay', 'story', atau 'technical').
        batch_size: Ukuran batch untuk DataLoader (default 16).
        max_length: Panjang tokenisasi maksimal (default 128).
        tokenizer: Pretrained tokenizer Hugging Face (default bert-base-uncased).

    Returns:
        Dict berisikan {"train": train_loader, "val": val_loader, "test": test_loader}.
    """
    if aspect_name not in VALID_ASPECTS:
        raise ValueError(
            f"Aspek '{aspect_name}' tidak valid! Wajib salah satu dari: {VALID_ASPECTS}"
        )

    gt_file = Path(ground_truth_path)
    split_file = Path(split_indices_path)

    if not gt_file.exists():
        raise FileNotFoundError(f"Berkas ground truth tidak ditemukan: {gt_file}")
    if not split_file.exists():
        raise FileNotFoundError(f"Berkas indeks split tidak ditemukan: {split_file}")

    # 1. Load data dan split indices
    df = pd.read_csv(gt_file, dtype={"review_id": str})
    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = set(splits.get("train_ids", []))
    val_ids = set(splits.get("val_ids", []))
    test_ids = set(splits.get("test_ids", []))

    if not (train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)):
        raise ValueError(
            "Data leakage terdeteksi: Terdapat ID ulasan yang tumpang tindih antara Train, Val, atau Test di split_indices.json!"
        )

    df["review_id"] = df["review_id"].astype(str)

    train_df = df[df["review_id"].isin(train_ids)].copy()
    val_df = df[df["review_id"].isin(val_ids)].copy()
    test_df = df[df["review_id"].isin(test_ids)].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Split index mismatch! Ditemukan 0 kecocokan review_id di ground truth CSV "
            f"(Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)})."
        )

    train_labels = validate_aspect_labels(train_df[aspect_name], aspect_name)
    val_labels = validate_aspect_labels(val_df[aspect_name], aspect_name)
    test_labels = validate_aspect_labels(test_df[aspect_name], aspect_name)

    # 2. Inisialisasi tokenizer jika tidak disediakan
    if tokenizer is None:
        logger.info("Menginisialisasi AutoTokenizer bert-base-uncased...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 3. Bentuk Datasets
    train_dataset = SteamAspectDataset(
        texts=train_df["clean_text"].fillna("").tolist(),
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = SteamAspectDataset(
        texts=val_df["clean_text"].fillna("").tolist(),
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_dataset = SteamAspectDataset(
        texts=test_df["clean_text"].fillna("").tolist(),
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # 4. Bentuk DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    logger.info(
        "DataLoaders siap untuk aspek [%s]: Train=%d, Val=%d, Test=%d (batch_size=%d, max_length=%d)",
        aspect_name,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        batch_size,
        max_length,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
