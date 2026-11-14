"""Data splitting with zero-leakage assertions and stratified composite labels."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetSplit")


def create_stratified_split(
    ground_truth_path: str | Path,
    output_split_path: str | Path = "data/splits/split_indices.json",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Membagi dataset ground truth ke partisi Train, Val, dan Test secara terstratifikasi

    dengan garansi zero-leakage.

    Args:
        ground_truth_path: Path berkas CSV ground truth berlabel.
        output_split_path: Path tujuan penyimpanan berkas JSON indeks split.
        train_ratio: Proporsi data train (default 0.70).
        val_ratio: Proporsi data validasi (default 0.15).
        test_ratio: Proporsi data uji (default 0.15).
        seed: Random seed untuk determinisme.

    Returns:
        Dict berisi {"train_ids": [...], "val_ids": [...], "test_ids": [...]}.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Total proporsi harus 1.0, didapat: {total_ratio}")

    gt_file = Path(ground_truth_path)
    if not gt_file.exists():
        raise FileNotFoundError(f"Berkas ground truth tidak ditemukan: {gt_file}")

    output_file = Path(output_split_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(gt_file, dtype={"review_id": str})
    if len(df) == 0:
        raise ValueError("Dataset ground truth kosong, tidak dapat melakukan partisi!")

    label_columns = ("gameplay", "story", "technical")
    missing_columns = [column for column in label_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Kolom label wajib tidak ditemukan: {missing_columns}; "
            f"indeks baris terdampak: {df.index.tolist()}"
        )

    invalid_labels = []
    for column in label_columns:
        numeric_labels = pd.to_numeric(df[column], errors="coerce")
        boolean_labels = df[column].astype(str).str.strip().str.lower().isin(("true", "false"))
        invalid_indices = df.index[
            boolean_labels
            | ~numeric_labels.isin((0, 1, 2))
            | (numeric_labels % 1 != 0)
        ].tolist()
        if invalid_indices:
            invalid_labels.append(f"{column}: {invalid_indices}")
    if invalid_labels:
        raise ValueError(
            "Label wajib berupa integer non-null dalam {0,1,2}; "
            f"indeks baris tidak valid: {'; '.join(invalid_labels)}"
        )

    labels = {
        column: pd.to_numeric(df[column]).astype(int).astype(str)
        for column in label_columns
    }
    composite = labels["gameplay"] + "_" + labels["story"] + "_" + labels["technical"]

    # Tangani kelas langka (< 2 kemunculan) agar stratified split tidak melempar ValueError
    counts = composite.value_counts()
    rare_classes = counts[counts < 2].index
    strat_labels = composite.copy()
    if len(rare_classes) > 0:
        strat_labels = strat_labels.replace(rare_classes, "other")

    # Jika setelah pengelompokan masih ada kelas dengan < 2 sampel, fallback aman ke unstratified
    if strat_labels.value_counts().min() < 2:
        logger.warning(
            "Terdapat kelas dengan sampel < 2 yang tidak dapat distratifikasi. "
            "Melakukan fallback ke unstratified random split untuk partisi Train."
        )
        strat_labels = None

    # 2. Split pertama: Train vs Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        stratify=strat_labels,
    )

    # 3. Split kedua: Temp dibagi menjadi Val dan Test secara proporsional
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)
    temp_strat = None
    if strat_labels is not None:
        temp_strat = strat_labels.loc[temp_df.index]
        temp_counts = temp_strat.value_counts()
        rare_temp = temp_counts[temp_counts < 2].index
        if len(rare_temp) > 0:
            temp_strat = temp_strat.replace(rare_temp, "other")
        if temp_strat.value_counts().min() < 2:
            logger.warning(
                "Partisi kedua (Val/Test) memiliki kelas < 2 sampel. "
                "Melakukan fallback ke unstratified random split."
            )
            temp_strat = None

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative_ratio,
        random_state=seed,
        stratify=temp_strat,
    )

    train_ids = [str(x) for x in train_df["review_id"].tolist()]
    val_ids = [str(x) for x in val_df["review_id"].tolist()]
    test_ids = [str(x) for x in test_df["review_id"].tolist()]

    # 4. ASSERTION CHECK WAJIB: Garansi Zero-Leakage & Integritas Partisi
    assert len(set(train_ids) & set(val_ids)) == 0, "Data leakage antara Train dan Val!"
    assert len(set(train_ids) & set(test_ids)) == 0, "Data leakage antara Train dan Test!"
    assert len(set(val_ids) & set(test_ids)) == 0, "Data leakage antara Val dan Test!"
    assert (
        len(train_ids) + len(val_ids) + len(test_ids) == len(df)
    ), "Ada ulasan yang hilang saat splitting!"

    splits_data = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    logger.info(
        "Partisi sukses: Train=%d, Val=%d, Test=%d (Total=%d) -> %s",
        len(train_ids),
        len(val_ids),
        len(test_ids),
        len(df),
        output_file,
    )
    return splits_data
