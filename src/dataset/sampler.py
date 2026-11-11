"""Sampling and clean corpus construction for Steam reviews."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.ingestion.cleaner import clean_review_text, is_valid_review

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetSampler")


def build_clean_corpus(
    raw_dir: str | Path = "data/raw",
    output_path: str | Path = "data/clean/corpus_clean.csv",
) -> pd.DataFrame:
    """Membaca seluruh berkas JSONL mentah, membersihkan teks, memfilter ulasan,

    dan menyimpan korpus bersih terdeduplikasi.

    Args:
        raw_dir: Direktori berisi berkas-berkas .jsonl ulasan mentah.
        output_path: Path penyimpanan berkas CSV korpus bersih.

    Returns:
        pd.DataFrame dengan kolom [review_id, app_id, clean_text, voted_up].
    """
    raw_path = Path(raw_dir)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(raw_path.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("Tidak ditemukan berkas .jsonl di direktori %s", raw_path)
        empty_df = pd.DataFrame(columns=["review_id", "app_id", "clean_text", "voted_up"])
        empty_df.to_csv(output_file, index=False, encoding="utf-8")
        return empty_df

    total_raw_reviews = 0
    filtered_out_reviews = 0
    seen_review_ids = set()
    cleaned_rows: List[dict] = []

    for file_path in jsonl_files:
        logger.info("Memproses berkas mentah: %s", file_path.name)
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue

                total_raw_reviews += 1
                try:
                    record = json.loads(line_str)
                except json.JSONDecodeError:
                    filtered_out_reviews += 1
                    continue

                if not isinstance(record, dict):
                    filtered_out_reviews += 1
                    continue

                app_id_raw = record.get("app_id")
                if (
                    isinstance(app_id_raw, bool)
                    or not isinstance(app_id_raw, (int, str))
                    or not str(app_id_raw).strip()
                ):
                    filtered_out_reviews += 1
                    continue
                try:
                    app_id = int(app_id_raw)
                except ValueError:
                    filtered_out_reviews += 1
                    continue
                if app_id <= 0:
                    filtered_out_reviews += 1
                    continue

                review_id = str(record.get("review_id", ""))
                if not review_id or review_id in seen_review_ids:
                    filtered_out_reviews += 1
                    continue
                seen_review_ids.add(review_id)

                raw_text = str(record.get("review_text", ""))
                clean_text = clean_review_text(raw_text)

                if not is_valid_review(clean_text):
                    filtered_out_reviews += 1
                    continue

                cleaned_rows.append(
                    {
                        "review_id": review_id,
                        "app_id": app_id,
                        "clean_text": clean_text,
                        "voted_up": bool(record.get("voted_up", False)),
                    }
                )

    clean_df = pd.DataFrame(cleaned_rows, columns=["review_id", "app_id", "clean_text", "voted_up"])
    clean_df.to_csv(output_file, index=False, encoding="utf-8")

    logger.info(
        "Pembersihan selesai: Total mentah: %d | Tersaring: %d | Tersimpan bersih: %d -> %s",
        total_raw_reviews,
        filtered_out_reviews,
        len(clean_df),
        output_file,
    )
    return clean_df


def generate_annotation_sample(
    corpus_path: str | Path = "data/clean/corpus_clean.csv",
    output_path: str | Path = "data/annotation/sample_anotasi.csv",
    n_samples: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Mengambil sampel ulasan terstratifikasi berdasarkan app_id untuk dianotasi manual.

    Skema output 6 kolom:
    review_id, app_id, clean_text, gameplay, story, technical

    Args:
        corpus_path: Path berkas CSV korpus bersih.
        output_path: Path tujuan CSV sampel anotasi (disimpan utf-8-sig).
        n_samples: Target jumlah ulasan yang disampel (default 1500).
        seed: Random seed untuk determinisme.

    Returns:
        pd.DataFrame sampel anotasi dengan kolom target berisikan string kosong "".
    """
    if n_samples <= 0:
        raise ValueError("n_samples harus lebih besar dari 0")

    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"Berkas korpus bersih tidak ditemukan: {corpus_file}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(corpus_file, dtype={"review_id": str, "app_id": int, "clean_text": str})

    if len(df) == 0:
        logger.warning("Korpus bersih kosong!")
        sample_df = pd.DataFrame(
            columns=["review_id", "app_id", "clean_text", "gameplay", "story", "technical"]
        )
        sample_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        return sample_df

    if len(df) <= n_samples:
        logger.warning(
            "Jumlah ulasan bersih (%d) <= target n_samples (%d). Mengambil seluruh data.",
            len(df),
            n_samples,
        )
        sampled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        # Stratified sampling terdistribusi proporsional antar app_id
        game_counts = df["app_id"].value_counts()
        quotas = (game_counts / len(df) * n_samples).round().astype(int)

        # Penyesuaian selisih akibat pembulatan
        diff = n_samples - quotas.sum()
        for app in game_counts.index:
            if diff == 0:
                break
            if diff > 0:
                quotas[app] += 1
                diff -= 1
            elif diff < 0 and quotas[app] > 0:
                quotas[app] -= 1
                diff += 1

        sampled_chunks: List[pd.DataFrame] = []
        for app_id, group in df.groupby("app_id", sort=False):
            q = min(len(group), quotas.get(app_id, 0))
            if q > 0:
                sampled_chunks.append(group.sample(n=q, random_state=seed))

        combined = pd.concat(sampled_chunks)

        # Jika masih ada kekurangan kuota karena game kecil
        if len(combined) < n_samples:
            remaining_ids = set(df.index) - set(combined.index)
            if remaining_ids:
                needed = n_samples - len(combined)
                filler = df.loc[list(remaining_ids)].sample(
                    n=min(needed, len(remaining_ids)), random_state=seed
                )
                combined = pd.concat([combined, filler])

        # Proteksi eksplisit jika combined melebihi target n_samples
        if len(combined) > n_samples:
            combined = combined.sample(n=n_samples, random_state=seed)

        sampled = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Buat skema anotasi 6 kolom
    sample_df = pd.DataFrame(
        {
            "review_id": sampled["review_id"],
            "app_id": sampled["app_id"],
            "clean_text": sampled["clean_text"],
            "gameplay": "",
            "story": "",
            "technical": "",
        }
    )

    sample_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info(
        "Sampel anotasi berhasil dibuat: %d baris (seed: %d) -> %s",
        len(sample_df),
        seed,
        output_file,
    )
    return sample_df
