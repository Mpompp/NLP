"""Smoke test untuk memverifikasi pipeline pembersihan korpus dan sampling anotasi."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.dataset.sampler import build_clean_corpus, generate_annotation_sample


def test_sampler_pipeline_smoke(tmp_path: Path):
    """Menguji build_clean_corpus dan generate_annotation_sample menggunakan dummy data di tmp_path."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_csv = tmp_path / "clean" / "corpus_clean.csv"
    sample_csv = tmp_path / "annotation" / "sample_anotasi.csv"

    # 1. Siapkan dummy ulasan JSONL
    # Game 1: Hades (1145360)
    hades_reviews = [
        # Valid dengan BBCode dan HTML
        {
            "review_id": "rev_001",
            "app_id": 1145360,
            "review_text": "[b]Awesome[/b] gameplay and great &amp; engaging story!",
            "voted_up": True,
            "timestamp_created": 1600000001,
            "author_playtime_forever": 3600,
        },
        # ASCII art (harus tersaring)
        {
            "review_id": "rev_002",
            "app_id": 1145360,
            "review_text": "⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿",
            "voted_up": True,
            "timestamp_created": 1600000002,
            "author_playtime_forever": 60,
        },
        # Terlalu pendek < 3 kata (harus tersaring)
        {
            "review_id": "rev_003",
            "app_id": 1145360,
            "review_text": "Good game",
            "voted_up": True,
            "timestamp_created": 1600000003,
            "author_playtime_forever": 120,
        },
        # Duplikat dari rev_001 (harus tersaring)
        {
            "review_id": "rev_001",
            "app_id": 1145360,
            "review_text": "[b]Awesome[/b] gameplay and great &amp; engaging story!",
            "voted_up": True,
            "timestamp_created": 1600000001,
            "author_playtime_forever": 3600,
        },
        # Valid kedua
        {
            "review_id": "rev_004",
            "app_id": 1145360,
            "review_text": "Combat mechanics are fast paced, highly responsive controls.",
            "voted_up": True,
            "timestamp_created": 1600000004,
            "author_playtime_forever": 7200,
        },
    ]

    # Game 2: Celeste (504230)
    celeste_reviews = [
        # Valid ketiga
        {
            "review_id": "rev_005",
            "app_id": 504230,
            "review_text": "The platforming difficulty is brutal but fair and rewarding.",
            "voted_up": True,
            "timestamp_created": 1600000005,
            "author_playtime_forever": 4500,
        },
        # Valid keempat
        {
            "review_id": "rev_006",
            "app_id": 504230,
            "review_text": "Beautiful pixel art accompanied by an emotional and touching soundtrack.",
            "voted_up": True,
            "timestamp_created": 1600000006,
            "author_playtime_forever": 5100,
        },
    ]

    # Tulis berkas dummy JSONL
    with open(raw_dir / "1145360.jsonl", "w", encoding="utf-8") as f:
        for r in hades_reviews:
            f.write(json.dumps(r) + "\n")

    with open(raw_dir / "504230.jsonl", "w", encoding="utf-8") as f:
        for r in celeste_reviews:
            f.write(json.dumps(r) + "\n")

    # 2. Jalankan build_clean_corpus
    clean_df = build_clean_corpus(raw_dir=raw_dir, output_path=clean_csv)

    # Verifikasi korpus bersih
    assert clean_csv.exists(), "Berkas corpus_clean.csv tidak terbuat"
    assert list(clean_df.columns) == ["review_id", "app_id", "clean_text", "voted_up"]
    # Dari total 7 ulasan mentah: rev_002 (ascii), rev_003 (pendek), rev_001 duplikat tersaring. Sisa 4.
    assert len(clean_df) == 4, f"Diharapkan 4 ulasan bersih, didapat {len(clean_df)}"

    # Pastikan teks bersih dari BBCode dan HTML
    sample_text = clean_df[clean_df["review_id"] == "rev_001"]["clean_text"].iloc[0]
    assert "[b]" not in sample_text and "[/b]" not in sample_text
    assert "&amp;" not in sample_text and "&" in sample_text

    # 3. Jalankan generate_annotation_sample (sampling 3 ulasan terdistribusi)
    sample_df = generate_annotation_sample(
        corpus_path=clean_csv,
        output_path=sample_csv,
        n_samples=3,
        seed=42,
    )

    # Verifikasi berkas sampel anotasi
    assert sample_csv.exists(), "Berkas sample_anotasi.csv tidak terbuat"
    expected_cols = ["review_id", "app_id", "clean_text", "gameplay", "story", "technical"]
    assert list(sample_df.columns) == expected_cols, f"Kolom tidak cocok: {list(sample_df.columns)}"
    assert len(sample_df) == 3, f"Diharapkan 3 sampel, didapat {len(sample_df)}"

    # Verifikasi kolom anotasi kosong dan siap diisi
    for col in ["gameplay", "story", "technical"]:
        values = sample_df[col].fillna("").tolist()
        assert all(v == "" for v in values), f"Kolom {col} harus kosong string"

    # Verifikasi encoding utf-8-sig (memiliki BOM \xef\xbb\xbf di awal file byte)
    raw_bytes = sample_csv.read_bytes()
    assert raw_bytes.startswith(b"\xef\xbb\xbf"), "Berkas CSV tidak disimpan dengan encoding utf-8-sig"


def test_clean_corpus_filters_non_dict_records_and_invalid_app_ids(tmp_path: Path, caplog):
    caplog.set_level("INFO", logger="DatasetSampler")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_file = raw_dir / "reviews.jsonl"
    records = [
        None,
        [],
        42,
        {"review_id": "null-app", "app_id": None, "review_text": "This review should be skipped."},
        {"review_id": "bad-app", "app_id": "invalid", "review_text": "This review should be skipped."},
        {"review_id": "valid", "app_id": 123, "review_text": "This review should be included."},
    ]
    raw_file.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    clean_df = build_clean_corpus(raw_dir=raw_dir, output_path=tmp_path / "clean.csv")

    assert clean_df["review_id"].tolist() == ["valid"]
    assert clean_df["app_id"].tolist() == [123]
    assert "Tersaring: 5" in caplog.text


def test_sampling_rejects_non_positive_sample_count(tmp_path: Path):
    with pytest.raises(ValueError, match="^n_samples harus lebih besar dari 0$"):
        generate_annotation_sample(
            corpus_path=tmp_path / "missing.csv",
            output_path=tmp_path / "sample.csv",
            n_samples=0,
        )


def test_sampling_exact_row_count_when_games_exceed_samples(tmp_path: Path):
    """Menguji presisi n_samples saat jumlah game (5 game) lebih banyak dari target sampel (3 sampel)."""
    clean_csv = tmp_path / "corpus_clean_5_games.csv"
    sample_csv = tmp_path / "sample_3.csv"

    # 5 game berbeda dengan masing-masing 2 ulasan (total 10 ulasan)
    rows = []
    for g_id in [101, 102, 103, 104, 105]:
        rows.append({"review_id": f"r_{g_id}_1", "app_id": g_id, "clean_text": "A great game to play", "voted_up": True})
        rows.append({"review_id": f"r_{g_id}_2", "app_id": g_id, "clean_text": "Another fantastic indie title", "voted_up": True})

    pd.DataFrame(rows).to_csv(clean_csv, index=False, encoding="utf-8")

    sample_df = generate_annotation_sample(
        corpus_path=clean_csv,
        output_path=sample_csv,
        n_samples=3,
        seed=42,
    )

    # Harus tepat 3 baris, tidak boleh 5 baris karena pembulatan ke atas
    assert len(sample_df) == 3, f"Diharapkan tepat 3 sampel, tetapi didapat {len(sample_df)}"
