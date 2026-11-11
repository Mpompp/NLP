"""Bulk inference engine for unseen review corpus across 3 BERT aspect models."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset.loader import VALID_ASPECTS
from src.models.bert_classifier import CustomBERTClassifier

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PredictEngine")


_SPLIT_FILE_ERROR = "File split_indices.json wajib ada untuk menyaring ulasan unseen agar tidak terjadi kebocoran data ground truth!"


def predict_unseen_corpus(
    corpus_path: Union[str, Path] = "data/clean/corpus_clean.csv",
    split_indices_path: Optional[Union[str, Path]] = "data/splits/split_indices.json",
    models_dir: Union[str, Path] = "models",
    output_path: Union[str, Path] = "data/clean/corpus_predictions.csv",
    batch_size: int = 32,
    device: Optional[Union[torch.device, str]] = None,
    tokenizer: Optional[Any] = None,
) -> pd.DataFrame:
    """Menjalankan batch inference massal pada ulasan unseen dengan 3 model BERT sekuensial.

    Args:
        corpus_path: Path berkas CSV corpus bersih.
        split_indices_path: Path berkas JSON split_indices (untuk mengecualikan data train/val/test).
        models_dir: Direktori root model BERT (bert_gameplay, bert_story, bert_technical).
        output_path: Path tujuan penyimpanan CSV hasil prediksi.
        batch_size: Ukuran batch inferensi (default: 32).
        device: PyTorch device ('cpu' atau 'cuda').
        tokenizer: HuggingFace tokenizer (default: bert-base-uncased).

    Returns:
        pd.DataFrame dengan kolom [review_id, app_id, clean_text, pred_gameplay, pred_story, pred_technical].
    """
    corpus_file = Path(corpus_path)
    models_path = Path(models_dir)
    output_file = Path(output_path)

    if split_indices_path is None or not Path(split_indices_path).is_file():
        raise FileNotFoundError(_SPLIT_FILE_ERROR)
    if not corpus_file.exists():
        raise FileNotFoundError(f"Berkas korpus bersih tidak ditemukan: {corpus_file}")

    with open(split_indices_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    if not isinstance(splits, dict) or any(
        not isinstance(splits.get(key), list) or any(not isinstance(review_id, str) for review_id in splits[key])
        for key in ("train_ids", "val_ids", "test_ids")
    ):
        raise ValueError("Format split_indices.json tidak valid: train_ids, val_ids, dan test_ids harus berupa list string.")
    if len(splits.get("train_ids", [])) == 0 or len(splits.get("val_ids", [])) == 0 or len(splits.get("test_ids", [])) == 0:
        raise ValueError("Partisi train_ids, val_ids, atau test_ids pada split_indices.json tidak boleh kosong!")

    checkpoint_paths = {
        aspect: models_path / f"bert_{aspect}" / "best_model.pt" for aspect in VALID_ASPECTS
    }
    for aspect, ckpt_path in checkpoint_paths.items():
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint model BERT untuk aspek '{aspect}' tidak ditemukan: {ckpt_path}")

    # 1. Setup device
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    logger.info("Menyiapkan inferensi korpus pada device: %s", target_device)

    # 2. Baca korpus bersih
    df = pd.read_csv(corpus_file, dtype={"review_id": str})
    total_raw = len(df)
    df["review_id"] = df["review_id"].astype(str)

    # 3. Filter ulasan unseen berdasarkan indeks train/val/test
    seen_ids = set(splits["train_ids"]) | set(splits["val_ids"]) | set(splits["test_ids"])
    df = df[~df["review_id"].isin(seen_ids)].copy()
    logger.info(
        "Filter data unseen: %d terbaca, %d dieksklusikan (train/val/test), %d unseen tersisa.",
        total_raw,
        total_raw - len(df),
        len(df),
    )

    if len(df) == 0:
        logger.warning("Tidak ada ulasan unseen untuk diprediksi!")
        empty_df = pd.DataFrame(
            columns=["review_id", "app_id", "clean_text", "pred_gameplay", "pred_story", "pred_technical"]
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        return empty_df

    # 4. Inisialisasi tokenizer sekali
    if tokenizer is None:
        logger.info("Memuat AutoTokenizer bert-base-uncased...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    texts = df["clean_text"].fillna("").astype(str).tolist()

    # 5. Jalankan inferensi sekuensial per aspek
    for aspect, ckpt_path in checkpoint_paths.items():
        logger.info("Memuat model BERT untuk aspek [%s] dari %s", aspect.upper(), ckpt_path)
        model = CustomBERTClassifier().to(target_device)
        state_dict = torch.load(ckpt_path, map_location=target_device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        preds_list: List[int] = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Predict [{aspect}]", total=num_batches):
            batch_texts = texts[i : i + batch_size]
            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(target_device)
            attention_mask = encodings["attention_mask"].to(target_device)

            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().tolist()

            preds_list.extend(preds)

        df[f"pred_{aspect}"] = preds_list

        # Pembersihan VRAM eksplisit setelah selesai tiap model
        del model
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    # 6. Bentuk skema keluaran dan simpan
    output_cols = ["review_id", "app_id", "clean_text", "pred_gameplay", "pred_story", "pred_technical"]
    result_df = df[output_cols].copy()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    logger.info("Hasil inferensi massal berhasil disimpan ke %s (%d ulasan)", output_file, len(result_df))
    return result_df
