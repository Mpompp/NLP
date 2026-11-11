"""Comparative evaluation module for Linear SVM Baseline vs BERT on identical test set."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
import torch

from src.dataset.loader import VALID_ASPECTS, get_dataloaders, validate_aspect_labels
from src.training.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Evaluator")


def format_comparison_table(comparison_results: Dict[str, Any]) -> str:
    """Format hasil evaluasi komparatif menjadi tabel teks ASCII rapi."""
    header = (
        "+----------------------+-------------------+------------+-----------------+-------------------+\n"
        "| Aspect               | Metric            | Linear SVM | BERT (Proposed) | Delta (BERT - SVM)|\n"
        "+----------------------+-------------------+------------+-----------------+-------------------+"
    )
    separator = (
        "+----------------------+-------------------+------------+-----------------+-------------------+"
    )
    rows = [header]

    for aspect, data in comparison_results.items():
        svm_data = data.get("svm", {})
        bert_data = data.get("bert", {})
        delta_data = data.get("delta", {})

        svm_macro = svm_data.get("macro_f1", 0.0)
        bert_macro = bert_data.get("macro_f1", 0.0)
        delta_macro = delta_data.get("macro_f1", bert_macro - svm_macro)

        svm_act = svm_data.get("active_macro_f1", 0.0)
        bert_act = bert_data.get("active_macro_f1", 0.0)
        delta_act = delta_data.get("active_macro_f1", bert_act - svm_act)

        row_macro = (
            f"| {aspect.upper():<20} | Macro-F1          | {svm_macro:>10.4f} | {bert_macro:>15.4f} | {delta_macro:>+17.4f} |"
        )
        row_act = (
            f"| {'':<20} | Active Macro-F1   | {svm_act:>10.4f} | {bert_act:>15.4f} | {delta_act:>+17.4f} |"
        )
        rows.extend([row_macro, row_act, separator])

    return "\n".join(rows)


def evaluate_test_set_comparison(
    ground_truth_path: Union[str, Path],
    split_indices_path: Union[str, Path],
    models_dir: Union[str, Path] = "models",
    reports_dir: Union[str, Path] = "reports",
    device: Optional[Union[torch.device, str]] = None,
    aspects: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Mengevaluasi komparatif performa model Linear SVM vs BERT pada Test Set yang identik.

    Args:
        ground_truth_path: Path berkas CSV ground truth.
        split_indices_path: Path berkas JSON split_indices.
        models_dir: Direktori checkpoint model tersimpan.
        reports_dir: Direktori tempat menyimpan laporan evaluasi komparatif.
        device: Device untuk inferensi BERT (cpu atau cuda).
        aspects: List aspek yang ingin dievaluasi (default: 3 aspek VALID_ASPECTS).

    Returns:
        Dict hasil perbandingan komprehensif.
    """
    gt_file = Path(ground_truth_path)
    split_file = Path(split_indices_path)
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)

    if not gt_file.exists():
        raise FileNotFoundError(f"Berkas ground truth tidak ditemukan: {gt_file}")
    if not split_file.exists():
        raise FileNotFoundError(f"Berkas indeks split tidak ditemukan: {split_file}")

    # Lazy import to avoid circular dependency with src.models
    from src.models.baseline_svm import BaselineSVMClassifier
    from src.models.bert_classifier import CustomBERTClassifier

    # 1. Setup device
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    # 2. Muat test data yang terkunci di split_indices.json
    df = pd.read_csv(gt_file, dtype={"review_id": str})
    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_ids = set(splits.get("test_ids", []))
    df["review_id"] = df["review_id"].astype(str)
    test_df = df[df["review_id"].isin(test_ids)].copy()

    if len(test_df) == 0:
        raise ValueError("Data test set kosong berdasarkan indeks split!")

    aspect_list = list(aspects) if aspects is not None else list(VALID_ASPECTS)
    comparison_results: Dict[str, Any] = {}

    for aspect in aspect_list:
        if aspect not in VALID_ASPECTS:
            raise ValueError(
                f"Aspek '{aspect}' tidak valid! Wajib salah satu dari: {VALID_ASPECTS}"
            )

        logger.info("=== Evaluasi Komparatif Test Set: Aspek [%s] ===", aspect.upper())

        bert_ckpt = models_path / f"bert_{aspect}" / "best_model.pt"
        svm_ckpt = models_path / f"svm_{aspect}" / "model.joblib"

        if not bert_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint model BERT tidak ditemukan: {bert_ckpt}")
        if not svm_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint model SVM tidak ditemukan: {svm_ckpt}")

        # a. Evaluasi Linear SVM
        svm_model = BaselineSVMClassifier()
        svm_model.load(svm_ckpt)
        test_texts = test_df["clean_text"].fillna("").tolist()
        test_labels = validate_aspect_labels(test_df[aspect], aspect)

        svm_preds = svm_model.predict(test_texts)
        svm_metrics = calculate_metrics(test_labels, svm_preds)

        # b. Evaluasi BERT
        bert_model = CustomBERTClassifier().to(target_device)
        state_dict = torch.load(bert_ckpt, map_location=target_device, weights_only=True)
        bert_model.load_state_dict(state_dict)
        bert_model.eval()

        loaders = get_dataloaders(
            ground_truth_path=gt_file,
            split_indices_path=split_file,
            aspect_name=aspect,
            batch_size=16,
        )
        test_loader = loaders["test"]

        bert_preds: List[int] = []
        bert_targets: List[int] = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(target_device)
                attention_mask = batch["attention_mask"].to(target_device)
                labels = batch["labels"].to(target_device)

                logits = bert_model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                bert_preds.extend(preds.cpu().tolist())
                bert_targets.extend(labels.cpu().tolist())

        bert_metrics = calculate_metrics(bert_targets, bert_preds)

        # Bebaskan memori BERT setelah evaluasi aspek selesai
        del bert_model, test_loader, loaders
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

        # c. Hitung margin keunggulan BERT
        delta_macro_f1 = bert_metrics["macro_f1"] - svm_metrics["macro_f1"]
        delta_active_macro_f1 = (
            bert_metrics["active_macro_f1"] - svm_metrics["active_macro_f1"]
        )

        comparison_results[aspect] = {
            "svm": {
                "macro_f1": svm_metrics["macro_f1"],
                "active_macro_f1": svm_metrics["active_macro_f1"],
                "metrics": svm_metrics,
            },
            "bert": {
                "macro_f1": bert_metrics["macro_f1"],
                "active_macro_f1": bert_metrics["active_macro_f1"],
                "metrics": bert_metrics,
            },
            "delta": {
                "macro_f1": delta_macro_f1,
                "active_macro_f1": delta_active_macro_f1,
            },
        }

    # 3. Simpan laporan perbandingan komparatif ke file JSON (merge jika file sudah ada)
    reports_path.mkdir(parents=True, exist_ok=True)
    report_file = reports_path / "evaluation_comparison.json"

    full_report_data: Dict[str, Any] = {}
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                full_report_data = json.load(f)
        except Exception as e:
            logger.warning("Gagal membaca laporan evaluasi sebelumnya (%s), membuat baru.", e)

    full_report_data.update(comparison_results)

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report_data, f, indent=2)

    logger.info("Laporan evaluasi komparatif disimpan ke %s", report_file)

    # 4. Format & cetak tabel perbandingan ASCII
    ascii_table = format_comparison_table(full_report_data)
    logger.info("\n%s", ascii_table)

    return full_report_data
