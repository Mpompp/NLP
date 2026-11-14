"""Evaluation metrics for Aspect-Category Sentiment Analysis (ACSA)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def calculate_metrics(
    y_true: Union[Sequence[int], np.ndarray, torch.Tensor],
    y_pred: Union[Sequence[int], np.ndarray, torch.Tensor],
) -> Dict[str, Any]:
    """Menghitung metrik klasifikasi 3-kelas dengan fokus pada Active Macro-F1 (kelas pos dan neg).

    Args:
        y_true: Ground truth labels (0: none, 1: pos, 2: neg).
        y_pred: Predicted class labels (0: none, 1: pos, 2: neg).

    Returns:
        Dict berisi macro_f1, active_macro_f1, f1_per_class, precision_per_class,
        recall_per_class, dan confusion_matrix.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().ravel()
    else:
        y_true = np.asarray(y_true, dtype=int).ravel()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy().ravel()
    else:
        y_pred = np.asarray(y_pred, dtype=int).ravel()

    # Hitung precision, recall, f1 per kelas untuk kelas 0, 1, dan 2
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0))

    # Active Macro-F1: rata-rata F1 HANYA kelas 1 (pos) dan kelas 2 (neg)
    active_macro_f1 = float(np.mean([f1[1], f1[2]]))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    return {
        "macro_f1": macro_f1,
        "active_macro_f1": active_macro_f1,
        "f1_per_class": {
            0: float(f1[0]),
            1: float(f1[1]),
            2: float(f1[2]),
            "none": float(f1[0]),
            "pos": float(f1[1]),
            "neg": float(f1[2]),
        },
        "precision_per_class": {
            0: float(precision[0]),
            1: float(precision[1]),
            2: float(precision[2]),
            "none": float(precision[0]),
            "pos": float(precision[1]),
            "neg": float(precision[2]),
        },
        "recall_per_class": {
            0: float(recall[0]),
            1: float(recall[1]),
            2: float(recall[2]),
            "none": float(recall[0]),
            "pos": float(recall[1]),
            "neg": float(recall[2]),
        },
        "confusion_matrix": cm,
    }
