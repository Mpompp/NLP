"""Manual PyTorch training engine with dynamic class weights for Steam ACSA."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.dataset.loader import VALID_ASPECTS, get_dataloaders
from src.models.bert_classifier import CustomBERTClassifier
from src.training.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Trainer")


def compute_dynamic_class_weights(
    labels: Sequence[int],
    device: Union[torch.device, str],
) -> torch.Tensor:
    """Menghitung bobot kelas dinamis w_k = N_total / (3 * N_k) untuk menangani ketimpangan kelas 0 (none).

    Bobot dinormalisasi sehingga mean(w) = 1.0 agar gradien stabil.
    """
    labels_list = list(labels)
    n_total = len(labels_list)
    if n_total == 0:
        return torch.ones(3, dtype=torch.float32, device=device)

    weights = []
    for k in (0, 1, 2):
        n_k = labels_list.count(k)
        # Proteksi jika ada kelas dengan 0 kemunculan di subset
        n_k = max(1, n_k)
        w_k = n_total / (3.0 * n_k)
        weights.append(w_k)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalisasi agar mean = 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()

    if isinstance(device, str):
        device = torch.device(device)
    return weights_tensor.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """Satu epoch manual training loop PyTorch dengan gradient clipping."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluasi model pada validation/test set dan hitung metrik Active Macro-F1."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            batch_samples = labels.size(0)
            total_loss += loss.item() * batch_samples
            total_samples += batch_samples
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, total_samples)
    metrics = calculate_metrics(all_targets, all_preds)
    return avg_loss, metrics


def train_aspect_model(
    aspect_name: str,
    ground_truth_path: Union[str, Path],
    split_indices_path: Union[str, Path],
    models_dir: Union[str, Path] = "models",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    device: Optional[Union[torch.device, str]] = None,
) -> Dict[str, Any]:
    """Melatih satu model BERT khusus untuk satu aspek dengan checkpointing Active Macro-F1 terbaik."""
    if epochs <= 0:
        raise ValueError(f"epochs harus bernilai integer positif > 0, diterima: {epochs}")

    if aspect_name not in VALID_ASPECTS:
        raise ValueError(
            f"Aspek '{aspect_name}' tidak valid! Wajib salah satu dari: {VALID_ASPECTS}"
        )

    # 1. Kunci determinisme
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 2. Setup Device
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    logger.info("Memulai pelatihan model [%s] pada device: %s", aspect_name, target_device)

    # 3. DataLoaders
    loaders = get_dataloaders(
        ground_truth_path=ground_truth_path,
        split_indices_path=split_indices_path,
        aspect_name=aspect_name,
        batch_size=batch_size,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # 4. Dynamic Class Weights & Loss
    train_labels = [int(lbl) for lbl in train_loader.dataset.labels]
    class_weights = compute_dynamic_class_weights(train_labels, device=target_device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info("Dynamic Class Weights [%s]: %s", aspect_name, class_weights.cpu().tolist())

    # 5. Inisialisasi Model, Optimizer, dan Warmup Scheduler
    model = CustomBERTClassifier().to(target_device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(0.10 * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # 6. Direktori Checkpointing
    aspect_dir = Path(models_dir) / f"bert_{aspect_name}"
    aspect_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = aspect_dir / "best_model.pt"
    summary_path = aspect_dir / "training_summary.json"

    best_active_f1 = -1.0
    best_epoch = -1
    best_val_loss = float("inf")
    best_metrics: Dict[str, Any] = {}

    # 7. Training Loop Sekuensial & Transparan
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=target_device,
        )

        val_loss, val_metrics = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=target_device,
        )

        active_f1 = val_metrics["active_macro_f1"]
        logger.info(
            "Epoch %d/%d [%s] - Train Loss: %.4f, Val Loss: %.4f, Macro-F1: %.4f, Active Macro-F1: %.4f",
            epoch,
            epochs,
            aspect_name,
            train_loss,
            val_loss,
            val_metrics["macro_f1"],
            active_f1,
        )

        # Checkpointing berbasis Active Macro-F1 dengan tie-breaker validation loss
        is_better = (active_f1 > best_active_f1) or (
            np.isclose(active_f1, best_active_f1) and val_loss < best_val_loss
        )
        if is_better:
            best_active_f1 = active_f1
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = val_metrics
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                "Checkpoint tersimpan -> %s (Active Macro-F1: %.4f, Val Loss: %.4f)",
                best_model_path,
                best_active_f1,
                best_val_loss,
            )

        # Wajib bersihkan VRAM GPU di akhir epoch evaluasi
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    # 8. Simpan ringkasan pelatihan
    summary = {
        "aspect": aspect_name,
        "best_epoch": best_epoch,
        "best_active_f1": best_active_f1,
        "best_val_loss": best_val_loss,
        "final_metrics": best_metrics,
        "total_epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Pelatihan [%s] selesai. Summary disimpan -> %s", aspect_name, summary_path)

    # 9. Bebaskan memori GPU & RAM sebelum keluar
    del model, optimizer, scheduler, criterion, train_loader, val_loader
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def train_all_aspects(
    ground_truth_path: Union[str, Path],
    split_indices_path: Union[str, Path],
    models_dir: Union[str, Path] = "models",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    device: Optional[Union[torch.device, str]] = None,
) -> Dict[str, Any]:
    """Melatih ketiga model BERT aspek (gameplay, story, technical) secara sekuensial."""
    if epochs <= 0:
        raise ValueError(f"epochs harus bernilai integer positif > 0, diterima: {epochs}")

    all_summaries = {}

    for aspect in VALID_ASPECTS:
        logger.info("==================================================")
        logger.info("Memulai pelatihan sekuensial aspek: %s", aspect.upper())
        logger.info("==================================================")

        try:
            summary = train_aspect_model(
                aspect_name=aspect,
                ground_truth_path=ground_truth_path,
                split_indices_path=split_indices_path,
                models_dir=models_dir,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_summaries[aspect] = summary

    return all_summaries
