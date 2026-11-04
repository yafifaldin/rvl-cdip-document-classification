"""Loop training, evaluasi, dan prediksi untuk model gambar."""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Menghitung probabilitas softmax untuk semua batch dalam loader.

    Args:
        model: Model yang mengembalikan logit.
        loader: DataLoader berisi ``(gambar, label)``.
        device: ``"cuda"`` atau ``"cpu"``.

    Returns:
        Array probabilitas berukuran ``(n_halaman, n_kelas)``.
    """
    model.eval()
    out = []
    for x, _ in tqdm(loader, desc="predict", leave=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        out.append(logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scheduler, scaler: torch.amp.GradScaler, criterion: nn.Module,
                    device: str, grad_accum: int = 1) -> float:
    """Melatih model selama satu epoch dengan mixed precision (fp16).

    Args:
        model: Model yang dilatih.
        loader: DataLoader data train.
        optimizer: Optimizer.
        scheduler: Scheduler learning rate (di-step per update optimizer).
        scaler: GradScaler untuk fp16.
        criterion: Fungsi loss.
        device: ``"cuda"`` atau ``"cpu"``.
        grad_accum: Jumlah batch yang diakumulasi sebelum satu update.

    Returns:
        Rata-rata loss training pada epoch ini.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    run_loss = 0.0
    pbar = tqdm(loader, desc="train", leave=False)
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            loss = criterion(model(x), y) / grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        run_loss += loss.item() * grad_accum
        pbar.set_postfix(loss=f"{run_loss / (step + 1):.4f}")
    return run_loss / len(loader)


def fit(model: nn.Module, param_groups: List[Dict], train_loader: DataLoader, val_loader: DataLoader,
        y_val: np.ndarray, cfg: Dict, device: str, ckpt_path: Path) -> List[Dict]:
    """Melatih model selama beberapa epoch dan menyimpan checkpoint terbaik.

    Scheduler yang dipakai adalah cosine dengan warmup. Checkpoint disimpan
    setiap kali akurasi holdout membaik.

    Args:
        model: Model yang dilatih (sudah dipindah ke ``device``).
        param_groups: Grup parameter untuk AdamW.
        train_loader: DataLoader train.
        val_loader: DataLoader holdout.
        y_val: Label holdout, urut sesuai ``val_loader``.
        cfg: Konfigurasi model (epochs, grad_accum, warmup, label_smoothing).
        device: ``"cuda"`` atau ``"cpu"``.
        ckpt_path: Lokasi file checkpoint terbaik.

    Returns:
        Riwayat per epoch: list dict ``{epoch, train_loss, val_acc, val_macro_f1, minutes}``.
    """
    grad_accum = cfg.get("grad_accum", 1)
    epochs = cfg["epochs"]
    optimizer = torch.optim.AdamW(param_groups)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total = steps_per_epoch * epochs
    if "warmup_epochs" in cfg:
        warmup = steps_per_epoch * cfg["warmup_epochs"]
    else:
        warmup = int(cfg.get("warmup_ratio", 0.05) * total)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device, grad_accum)
        pred = predict(model, val_loader, device).argmax(1)
        row = {"epoch": epoch, "train_loss": loss, "val_acc": accuracy_score(y_val, pred),
               "val_macro_f1": f1_score(y_val, pred, average="macro"), "minutes": (time.time() - t0) / 60}
        history.append(row)
        print(f"epoch {epoch}: train_loss {loss:.4f} | val_acc {row['val_acc']:.4f} | "
              f"val_macroF1 {row['val_macro_f1']:.4f} | {row['minutes']:.1f} min")
        if row["val_acc"] > best:
            best = row["val_acc"]
            torch.save(model.state_dict(), ckpt_path)
    return history


def load_best(model: nn.Module, ckpt_path: Path, device: Optional[str] = None) -> nn.Module:
    """Memuat bobot terbaik ke dalam model.

    Args:
        model: Model dengan arsitektur yang sama.
        ckpt_path: File checkpoint dari :func:`fit`.
        device: Device tujuan; default ``"cpu"``.

    Returns:
        Model yang sama dengan bobot terbaik.
    """
    model.load_state_dict(torch.load(ckpt_path, map_location=device or "cpu"))
    return model
