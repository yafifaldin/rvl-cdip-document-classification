"""Script training model gambar (Donut atau DiT) dari awal sampai prediksi test.

Contoh pemakaian::

    python -m src.train_image --model donut
    python -m src.train_image --model dit

Output di ``checkpoint/<model>/``: ``best.pt``, ``val_probs.npy``, ``val_ids.csv``,
``test_probs.npy``, ``test_ids.csv``, ``history.csv``. Id holdout disimpan di
``checkpoint/holdout_ids.csv`` pada run pertama dan dipakai ulang oleh semua model.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from src.config import load_config
from src.data import (ArrayDataset, DocDataset, audit_images, cache_images, load_metadata,
                      make_loader, split_holdout)
from src.engine import fit, load_best, predict
from src.models import build_model_and_groups
from src.utils import get_device, save_probs, seed_everything, top_confusions


def parse_args() -> argparse.Namespace:
    """Membaca argumen command line.

    Returns:
        Namespace berisi ``model``, ``data_dir``, ``ckpt_dir``, ``num_workers``.
    """
    p = argparse.ArgumentParser(description="Training model gambar RVL-CDIP")
    p.add_argument("--model", choices=["donut", "dit"], required=True, help="Nama konfigurasi di configs/")
    p.add_argument("--data-dir", default="data", help="Folder data kompetisi")
    p.add_argument("--ckpt-dir", default="checkpoint", help="Folder output checkpoint dan probabilitas")
    p.add_argument("--num-workers", type=int, default=os.cpu_count() or 2)
    return p.parse_args()


def main() -> None:
    """Menjalankan pipeline: audit data -> split -> training -> evaluasi -> prediksi test."""
    args = parse_args()
    cfg = load_config(args.model)
    seed_everything(cfg["seed"])
    device = get_device()
    out_dir = Path(args.ckpt_dir) / args.model

    train_df, test_df, _, classes = load_metadata(args.data_dir)

    # Audit: hanya file rusak atau benar-benar kosong yang dibuang dari train.
    audit = audit_images(train_df["path"].tolist(), args.num_workers)
    bad = set(audit.loc[(~audit["ok"]) | (audit["std"] < 1.0), "path"])
    print(f"Halaman train yang dibuang (rusak/kosong): {len(bad)}")
    train_df = train_df[~train_df["path"].isin(bad)].reset_index(drop=True)

    tr_df, va_df = split_holdout(train_df, Path(args.ckpt_dir) / "holdout_ids.csv", cfg["val_frac"], cfg["seed"])
    print(f"train {len(tr_df)} | holdout {len(va_df)} | test {len(test_df)}")

    if cfg.get("cache_images", False):
        side = cfg["img_height"]
        tr_arr, tr_ok = cache_images(tr_df["path"].tolist(), side, args.num_workers)
        tr_df, tr_arr = tr_df[tr_ok].reset_index(drop=True), tr_arr[tr_ok]
        va_arr, _ = cache_images(va_df["path"].tolist(), side, args.num_workers)
        te_arr, _ = cache_images(test_df["path"].tolist(), side, args.num_workers)
        ds_tr = ArrayDataset(tr_arr, tr_df["label"], train=True)
        ds_va = ArrayDataset(va_arr, va_df["label"])
        ds_te = ArrayDataset(te_arr, [-1] * len(test_df))
    else:
        size = (cfg["img_width"], cfg["img_height"])
        ds_tr, ds_va, ds_te = DocDataset(tr_df, size, train=True), DocDataset(va_df, size), DocDataset(test_df, size)

    bs = cfg["batch_size"]
    train_loader = make_loader(ds_tr, bs, True, args.num_workers)
    val_loader = make_loader(ds_va, bs, False, args.num_workers)
    test_loader = make_loader(ds_te, bs, False, args.num_workers)

    model, groups = build_model_and_groups(cfg, len(classes))
    model.to(device)
    history = fit(model, groups, train_loader, val_loader, va_df["label"].values, cfg, device, out_dir / "best.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    load_best(model, out_dir / "best.pt", device)
    val_probs, test_probs = predict(model, val_loader, device), predict(model, test_loader, device)
    save_probs(out_dir, va_df["id"], val_probs, "val")
    save_probs(out_dir, test_df["id"], test_probs, "test")

    y, pred = va_df["label"].values, val_probs.argmax(1)
    print(classification_report(y, pred, labels=list(range(len(classes))), target_names=classes, digits=3,
                                zero_division=0))
    for cnt, a, b in top_confusions(y, pred, classes):
        print(f"  {cnt:4d}  {a} -> {b}")


if __name__ == "__main__":
    main()
