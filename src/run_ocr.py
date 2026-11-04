"""Script OCR seluruh halaman train dan test memakai docTR (butuh GPU agar cepat).

Contoh pemakaian::

    python -m src.run_ocr

Output di ``checkpoint/ocr/``: file ``ocr_shard_*.jsonl.gz`` (kata + kotak per
halaman) dan ``ocr_text.csv`` (teks gabungan per halaman). Jika proses
terhenti, jalankan lagi perintah yang sama; halaman yang sudah selesai dilewati.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data import load_metadata
from src.ocr import build_text_table, ocr_quality_summary, run_ocr
from src.utils import get_device


def parse_args() -> argparse.Namespace:
    """Membaca argumen command line.

    Returns:
        Namespace berisi ``data_dir``, ``out_dir``, ``num_workers``.
    """
    p = argparse.ArgumentParser(description="OCR docTR untuk RVL-CDIP")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="checkpoint/ocr")
    p.add_argument("--num-workers", type=int, default=os.cpu_count() or 2)
    return p.parse_args()


def main() -> None:
    """Menjalankan OCR, menggabungkan shard, lalu mencetak ringkasan kualitas per kelas."""
    args = parse_args()
    cfg = load_config("ocr")
    train_df, test_df, _, _ = load_metadata(args.data_dir)
    pages = pd.concat([test_df.assign(split="test")[["id", "path", "split"]],
                       train_df.assign(split="train")[["id", "path", "split"]]], ignore_index=True)
    n = run_ocr(pages, Path(args.out_dir), cfg, get_device(), args.num_workers)
    print(f"Halaman yang di-OCR pada run ini: {n}")

    text_df = build_text_table(Path(args.out_dir))
    text_df.to_csv(Path(args.out_dir) / "ocr_text.csv", index=False)
    print(f"Total halaman di ocr_text.csv: {len(text_df)} (target {len(pages)})")
    print(ocr_quality_summary(text_df, train_df).to_string())


if __name__ == "__main__":
    main()
