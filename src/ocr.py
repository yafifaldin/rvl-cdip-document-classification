"""Ekstraksi teks dan posisi kata (bounding box) dari gambar dokumen memakai docTR.

docTR adalah OCR berbasis PyTorch yang terdiri dari dua neural network:
detektor teks (``db_resnet50``) dan pengenal teks (``crnn_vgg16_bn``).

Hasil OCR disimpan per *shard* (file ``ocr_shard_*.jsonl.gz``) supaya jika
proses terhenti (misalnya batas 12 jam Kaggle), paling banyak satu shard yang hilang
dan proses bisa dilanjutkan.
"""

import gzip
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def build_ocr_predictor(det_arch: str, reco_arch: str, det_bs: int, reco_bs: int, device: str):
    """Membuat predictor OCR docTR.

    ``assume_straight_pages=True`` hanya membaca teks horizontal. Teks vertikal
    (umumnya nomor Bates di tepi halaman) sengaja dilewati karena tidak berguna
    untuk klasifikasi dan mode kotak miring jauh lebih lambat.

    Args:
        det_arch: Arsitektur detektor teks.
        reco_arch: Arsitektur pengenal teks.
        det_bs: Ukuran batch internal detektor.
        reco_bs: Ukuran batch internal pengenal.
        device: ``"cuda"`` atau ``"cpu"``.

    Returns:
        Predictor docTR dalam mode evaluasi.
    """
    from doctr.models import ocr_predictor   # impor di sini agar modul lain tidak wajib memasang docTR

    try:
        predictor = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True,
                                  assume_straight_pages=True, det_bs=det_bs, reco_bs=reco_bs)
    except TypeError:
        predictor = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True,
                                  assume_straight_pages=True)
    return predictor.to(device).eval()


class PageDataset(Dataset):
    """Dataset halaman RGB penuh untuk OCR (decode dilakukan di worker CPU)."""

    def __init__(self, ids: List[str], paths: List[str]):
        """Membuat dataset.

        Args:
            ids: Id halaman.
            paths: Path gambar, urut sesuai ``ids``.
        """
        self.ids, self.paths = ids, paths

    def __len__(self) -> int:
        """Mengembalikan jumlah halaman."""
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[str, np.ndarray]:
        """Mengambil satu halaman.

        Args:
            i: Indeks halaman.

        Returns:
            Tuple ``(id, array_RGB_uint8)``; halaman rusak diganti halaman putih.
        """
        try:
            with Image.open(self.paths[i]) as im:
                im.seek(0)
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        except Exception:
            arr = np.full((1000, 760, 3), 255, dtype=np.uint8)
        return self.ids[i], arr


def page_record(page_id: str, split: str, page) -> Dict:
    """Mengubah hasil OCR satu halaman menjadi dictionary yang bisa disimpan sebagai JSON.

    Koordinat kotak diskalakan ke 0–1000, format yang juga dipakai LayoutLM.

    Args:
        page_id: Id halaman.
        split: ``"train"`` atau ``"test"``.
        page: Objek halaman hasil docTR.

    Returns:
        Dictionary ``{id, split, words, boxes, conf}``.
    """
    words, boxes, conf = [], [], []
    for block in page.blocks:
        for line in block.lines:
            for w in line.words:
                (x0, y0), (x1, y1) = w.geometry
                words.append(w.value)
                boxes.append([int(np.clip(v, 0, 1) * 1000) for v in (x0, y0, x1, y1)])
                conf.append(round(float(w.confidence), 3))
    return {"id": page_id, "split": split, "words": words, "boxes": boxes, "conf": conf}


def done_ids(shard_dir: Path) -> Set[str]:
    """Mengumpulkan id halaman yang sudah pernah di-OCR.

    Args:
        shard_dir: Folder berisi file ``ocr_shard_*.jsonl.gz``.

    Returns:
        Himpunan id yang sudah selesai.
    """
    out: Set[str] = set()
    for rec in iter_records(shard_dir):
        out.add(rec["id"])
    return out


def iter_records(shard_dir: Path) -> Iterable[Dict]:
    """Membaca semua record OCR dari folder shard satu per satu.

    Args:
        shard_dir: Folder berisi file ``ocr_shard_*.jsonl.gz``.

    Yields:
        Dictionary record per halaman.
    """
    for f in sorted(Path(shard_dir).rglob("ocr_shard_*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                yield json.loads(line)


def run_ocr(df: pd.DataFrame, out_dir: Path, cfg: Dict, device: str, num_workers: int = 4) -> int:
    """Menjalankan OCR pada semua halaman yang belum diproses dan menyimpannya per shard.

    Args:
        df: DataFrame dengan kolom ``id``, ``path``, ``split``.
        out_dir: Folder output shard.
        cfg: Konfigurasi dari ``configs/ocr.yaml``.
        device: ``"cuda"`` atau ``"cpu"``.
        num_workers: Jumlah worker decode gambar.

    Returns:
        Jumlah halaman yang di-OCR pada pemanggilan ini.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = df[~df["id"].isin(done_ids(out_dir))].reset_index(drop=True)
    if todo.empty:
        return 0
    predictor = build_ocr_predictor(cfg["det_arch"], cfg["reco_arch"], cfg["det_bs"], cfg["reco_bs"], device)
    loader = DataLoader(PageDataset(todo["id"].tolist(), todo["path"].tolist()),
                        batch_size=cfg["pages_per_call"], num_workers=num_workers, collate_fn=lambda b: b)
    split_of = dict(zip(todo["id"], todo["split"]))
    tag, shard, buffer = time.strftime("%Y%m%d%H%M%S"), 0, []

    def flush() -> None:
        """Menulis isi buffer ke satu file shard lalu mengosongkan buffer."""
        nonlocal shard, buffer
        if buffer:
            with gzip.open(out_dir / f"ocr_shard_{tag}_{shard:03d}.jsonl.gz", "wt") as fh:
                for r in buffer:
                    fh.write(json.dumps(r) + "\n")
            shard, buffer = shard + 1, []

    t0, n = time.time(), 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="OCR"):
            ids, pages = zip(*batch)
            result = predictor(list(pages))
            buffer.extend(page_record(i, split_of[i], p) for i, p in zip(ids, result.pages))
            n += len(ids)
            if len(buffer) >= cfg["shard_size"]:
                flush()
            if (time.time() - t0) / 3600 > cfg["time_budget_hours"]:
                break
    flush()
    return n


def build_text_table(shard_dir: Path) -> pd.DataFrame:
    """Menggabungkan semua shard menjadi tabel teks per halaman.

    Args:
        shard_dir: Folder berisi file shard OCR.

    Returns:
        DataFrame ``id, split, n_words, mean_conf, text`` (satu baris per halaman).
    """
    rows = [{"id": r["id"], "split": r["split"], "n_words": len(r["words"]),
             "mean_conf": float(np.mean(r["conf"])) if r["conf"] else 0.0,
             "text": " ".join(r["words"])} for r in iter_records(shard_dir)]
    return pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True)


def ocr_quality_summary(text_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan kualitas OCR per kelas (jumlah kata dan confidence).

    Args:
        text_df: Hasil :func:`build_text_table`.
        train_df: Data train dengan kolom ``id`` dan ``class_name``.

    Returns:
        DataFrame per kelas: ``pages, median_words, pct_under_5_words, mean_conf``.
    """
    q = text_df.merge(train_df[["id", "class_name"]], on="id", how="inner")
    return q.groupby("class_name").agg(
        pages=("id", "size"),
        median_words=("n_words", "median"),
        pct_under_5_words=("n_words", lambda s: round(100 * (s < 5).mean(), 1)),
        mean_conf=("mean_conf", "mean"),
    ).round(3).sort_values("median_words")
