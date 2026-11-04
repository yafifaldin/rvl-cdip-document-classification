"""Fungsi bantu umum yang dipakai oleh banyak modul."""

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix


def seed_everything(seed: int) -> None:
    """Mengatur seed semua sumber angka acak supaya hasil bisa direproduksi.

    Catatan: training di GPU tetap tidak 100% deterministik (operasi fp16 dan
    operasi paralel di GPU), sehingga hasil run ulang bisa sedikit berbeda.

    Args:
        seed: Nilai seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Memilih device komputasi.

    Returns:
        ``"cuda"`` jika GPU tersedia, selain itu ``"cpu"``.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_file(name: str, root: str = "data") -> Path:
    """Mencari tepat satu file berdasarkan nama di dalam sebuah folder (rekursif).

    Dipakai agar kode tidak bergantung pada struktur folder input yang bisa
    berbeda antara Kaggle dan komputer lokal.

    Args:
        name: Nama file yang dicari, misalnya ``"train.csv"``.
        root: Folder awal pencarian.

    Returns:
        Path file yang ditemukan.

    Raises:
        FileNotFoundError: Jika file tidak ditemukan atau ditemukan lebih dari satu.
    """
    hits = sorted(Path(root).rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"Harus ada tepat satu '{name}' di bawah '{root}', ditemukan {len(hits)}: {hits}")
    return hits[0]


def top_confusions(y_true: Sequence[int], y_pred: Sequence[int], classes: List[str],
                   k: int = 12) -> List[Tuple[int, str, str]]:
    """Mengambil pasangan kelas yang paling sering tertukar.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label hasil prediksi.
        classes: Nama kelas, urut sesuai indeks label.
        k: Jumlah pasangan yang dikembalikan.

    Returns:
        List ``(jumlah, kelas_asli, kelas_prediksi)``, urut dari yang terbanyak.
    """
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = [(int(cm[i, j]), classes[i], classes[j])
             for i in labels for j in labels if i != j and cm[i, j] > 0]
    return sorted(pairs, reverse=True)[:k]


def save_probs(out_dir: Path, ids: Sequence[str], probs: np.ndarray, prefix: str) -> None:
    """Menyimpan matriks probabilitas beserta urutan id-nya.

    Args:
        out_dir: Folder tujuan.
        ids: Id halaman, urut sesuai baris ``probs``.
        probs: Array berukuran ``(n_halaman, n_kelas)``.
        prefix: Awalan nama file, misalnya ``"val"`` atau ``"test"``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{prefix}_probs.npy", probs)
    pd.DataFrame({"id": list(ids)}).to_csv(out_dir / f"{prefix}_ids.csv", index=False)


def load_probs(model_dir: Path, prefix: str) -> pd.DataFrame:
    """Membaca probabilitas yang disimpan oleh :func:`save_probs`.

    Args:
        model_dir: Folder output sebuah model.
        prefix: ``"val"`` atau ``"test"``.

    Returns:
        DataFrame dengan index = id halaman dan satu kolom per kelas,
        sehingga bisa disejajarkan dengan model lain lewat ``.loc[ids]``.
    """
    model_dir = Path(model_dir)
    ids = pd.read_csv(model_dir / f"{prefix}_ids.csv")["id"].values
    probs = np.load(model_dir / f"{prefix}_probs.npy")
    if probs.shape[0] != len(ids):
        raise ValueError(f"Jumlah baris probabilitas ({probs.shape[0]}) != jumlah id ({len(ids)}) di {model_dir}")
    return pd.DataFrame(probs, index=ids)
