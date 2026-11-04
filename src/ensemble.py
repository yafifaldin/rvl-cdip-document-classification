"""Penggabungan probabilitas beberapa model (ensemble) dan pencarian bobotnya.

Dua cara penggabungan dicoba:

- ``mean``: rata-rata berbobot dari probabilitas.
- ``geo`` : rata-rata berbobot dari log-probabilitas (rata-rata geometrik).
  Cara ini biasanya lebih baik jika tingkat "keyakinan" model berbeda-beda,
  misalnya Logistic Regression vs. neural network dengan label smoothing.

Bobot dicari dengan grid search pada data holdout. Karena bobot dipilih pada
holdout yang sama, akurasi holdout ensemble sedikit optimistis; skor
leaderboard tetap menjadi pemeriksaan sebenarnya.
"""

from itertools import product
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def combine(probs: Sequence[np.ndarray], weights: Sequence[float], mode: str = "geo") -> np.ndarray:
    """Menggabungkan probabilitas beberapa model.

    Args:
        probs: List array probabilitas, masing-masing ``(n, n_kelas)``.
        weights: Bobot tiap model.
        mode: ``"mean"`` atau ``"geo"``.

    Returns:
        Skor gabungan ``(n, n_kelas)``; kelas prediksi = ``argmax`` per baris.
    """
    if mode == "mean":
        return sum(w * P for w, P in zip(weights, probs))
    return sum(w * np.log(np.clip(P, 1e-7, 1.0)) for w, P in zip(weights, probs))


def weight_grid(n_models: int, step: float = 0.05) -> List[tuple]:
    """Membuat semua kombinasi bobot non-negatif yang jumlahnya 1.

    Args:
        n_models: Jumlah model.
        step: Jarak antar nilai bobot.

    Returns:
        List tuple bobot.
    """
    steps = np.round(np.arange(0, 1 + 1e-9, step), 2)
    grid = []
    for head in product(steps, repeat=n_models - 1):
        last = round(1 - sum(head), 2)
        if last >= -1e-9:
            grid.append(tuple(head) + (abs(last),))
    return grid


def search_weights(probs: Sequence[np.ndarray], y: np.ndarray, names: Sequence[str],
                   step: float = 0.05, modes: Sequence[str] = ("mean", "geo")) -> pd.DataFrame:
    """Mencari bobot ensemble terbaik pada holdout dengan grid search.

    Args:
        probs: List probabilitas holdout tiap model, baris sudah sejajar.
        y: Label holdout.
        names: Nama model, urut sesuai ``probs``.
        step: Jarak antar nilai bobot.
        modes: Cara penggabungan yang dicoba.

    Returns:
        DataFrame semua kombinasi (kolom ``mode``, ``w_<nama>``, ``val_acc``),
        urut dari akurasi tertinggi. Urutan stabil, sehingga jika seri,
        kombinasi yang lebih dulu dicoba yang dipilih.
    """
    rows = []
    for mode in modes:
        for ws in weight_grid(len(probs), step):
            acc = accuracy_score(y, combine(probs, ws, mode).argmax(1))
            rows.append((mode, *ws, acc))
    cols = ["mode"] + [f"w_{n}" for n in names] + ["val_acc"]
    return pd.DataFrame(rows, columns=cols).sort_values("val_acc", ascending=False, kind="stable").reset_index(drop=True)


def per_class_f1(y: np.ndarray, preds: dict, classes: List[str]) -> pd.DataFrame:
    """Membuat tabel F1 per kelas untuk beberapa set prediksi.

    Args:
        y: Label sebenarnya.
        preds: Dictionary ``{nama_kolom: array_prediksi}``.
        classes: Nama kelas.

    Returns:
        DataFrame dengan index nama kelas dan satu kolom per set prediksi.
    """
    labels = list(range(len(classes)))
    return pd.DataFrame({k: f1_score(y, v, average=None, labels=labels) for k, v in preds.items()},
                        index=classes).round(3)


def write_submission(ids: Sequence[str], pred: np.ndarray, sample_sub: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Menulis file submission sesuai format ``sample_submission.csv``.

    Args:
        ids: Id test, urut sesuai ``pred``.
        pred: Label prediksi (0–15).
        sample_sub: Isi ``sample_submission.csv``.
        path: Lokasi file output.

    Returns:
        DataFrame submission yang ditulis.

    Raises:
        ValueError: Jika ada id test yang tidak punya prediksi.
    """
    sub = sample_sub[["id"]].merge(pd.DataFrame({"id": list(ids), "label": pred}), on="id", how="left")
    if sub["label"].isna().any() or len(sub) != len(sample_sub):
        raise ValueError("Submission tidak lengkap: ada id test tanpa prediksi")
    sub["label"] = sub["label"].astype(int)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return sub
