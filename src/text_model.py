"""Model teks: fitur TF-IDF dari hasil OCR + Logistic Regression multinomial.

Model ini melengkapi model gambar. Model gambar mengenali *tampilan* halaman,
sedangkan model teks membaca *kata-kata* di halaman (misalnya "invoice",
"period covered", "subject:"), sehingga kelas yang tampilannya mirip tetapi
isinya berbeda bisa dibedakan.

Catatan: Logistic Regression multinomial secara matematis setara dengan neural
network satu layer (linear + softmax) yang dilatih dengan cross-entropy.
Di proyek ini model tersebut dilatih memakai scikit-learn.
"""

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def normalize_text(text: str) -> str:
    """Normalisasi teks OCR: huruf kecil dan setiap digit diganti ``0``.

    Mengganti digit membuat nomor Bates, tanggal, dan angka nominal menjadi
    beberapa pola bersama saja, bukan ribuan token unik yang tidak berguna.

    Args:
        text: Teks mentah hasil OCR.

    Returns:
        Teks yang sudah dinormalisasi.
    """
    return re.sub(r"\d", "0", str(text).lower())


class TextFeaturizer:
    """Mengubah teks OCR menjadi matriks fitur sparse.

    Fitur yang dipakai:

    - TF-IDF kata 1–2-gram: menangkap kata kunci.
    - TF-IDF karakter 3–5-gram: tahan terhadap salah eja OCR
      ("labornt**ory**" masih berbagi potongan dengan "laboratory").
    - ``min_df=3``: token yang muncul di kurang dari 3 halaman dibuang,
      sehingga sebagian besar "sampah" OCR hilang.
    - Dua fitur halaman: jumlah kata (log) dan rata-rata confidence OCR.
    """

    def __init__(self, cfg: Dict):
        """Membuat featurizer.

        Args:
            cfg: Konfigurasi dari ``configs/text.yaml``.
        """
        self.word = TfidfVectorizer(analyzer="word", ngram_range=tuple(cfg["word_ngram"]), min_df=cfg["min_df"],
                                    max_features=cfg["max_word_features"], sublinear_tf=True,
                                    token_pattern=r"(?u)\b\w+\b")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(cfg["char_ngram"]), min_df=cfg["min_df"],
                                    max_features=cfg["max_char_features"], sublinear_tf=True)

    @staticmethod
    def _meta(n_words: Sequence[float], mean_conf: Sequence[float]) -> sp.csr_matrix:
        """Membuat fitur halaman (jumlah kata dan confidence).

        Args:
            n_words: Jumlah kata OCR per halaman.
            mean_conf: Rata-rata confidence OCR per halaman.

        Returns:
            Matriks sparse berukuran ``(n, 2)``.
        """
        n = np.asarray(n_words, dtype=float)
        c = np.asarray(mean_conf, dtype=float)
        return sp.csr_matrix(np.column_stack([np.log1p(n) / np.log1p(1000), c]))

    def fit_transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """Mempelajari kosakata dari data lalu mengubahnya menjadi fitur.

        Args:
            df: DataFrame dengan kolom ``text``, ``n_words``, ``mean_conf``.

        Returns:
            Matriks fitur sparse.
        """
        t = [normalize_text(x) for x in df["text"].fillna("")]
        return sp.hstack([self.word.fit_transform(t), self.char.fit_transform(t),
                          self._meta(df["n_words"], df["mean_conf"])]).tocsr()

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """Mengubah data baru menjadi fitur memakai kosakata yang sudah dipelajari.

        Args:
            df: DataFrame dengan kolom ``text``, ``n_words``, ``mean_conf``.

        Returns:
            Matriks fitur sparse.
        """
        t = [normalize_text(x) for x in df["text"].fillna("")]
        return sp.hstack([self.word.transform(t), self.char.transform(t),
                          self._meta(df["n_words"], df["mean_conf"])]).tocsr()


def fit_logreg(X: sp.csr_matrix, y: np.ndarray, C: float) -> LogisticRegression:
    """Melatih Logistic Regression multinomial.

    Args:
        X: Matriks fitur.
        y: Label.
        C: Kebalikan kekuatan regularisasi (semakin besar, semakin lemah regularisasinya).

    Returns:
        Model yang sudah dilatih.
    """
    return LogisticRegression(C=C, max_iter=1000, solver="lbfgs").fit(X, y)


def select_c(X_tr: sp.csr_matrix, y_tr: np.ndarray, X_va: sp.csr_matrix, y_va: np.ndarray,
             c_grid: List[float]) -> Tuple[float, np.ndarray, Dict[float, float]]:
    """Memilih nilai C terbaik berdasarkan akurasi holdout.

    Args:
        X_tr: Fitur train.
        y_tr: Label train.
        X_va: Fitur holdout.
        y_va: Label holdout.
        c_grid: Kandidat nilai C.

    Returns:
        Tuple ``(C_terbaik, probabilitas_holdout_C_terbaik, akurasi_per_C)``.
    """
    scores, probs = {}, {}
    for C in c_grid:
        P = fit_logreg(X_tr, y_tr, C).predict_proba(X_va)
        scores[C], probs[C] = accuracy_score(y_va, P.argmax(1)), P
        print(f"C={C:>5}: akurasi holdout model teks {scores[C]:.4f}")
    best = max(scores, key=scores.get)
    return best, probs[best], scores


def gate_uniform(P: np.ndarray, n_words: Sequence[int], min_words: int) -> np.ndarray:
    """Mengganti probabilitas teks menjadi uniform pada halaman yang hampir kosong.

    Probabilitas uniform menambah nilai yang sama ke setiap kelas, sehingga
    tidak memengaruhi keputusan ensemble. Untuk halaman seperti itu, model
    gambar yang menentukan.

    Args:
        P: Probabilitas model teks ``(n, n_kelas)``.
        n_words: Jumlah kata OCR per halaman.
        min_words: Batas minimal jumlah kata.

    Returns:
        Salinan ``P`` yang sudah di-gate.
    """
    P = P.copy()
    P[np.asarray(n_words) < min_words] = 1.0 / P.shape[1]
    return P
