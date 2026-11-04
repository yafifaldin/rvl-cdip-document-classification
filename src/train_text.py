"""Script training model teks (TF-IDF + Logistic Regression) dari hasil OCR.

Contoh pemakaian (setelah ``src.run_ocr`` dan minimal satu ``src.train_image``)::

    python -m src.train_text

Output di ``checkpoint/text/``: ``val_probs.npy``, ``val_ids.csv``,
``test_probs.npy``, ``test_ids.csv``. Probabilitas yang disimpan sudah di-gate
(uniform pada halaman dengan sangat sedikit kata OCR).
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from src.config import load_config
from src.data import load_metadata
from src.text_model import TextFeaturizer, fit_logreg, gate_uniform, select_c
from src.utils import save_probs


def parse_args() -> argparse.Namespace:
    """Membaca argumen command line.

    Returns:
        Namespace berisi ``data_dir``, ``ocr_csv``, ``ckpt_dir``.
    """
    p = argparse.ArgumentParser(description="Training model teks RVL-CDIP")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--ocr-csv", default="checkpoint/ocr/ocr_text.csv")
    p.add_argument("--ckpt-dir", default="checkpoint")
    return p.parse_args()


def main() -> None:
    """Memilih C pada holdout, menyimpan probabilitas holdout, lalu refit untuk prediksi test."""
    args = parse_args()
    cfg = load_config("text")
    train_df, test_df, _, classes = load_metadata(args.data_dir)
    ocr = pd.read_csv(args.ocr_csv).set_index("id")
    ocr["text"] = ocr["text"].fillna("")

    holdout = pd.read_csv(Path(args.ckpt_dir) / "holdout_ids.csv")["id"].tolist()
    lab = train_df.set_index("id")["label"]
    tr_ids = [i for i in train_df["id"] if i not in set(holdout)]
    y_tr, y_va = lab.loc[tr_ids].values, lab.loc[holdout].values

    feat = TextFeaturizer(cfg)
    X_tr = feat.fit_transform(ocr.loc[tr_ids])
    X_va = feat.transform(ocr.loc[holdout])
    best_c, P_va, _ = select_c(X_tr, y_tr, X_va, y_va, cfg["c_grid"])
    print(f"C terbaik: {best_c}")
    print(classification_report(y_va, P_va.argmax(1), labels=list(range(len(classes))), target_names=classes,
                                digits=3, zero_division=0))

    out_dir = Path(args.ckpt_dir) / "text"
    save_probs(out_dir, holdout, gate_uniform(P_va, ocr.loc[holdout, "n_words"], cfg["min_words"]), "val")

    fit_ids = train_df["id"].tolist() if cfg["refit_full"] else tr_ids
    feat = TextFeaturizer(cfg)
    clf = fit_logreg(feat.fit_transform(ocr.loc[fit_ids]), lab.loc[fit_ids].values, best_c)
    te_ids = test_df["id"].tolist()
    P_te = clf.predict_proba(feat.transform(ocr.loc[te_ids]))
    save_probs(out_dir, te_ids, gate_uniform(P_te, ocr.loc[te_ids, "n_words"], cfg["min_words"]), "test")


if __name__ == "__main__":
    main()
