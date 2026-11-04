"""Script ensemble: mencari bobot pada holdout lalu menulis file submission.

Contoh pemakaian::

    python -m src.make_ensemble --models donut dit text

Setiap model dibaca dari ``checkpoint/<nama>/`` (``val_probs.npy``,
``val_ids.csv``, ``test_probs.npy``, ``test_ids.csv``). Hasil akhir ditulis ke
``submission.csv`` dan tabel bobot ke ``checkpoint/ensemble_grid.csv``.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data import load_metadata
from src.ensemble import combine, per_class_f1, search_weights, write_submission
from src.utils import load_probs, top_confusions


def parse_args() -> argparse.Namespace:
    """Membaca argumen command line.

    Returns:
        Namespace berisi ``models``, ``data_dir``, ``ckpt_dir``, ``out``.
    """
    p = argparse.ArgumentParser(description="Ensemble probabilitas beberapa model")
    p.add_argument("--models", nargs="+", default=["donut", "dit", "text"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--ckpt-dir", default="checkpoint")
    p.add_argument("--out", default="submission.csv")
    return p.parse_args()


def main() -> None:
    """Menyejajarkan probabilitas per id, mencari bobot terbaik, dan menulis submission."""
    args = parse_args()
    cfg = load_config("text")
    train_df, test_df, sample_sub, classes = load_metadata(args.data_dir)
    holdout = pd.read_csv(Path(args.ckpt_dir) / "holdout_ids.csv")["id"].values
    y = train_df.set_index("id").loc[holdout, "label"].values
    te_ids = test_df["id"].values

    val = [load_probs(Path(args.ckpt_dir) / m, "val").loc[holdout].values for m in args.models]
    test = [load_probs(Path(args.ckpt_dir) / m, "test").loc[te_ids].values for m in args.models]

    grid = search_weights(val, y, args.models, cfg["weight_step"])
    grid.to_csv(Path(args.ckpt_dir) / "ensemble_grid.csv", index=False)
    best = grid.iloc[0]
    weights = [best[f"w_{m}"] for m in args.models]
    print("10 kombinasi terbaik:\n", grid.head(10).to_string(index=False))

    preds = {m: P.argmax(1) for m, P in zip(args.models, val)}
    preds["ensemble"] = combine(val, weights, best["mode"]).argmax(1)
    print(per_class_f1(y, preds, classes).to_string())
    for cnt, a, b in top_confusions(y, preds["ensemble"], classes, 10):
        print(f"  {cnt:4d}  {a} -> {b}")

    sub = write_submission(te_ids, combine(test, weights, best["mode"]).argmax(1), sample_sub, Path(args.out))
    print(f"Submission ditulis: {args.out} ({len(sub)} baris) | holdout ensemble {best['val_acc']:.4f}")


if __name__ == "__main__":
    main()
