"""Membuat grafik untuk laporan di README dari angka hasil eksperimen.

Semua angka di file ini disalin dari output notebook di folder ``notebooks/``
(log training, classification report, dan riwayat submission Kaggle), sehingga
grafik bisa dibuat ulang tanpa GPU::

    python reports/make_figures.py

Grafik disimpan di ``reports/figures/``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"

# Palet kategorikal (3 slot pertama palet referensi, lolos uji buta warna) dan warna teks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

CLASSES = ["advertisement", "budget", "email", "file_folder", "form", "handwritten", "invoice", "letter",
           "memo", "news_article", "presentation", "questionnaire", "resume", "scientific_publication",
           "scientific_report", "specification"]

# F1 per kelas pada holdout (3.201 halaman), dari classification report di notebook 02, 03, dan 05.
F1 = {
    "Donut": [.891, .861, .993, .913, .745, .937, .894, .882, .937, .889, .838, .878, .975, .922, .797, .914],
    "DiT": [.901, .869, .995, .934, .723, .953, .868, .902, .925, .899, .820, .833, .961, .924, .749, .927],
    "Teks (OCR)": [.738, .749, .959, .628, .726, .786, .844, .851, .852, .784, .579, .870, .975, .871, .741, .908],
    "Ensemble 2 model": [.905, .877, .995, .927, .782, .958, .918, .912, .944, .915, .850, .907, .980, .938, .817, .940],
    "Ensemble 3 model": [.920, .892, .995, .929, .809, .954, .940, .921, .946, .930, .859, .923, .988, .940, .862, .940],
}


def style_axes(ax: plt.Axes) -> None:
    """Menerapkan gaya grafik yang tenang: grid tipis, tanpa garis tepi atas/kanan.

    Args:
        ax: Axes matplotlib yang akan diberi gaya.
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    """Menyimpan grafik ke ``reports/figures/<name>``.

    Args:
        fig: Figure matplotlib.
        name: Nama file PNG.
    """
    FIG.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_score_progression() -> None:
    """Grafik perkembangan skor holdout dan public leaderboard di setiap tahap."""
    stages = ["Donut", "Donut + DiT\n(Version 2)", "Donut + DiT\n(Version 3)", "Donut + DiT\n+ Teks"]
    holdout = [0.8919, 0.9107, 0.9103, 0.9216]
    public = [0.88122, 0.89836, 0.90129, 0.91677]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    x = range(len(stages))
    ax.plot(x, holdout, color=BLUE, linewidth=2, marker="o", markersize=8, label="Holdout (3.201 halaman)")
    ax.plot(x, public, color=ORANGE, linewidth=2, marker="s", markersize=8, label="Public leaderboard")
    for i, (h, p) in enumerate(zip(holdout, public)):
        ax.annotate(f"{h:.4f}", (i, h), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=INK)
        ax.annotate(f"{p:.5f}", (i, p), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages, color=INK)
    ax.set_ylim(0.87, 0.93)
    ax.set_ylabel("Akurasi", color=INK2)
    ax.set_title("Perkembangan skor di setiap tahap", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    save(fig, "score_progression.png")


def plot_training_curves() -> None:
    """Grafik akurasi holdout dan loss training per epoch untuk Donut dan DiT (dua panel, satu sumbu per panel)."""
    donut_acc, donut_loss = [0.8341, 0.8882, 0.8919], [1.4007, 0.9734, 0.8540]
    dit_acc = [0.8035, 0.8322, 0.8569, 0.8532, 0.8604, 0.8735, 0.8675, 0.8766, 0.8794, 0.8857, 0.8816, 0.8832,
               0.8869, 0.8863, 0.8863]
    dit_loss = [1.5631, 1.1127, 1.0002, 0.9307, 0.8814, 0.8377, 0.7912, 0.7517, 0.7219, 0.6904, 0.6638, 0.6469,
                0.6359, 0.6286, 0.6261]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, d, t, title, ylab in ((a1, donut_acc, dit_acc, "Akurasi holdout per epoch", "Akurasi"),
                                  (a2, donut_loss, dit_loss, "Loss training per epoch", "Loss (label smoothing 0,1)")):
        style_axes(ax)
        ax.plot(range(1, len(d) + 1), d, color=BLUE, linewidth=2, marker="o", markersize=7, label="Donut (1024x768)")
        ax.plot(range(1, len(t) + 1), t, color=ORANGE, linewidth=2, marker="s", markersize=6, label="DiT (224x224)")
        ax.set_xlabel("Epoch", color=INK2)
        ax.set_ylabel(ylab, color=INK2)
        ax.set_title(title, color=INK, loc="left", fontsize=12)
        ax.legend(frameon=False, fontsize=9)
    a1.annotate("terbaik 0,8919", (3, 0.8919), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
    a1.annotate("terbaik 0,8869 (epoch 13)", (13, 0.8869), textcoords="offset points", xytext=(-60, 10), fontsize=8,
                color=INK)
    a2.axhline(0.57, color=INK2, linewidth=1, linestyle="--")
    a2.annotate("batas bawah loss ~0,57", (15, 0.57), textcoords="offset points", xytext=(-110, -12), fontsize=8,
                color=INK2)
    a2.set_ylim(0.5, None)
    save(fig, "training_curves.png")


def plot_model_complementarity() -> None:
    """Grafik F1 per kelas untuk tiga model tunggal, untuk melihat di mana kekuatan tiap model."""
    order = sorted(range(len(CLASSES)), key=lambda i: F1["Ensemble 3 model"][i])
    names = [CLASSES[i] for i in order]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    y = range(len(names))
    for key, color, marker in (("Donut", BLUE, "o"), ("DiT", ORANGE, "s"), ("Teks (OCR)", AQUA, "D")):
        ax.scatter([F1[key][i] for i in order], list(y), color=color, marker=marker, s=46, label=key, zorder=3,
                   edgecolors=SURFACE, linewidths=1.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, color=INK)
    ax.set_xlim(0.55, 1.0)
    ax.set_xlabel("F1 pada holdout", color=INK2)
    ax.set_title("F1 per kelas untuk setiap model tunggal", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "model_complementarity.png")


def plot_text_gain() -> None:
    """Grafik F1 per kelas sebelum dan sesudah model teks ditambahkan ke ensemble."""
    two, three = F1["Ensemble 2 model"], F1["Ensemble 3 model"]
    order = sorted(range(len(CLASSES)), key=lambda i: three[i] - two[i])
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    for row, i in enumerate(order):
        ax.plot([two[i], three[i]], [row, row], color=GRID, linewidth=2, zorder=2)
        delta = three[i] - two[i]
        ax.annotate(f"{delta*100:+.1f}", (max(two[i], three[i]), row), textcoords="offset points", xytext=(8, -3),
                    fontsize=8, color=INK)
    ax.scatter([two[i] for i in order], range(len(order)), color=BLUE, marker="o", s=46, zorder=3,
               label="Donut + DiT", edgecolors=SURFACE, linewidths=1.5)
    ax.scatter([three[i] for i in order], range(len(order)), color=ORANGE, marker="s", s=46, zorder=3,
               label="Donut + DiT + Teks", edgecolors=SURFACE, linewidths=1.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([CLASSES[i] for i in order], color=INK)
    ax.set_xlim(0.76, 1.01)
    ax.set_xlabel("F1 pada holdout (angka di kanan = perubahan dalam poin)", color=INK2)
    ax.set_title("Pengaruh model teks terhadap F1 per kelas", color=INK, loc="left", fontsize=12, pad=26)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, "text_gain_per_class.png")


def plot_ocr_words() -> None:
    """Grafik median jumlah kata OCR per kelas (dari ringkasan notebook 04)."""
    words = {"file_folder": 2.0, "presentation": 59.0, "advertisement": 70.0, "handwritten": 72.0, "email": 78.0,
             "budget": 94.0, "invoice": 115.5, "scientific_report": 132.0, "form": 134.0, "memo": 186.0,
             "letter": 193.0, "specification": 201.0, "questionnaire": 232.0, "resume": 321.0,
             "news_article": 405.0, "scientific_publication": 635.0}
    names, vals = list(words), list(words.values())
    fig, ax = plt.subplots(figsize=(8.5, 6))
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(names, vals, color=BLUE, height=0.65)
    for i, v in enumerate(vals):
        ax.text(v + 6, i, f"{v:g}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Median jumlah kata OCR per halaman", color=INK2)
    ax.set_title("Jumlah teks hasil OCR per kelas", color=INK, loc="left", fontsize=12)
    ax.tick_params(axis="y", colors=INK)
    save(fig, "ocr_words_per_class.png")


def plot_class_distribution() -> None:
    """Grafik jumlah halaman per kelas pada data train (dari log notebook 02)."""
    counts = {"questionnaire": 1948, "news_article": 1971, "letter": 1972, "specification": 1978, "invoice": 1982,
              "presentation": 1992, "memo": 1994, "scientific_report": 1999, "budget": 2004, "form": 2005,
              "advertisement": 2012, "email": 2013, "file_folder": 2022, "handwritten": 2026, "resume": 2029,
              "scientific_publication": 2056}
    fig, ax = plt.subplots(figsize=(8.5, 6))
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(list(counts), list(counts.values()), color=BLUE, height=0.65)
    for i, v in enumerate(counts.values()):
        ax.text(v + 15, i, f"{v:,}".replace(",", "."), va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 2350)
    ax.set_xlabel("Jumlah halaman", color=INK2)
    ax.set_title("Distribusi kelas pada data train (32.003 halaman)", color=INK, loc="left", fontsize=12)
    ax.tick_params(axis="y", colors=INK)
    save(fig, "class_distribution.png")


def main() -> None:
    """Membuat semua grafik laporan."""
    plot_class_distribution()
    plot_score_progression()
    plot_training_curves()
    plot_model_complementarity()
    plot_text_gain()
    plot_ocr_words()
    print(f"Grafik disimpan di {FIG}")


if __name__ == "__main__":
    main()
