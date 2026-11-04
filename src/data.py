"""Pembacaan data, audit kualitas gambar, split holdout, Dataset, dan DataLoader.

Prinsip pengolahan gambar di proyek ini:

- Gambar **tidak dibersihkan** (tanpa denoising, binarisasi, atau crop border),
  karena data train dan test punya kondisi scan yang sama. Noise, border hitam,
  dan ruang kosong justru bisa menjadi ciri kelas (misalnya ``file_folder``).
- Downsampling memakai filter yang antialiased (BOX/BILINEAR di Pillow) supaya
  goresan tipis dari scan 1-bit tetap terlihat sebagai piksel abu-abu.
- Augmentasi dibuat ringan dan mirip variasi scan (rotasi kecil, geser,
  kontras). Tidak ada horizontal flip karena akan membalik teks.
"""

from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageStat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.utils import find_file


def resolve_path(root: Path, rel: str) -> Optional[str]:
    """Mengubah path relatif dari CSV menjadi path file yang benar-benar ada.

    ``train.csv`` menulis path seperti ``train/invoice/x.tif``, sedangkan
    setelah diekstrak foldernya menjadi ``train/train/invoice/x.tif``.
    Fungsi ini menangani kedua kemungkinan tersebut.

    Args:
        root: Folder yang berisi ``train.csv``.
        rel: Path relatif dari kolom ``filename``.

    Returns:
        Path absolut sebagai string, atau ``None`` jika file tidak ditemukan.
    """
    rel = str(rel)
    for cand in (root / rel, root / Path(rel).parts[0] / rel):
        if cand.exists():
            return str(cand)
    return None


def load_metadata(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Membaca ``train.csv``, ``test.csv``, dan ``sample_submission.csv``.

    Args:
        data_dir: Folder tempat data kompetisi diekstrak.

    Returns:
        Tuple ``(train_df, test_df, sample_sub, classes)``. ``train_df`` dan
        ``test_df`` mendapat kolom ``path`` berisi lokasi file gambar.
        ``classes`` adalah nama 16 kelas, urut sesuai label 0–15.

    Raises:
        FileNotFoundError: Jika ada file test yang tidak ditemukan.
        ValueError: Jika pemetaan ``class_name`` <-> ``label`` tidak satu-satu.
    """
    root = find_file("train.csv", data_dir).parent
    train_df = pd.read_csv(root / "train.csv")
    test_df = pd.read_csv(root / "test.csv")
    sample_sub = pd.read_csv(root / "sample_submission.csv")

    train_df["path"] = [resolve_path(root, f) for f in train_df["filename"]]
    test_df["path"] = [resolve_path(root, f) for f in test_df["filename"]]
    if test_df["path"].isna().any():
        raise FileNotFoundError(f"{int(test_df['path'].isna().sum())} file test tidak ditemukan")
    train_df = train_df[train_df["path"].notna()].reset_index(drop=True)

    cls_map = train_df.groupby("label")["class_name"].agg(lambda s: s.unique().tolist())
    if not all(len(v) == 1 for v in cls_map):
        raise ValueError("Pemetaan class_name <-> label tidak satu-satu")
    classes = [cls_map[i][0] for i in sorted(cls_map.index)]
    return train_df, test_df, sample_sub, classes


def _audit_one(path: str) -> Tuple[str, bool, int, int, float, float]:
    """Memeriksa satu file gambar (dipanggil paralel oleh :func:`audit_images`).

    Args:
        path: Path file gambar.

    Returns:
        ``(path, bisa_dibaca, lebar, tinggi, rata2_piksel, std_piksel)``.
    """
    try:
        with Image.open(path) as im:
            im.seek(0)
            w, h = im.size
            g = im.convert("L")
            g.thumbnail((256, 256))
            st = ImageStat.Stat(g)
            return path, True, w, h, st.mean[0], st.stddev[0]
    except Exception:
        return path, False, 0, 0, float("nan"), float("nan")


def audit_images(paths: Sequence[str], num_workers: int = 4) -> pd.DataFrame:
    """Audit kualitas file gambar: bisa dibaca atau tidak, ukuran, dan kontras.

    Hanya file yang rusak atau benar-benar kosong (std piksel ~0) yang
    nantinya dibuang dari data train. Halaman yang sulit tapi valid tetap dipakai.

    Args:
        paths: Daftar path gambar.
        num_workers: Jumlah proses paralel.

    Returns:
        DataFrame dengan kolom ``path, ok, w, h, mean, std``.
    """
    with Pool(num_workers) as pool:
        rows = list(tqdm(pool.imap(_audit_one, paths, chunksize=64), total=len(paths), desc="audit"))
    return pd.DataFrame(rows, columns=["path", "ok", "w", "h", "mean", "std"])


def split_holdout(train_df: pd.DataFrame, holdout_file: Path, val_frac: float = 0.10,
                  seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Membagi data train menjadi train dan holdout (validasi) secara stratified.

    Jika ``holdout_file`` sudah ada, id holdout dibaca dari file itu. Dengan
    begitu semua model dievaluasi pada gambar holdout yang **persis sama**,
    sehingga probabilitasnya bisa digabung dan bobot ensemble bisa diatur.

    Args:
        train_df: Data train lengkap.
        holdout_file: File CSV berisi kolom ``id`` milik holdout.
        val_frac: Proporsi holdout jika split dibuat baru.
        seed: Seed untuk split.

    Returns:
        Tuple ``(tr_df, va_df)``.
    """
    holdout_file = Path(holdout_file)
    if holdout_file.exists():
        ids = pd.read_csv(holdout_file)["id"].tolist()
        va_df = train_df.set_index("id").loc[ids].reset_index()
        tr_df = train_df[~train_df["id"].isin(set(ids))].reset_index(drop=True)
    else:
        tr_df, va_df = train_test_split(train_df, test_size=val_frac, stratify=train_df["label"],
                                        random_state=seed)
        tr_df, va_df = tr_df.reset_index(drop=True), va_df.reset_index(drop=True)
        holdout_file.parent.mkdir(parents=True, exist_ok=True)
        va_df[["id", "label", "class_name"]].to_csv(holdout_file, index=False)
    return tr_df, va_df


def load_page(path: str, size: Tuple[int, int], resample=Image.Resampling.BILINEAR) -> Image.Image:
    """Membaca halaman pertama file TIFF sebagai grayscale lalu mengubah ukurannya.

    File yang tidak bisa dibaca diganti gambar putih, supaya setiap halaman
    test tetap mendapat prediksi.

    Args:
        path: Path file gambar.
        size: Ukuran tujuan ``(lebar, tinggi)``.
        resample: Filter resize Pillow (BILINEAR untuk Donut, BOX untuk DiT).

    Returns:
        Gambar PIL mode ``"L"`` berukuran ``size``.
    """
    try:
        with Image.open(path) as im:
            im.seek(0)
            im = im.convert("L")
    except Exception:
        im = Image.new("L", size, 255)
    return im.resize(size, resample)


def _load_small(args: Tuple[str, int]) -> Tuple[np.ndarray, bool]:
    """Membaca satu halaman kecil berbentuk persegi (dipanggil paralel oleh :func:`cache_images`).

    Args:
        args: Tuple ``(path, ukuran_sisi)``.

    Returns:
        ``(array_uint8, bisa_dibaca)``.
    """
    path, side = args
    try:
        with Image.open(path) as im:
            im.seek(0)
            im = im.convert("L")
            return np.asarray(im.resize((side, side), Image.Resampling.BOX), dtype=np.uint8), True
    except Exception:
        return np.full((side, side), 255, dtype=np.uint8), False


def cache_images(paths: Sequence[str], side: int = 224, num_workers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Membaca semua halaman sekali ke RAM dalam ukuran kecil.

    Untuk DiT (224 px), 40 ribu halaman hanya butuh sekitar 2 GB. Dengan cache
    ini training tidak lagi tertahan oleh proses decode TIFF.

    Args:
        paths: Daftar path gambar.
        side: Panjang sisi gambar hasil (persegi).
        num_workers: Jumlah proses paralel.

    Returns:
        Tuple ``(array_gambar, mask_ok)`` dengan bentuk ``(n, side, side)`` dan ``(n,)``.
    """
    with Pool(num_workers) as pool:
        res = list(tqdm(pool.imap(_load_small, [(p, side) for p in paths], chunksize=64),
                        total=len(paths), desc="cache"))
    return np.stack([r[0] for r in res]), np.array([r[1] for r in res])


def build_train_aug(blur: bool = True) -> T.Compose:
    """Membuat augmentasi ringan yang meniru variasi hasil scan.

    Args:
        blur: Tambahkan Gaussian blur kecil (dipakai untuk Donut, tidak untuk DiT 224 px).

    Returns:
        Transformasi torchvision yang bisa dipakai pada gambar PIL atau tensor.
    """
    ops = [
        T.RandomApply([T.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=255)], p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
    ]
    if blur:
        ops.append(T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1))
    return T.Compose(ops)


def _to_model_tensor(x: torch.Tensor) -> torch.Tensor:
    """Normalisasi tensor 1 kanal [0, 1] menjadi 3 kanal [-1, 1] (format Donut dan DiT).

    Args:
        x: Tensor berukuran ``(1, H, W)``.

    Returns:
        Tensor berukuran ``(3, H, W)``.
    """
    return ((x - 0.5) / 0.5).expand(3, -1, -1).contiguous()


class DocDataset(Dataset):
    """Dataset yang membaca gambar dari disk setiap kali diakses.

    Dipakai untuk Donut (resolusi 1024x768), karena gambar sebesar itu tidak
    muat jika semuanya disimpan di RAM.
    """

    def __init__(self, df: pd.DataFrame, size: Tuple[int, int], train: bool = False):
        """Membuat dataset.

        Args:
            df: DataFrame dengan kolom ``path`` dan (opsional) ``label``.
            size: Ukuran gambar ``(lebar, tinggi)``.
            train: Jika ``True``, augmentasi diterapkan.
        """
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist() if "label" in df else [-1] * len(df)
        self.size = size
        self.aug = build_train_aug(blur=True) if train else None

    def __len__(self) -> int:
        """Mengembalikan jumlah halaman."""
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        """Mengambil satu halaman.

        Args:
            i: Indeks halaman.

        Returns:
            Tuple ``(tensor_gambar, label)``; label ``-1`` untuk data test.
        """
        im = load_page(self.paths[i], self.size)
        if self.aug is not None:
            im = self.aug(im)
        return _to_model_tensor(TF.to_tensor(im)), self.labels[i]


class ArrayDataset(Dataset):
    """Dataset yang membaca gambar dari array di RAM (hasil :func:`cache_images`).

    Dipakai untuk DiT (224 px), sehingga training dibatasi kecepatan GPU,
    bukan kecepatan decode file TIFF.
    """

    def __init__(self, arr: np.ndarray, labels: Sequence[int], train: bool = False):
        """Membuat dataset.

        Args:
            arr: Array gambar ``(n, H, W)`` bertipe uint8.
            labels: Label per halaman; ``-1`` untuk data test.
            train: Jika ``True``, augmentasi diterapkan.
        """
        self.arr = arr
        self.labels = list(labels)
        self.aug = T.Compose([
            T.RandomApply([T.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=1.0)], p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
        ]) if train else None

    def __len__(self) -> int:
        """Mengembalikan jumlah halaman."""
        return len(self.arr)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        """Mengambil satu halaman.

        Args:
            i: Indeks halaman.

        Returns:
            Tuple ``(tensor_gambar, label)``.
        """
        x = torch.from_numpy(self.arr[i]).float().div_(255).unsqueeze(0)
        if self.aug is not None:
            x = self.aug(x)
        return _to_model_tensor(x), self.labels[i]


def make_loader(dataset: Dataset, batch_size: int, train: bool, num_workers: int = 4) -> DataLoader:
    """Membuat DataLoader dengan pengaturan standar proyek ini.

    Args:
        dataset: Dataset yang akan dibungkus.
        batch_size: Ukuran batch.
        train: Jika ``True``, data diacak dan batch terakhir yang tidak penuh dibuang.
        num_workers: Jumlah worker pemuat data.

    Returns:
        DataLoader siap pakai.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=train,
                      persistent_workers=num_workers > 0)
