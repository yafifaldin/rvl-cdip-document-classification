"""Membaca konfigurasi eksperimen dari file YAML di folder ``configs/``.

Semua hyperparameter disimpan di file YAML supaya eksperimen mudah direproduksi
dan tidak ada angka "ajaib" yang tersebar di dalam kode.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> Dict[str, Any]:
    """Membaca satu file konfigurasi YAML.

    Args:
        name: Nama konfigurasi tanpa ekstensi (misalnya ``"donut"``, ``"dit"``,
            ``"ocr"``, ``"text"``) atau path lengkap ke file ``.yaml``.

    Returns:
        Dictionary berisi pasangan nama hyperparameter dan nilainya.

    Raises:
        FileNotFoundError: Jika file konfigurasi tidak ditemukan.
    """
    path = Path(name)
    if path.suffix != ".yaml":
        path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
