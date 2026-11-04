"""Paket utama untuk klasifikasi dokumen RVL-CDIP (16 kelas).

Modul di dalam paket ini:

- ``config``     : membaca file konfigurasi YAML di folder ``configs/``.
- ``utils``      : fungsi bantu umum (seed, device, pencarian file, analisis error).
- ``data``       : membaca metadata, audit gambar, split holdout, Dataset dan DataLoader.
- ``models``     : arsitektur Donut encoder classifier dan DiT, beserta grup parameter optimizer.
- ``engine``     : loop training, prediksi, dan penyimpanan checkpoint.
- ``ocr``        : ekstraksi teks dan bounding box dengan docTR.
- ``text_model`` : fitur TF-IDF dan model teks Logistic Regression.
- ``ensemble``   : penggabungan probabilitas model dan pencarian bobot.

Script yang bisa dijalankan dari terminal (``python -m src.<nama>``):
``train_image``, ``run_ocr``, ``train_text``, ``make_ensemble``.
"""

__version__ = "1.0.0"
