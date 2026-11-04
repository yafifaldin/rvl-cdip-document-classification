"""Arsitektur model gambar dan pengelompokan parameter untuk optimizer.

Dua model gambar yang dipakai:

1. **Donut encoder classifier**: encoder Swin Transformer dari Donut
   (``naver-clova-ix/donut-base``), dipretrain pada jutaan dokumen. Decoder
   teksnya dibuang; di atas encoder ditambahkan mean pooling dan head linear.
   Dipakai pada resolusi tinggi (1024x768) agar detail teks masih terbaca.
2. **DiT-base** (``microsoft/dit-base``): Vision Transformer (BEiT) yang
   dipretrain secara self-supervised pada 42 juta halaman dokumen IIT-CDIP,
   sumber yang sama dengan RVL-CDIP. Dipakai pada 224 px.

Checkpoint ``*-finetuned-rvlcdip`` sengaja **tidak** dipakai, karena sudah
dilatih pada RVL-CDIP penuh yang kemungkinan besar memuat gambar test kompetisi.
"""

import gc
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, VisionEncoderDecoderModel


class DonutClassifier(nn.Module):
    """Encoder Donut (Swin) + mean pooling + head linear untuk klasifikasi."""

    def __init__(self, pretrained: str, num_classes: int, gradient_checkpointing: bool = True):
        """Membuat model.

        Args:
            pretrained: Nama atau path checkpoint Donut di Hugging Face.
            num_classes: Jumlah kelas output.
            gradient_checkpointing: Hemat memori GPU dengan menghitung ulang
                aktivasi saat backward (sekitar 30% lebih lambat).
        """
        super().__init__()
        ved = VisionEncoderDecoderModel.from_pretrained(pretrained)
        self.encoder = ved.encoder
        del ved
        gc.collect()
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        hid = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(hid), nn.Dropout(0.1), nn.Linear(hid, num_classes))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: Batch gambar ``(B, 3, H, W)`` yang sudah dinormalisasi.

        Returns:
            Logit berukuran ``(B, num_classes)``.
        """
        h = self.encoder(pixel_values=pixel_values).last_hidden_state   # (B, token, hidden)
        return self.head(h.mean(dim=1))


class DiTClassifier(nn.Module):
    """Pembungkus DiT (BEiT) dari Hugging Face agar antarmukanya sama dengan :class:`DonutClassifier`."""

    def __init__(self, pretrained: str, num_classes: int, drop_path: float = 0.1):
        """Membuat model dengan head klasifikasi baru.

        Args:
            pretrained: Nama atau path checkpoint DiT.
            num_classes: Jumlah kelas output.
            drop_path: Stochastic depth rate untuk regularisasi.
        """
        super().__init__()
        self.net = AutoModelForImageClassification.from_pretrained(
            pretrained, num_labels=num_classes, ignore_mismatched_sizes=True, drop_path_rate=drop_path)

    @property
    def config(self):
        """Konfigurasi Hugging Face dari model DiT."""
        return self.net.config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: Batch gambar ``(B, 3, 224, 224)``.

        Returns:
            Logit berukuran ``(B, num_classes)``.
        """
        return self.net(pixel_values=pixel_values).logits


def _no_weight_decay(name: str, p: torch.Tensor) -> bool:
    """Menentukan apakah sebuah parameter dikecualikan dari weight decay.

    Bias, parameter normalisasi, embedding posisi, token CLS, dan layer scale
    tidak diberi weight decay (praktik standar untuk Transformer).

    Args:
        name: Nama parameter.
        p: Tensor parameter.

    Returns:
        ``True`` jika parameter tidak perlu weight decay.
    """
    return (p.ndim == 1 or name.endswith(".bias") or "norm" in name.lower()
            or "position" in name or "cls_token" in name or "lambda" in name)


def donut_param_groups(model: DonutClassifier, lr_encoder: float, lr_head: float,
                       weight_decay: float) -> List[Dict]:
    """Grup parameter Donut: LR kecil untuk encoder pretrained, LR besar untuk head baru.

    Args:
        model: Model Donut.
        lr_encoder: Learning rate encoder.
        lr_head: Learning rate head.
        weight_decay: Weight decay untuk parameter yang memakai decay.

    Returns:
        List grup parameter untuk ``torch.optim.AdamW``.
    """
    enc_decay, enc_nodecay, head = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("head"):
            head.append(p)
        elif _no_weight_decay(n, p):
            enc_nodecay.append(p)
        else:
            enc_decay.append(p)
    return [
        {"params": enc_decay, "lr": lr_encoder, "weight_decay": weight_decay},
        {"params": enc_nodecay, "lr": lr_encoder, "weight_decay": 0.0},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay},
    ]


def llrd_param_groups(model: DiTClassifier, lr: float, layer_decay: float,
                      weight_decay: float) -> List[Dict]:
    """Grup parameter dengan layer-wise learning rate decay (LLRD) untuk DiT.

    Layer bawah (dekat input) menyimpan fitur umum hasil pretraining, sehingga
    diberi LR kecil. Layer atas dan head diberi LR lebih besar agar cepat
    beradaptasi. LR layer ke-i = ``lr * layer_decay ** (n_layer + 1 - i)``.

    Args:
        model: Model DiT.
        lr: Learning rate untuk layer teratas dan head.
        layer_decay: Faktor pengali per layer (misalnya 0.75).
        weight_decay: Weight decay untuk parameter yang memakai decay.

    Returns:
        List grup parameter untuk ``torch.optim.AdamW``.
    """
    n_layers = model.config.num_hidden_layers
    groups: Dict = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".encoder.layer." in n:
            lid = int(n.split(".encoder.layer.")[1].split(".")[0]) + 1
        elif ".embeddings." in n:
            lid = 0
        else:
            lid = n_layers + 1                    # pooler, classifier, relative position bias bersama
        no_wd = _no_weight_decay(n, p)
        key = (lid, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * layer_decay ** (n_layers + 1 - lid),
                           "weight_decay": 0.0 if no_wd else weight_decay}
        groups[key]["params"].append(p)
    return list(groups.values())


def build_model_and_groups(cfg: Dict, num_classes: int):
    """Membangun model gambar dan grup parameter optimizer sesuai konfigurasi.

    Args:
        cfg: Konfigurasi dari ``configs/donut.yaml`` atau ``configs/dit.yaml``.
        num_classes: Jumlah kelas output.

    Returns:
        Tuple ``(model, param_groups)``.

    Raises:
        ValueError: Jika ``cfg["model"]`` bukan ``"donut"`` atau ``"dit"``.
    """
    if cfg["model"] == "donut":
        model = DonutClassifier(cfg["pretrained"], num_classes, cfg.get("gradient_checkpointing", True))
        groups = donut_param_groups(model, cfg["lr_encoder"], cfg["lr_head"], cfg["weight_decay"])
    elif cfg["model"] == "dit":
        model = DiTClassifier(cfg["pretrained"], num_classes, cfg.get("drop_path", 0.1))
        groups = llrd_param_groups(model, cfg["lr"], cfg["layer_decay"], cfg["weight_decay"])
    else:
        raise ValueError(f"Model tidak dikenal: {cfg['model']}")
    return model, groups
