# -*- coding: utf-8 -*-
"""
finetune.py — Strategi fine-tuning untuk ConvNeXt-Tiny (timm) pada eksperimen L1.
=================================================================================
Ditulis agar cocok dengan pipeline eginugraha/writer-identification:
  - Model dibuat via timm.create_model("convnext_tiny", pretrained=True, num_classes=...)
  - Optimizer baseline: AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
  - Scheduler: LinearLR warmup (3 ep) -> CosineAnnealingLR  (TIDAK perlu diubah)
  - Loss: CrossEntropyLoss  (S4 menambah label_smoothing=0.1)

Cara pakai: taruh file ini di  src/cvl/finetune.py  lalu ikuti PATCH di bawah.

Strategi:
  S0  baseline          : fine-tune penuh, LR seragam (kode asli, tanpa perubahan)
  S1  feature-extract   : bekukan SELURUH backbone, latih head saja
  S2  selective         : bekukan stem + stages 0-1, latih stages 2-3 + head, LR 1e-4
  S3  selective + LLRD  : seperti S2 + layer-wise LR decay (head 1e-4, turun 0.7/stage)
  S4  S3 + smoothing    : S3 + label_smoothing=0.1 (ubah loss, lihat PATCH)

------------------------------------------------------------------------------
PATCH untuk src/cvl/train.py  (pakai env-var CVL_FINETUNE, sesuai desain pipeline)
------------------------------------------------------------------------------
  import os
  from .finetune import freeze_layers, build_param_groups, STRATEGIES

  strategy = os.environ.get("CVL_FINETUNE", "S0")     # default S0 = perilaku lama

  # --- SEBELUM (baseline):
  # opt = torch.optim.AdamW(model.parameters(), lr=rc.lr, weight_decay=rc.weight_decay)

  # --- SESUDAH:
  freeze_layers(model, strategy)
  opt = torch.optim.AdamW(
      build_param_groups(model, strategy, base_lr=rc.lr),
      weight_decay=rc.weight_decay,
  )

  # --- ganti pembuatan loss (agar S4 aktif; S0-S3 label_smoothing=0.0):
  #   crit = torch.nn.CrossEntropyLoss()
  # menjadi:
  #   crit = torch.nn.CrossEntropyLoss(
  #       label_smoothing=STRATEGIES[os.environ.get("CVL_FINETUNE","S0")]["label_smoothing"])

TIDAK perlu ubah RunConfig/yaml — strategi dipilih via env-var CVL_FINETUNE=S3.
Scheduler, early stopping, AMP, augmentasi: BIARKAN seperti aslinya.
------------------------------------------------------------------------------
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Definisi strategi
#   base_lr   : None = pakai rc.lr apa adanya (S0/S1); angka = override (S2-S4)
#   llrd_decay: faktor pengali LR per level ke arah lapisan awal (None = seragam)
# ---------------------------------------------------------------------------
STRATEGIES = {
    "S0": dict(freeze=(),                                   base_lr=None,  llrd_decay=None, label_smoothing=0.0),
    "S1": dict(freeze=("stem", "stages.0", "stages.1",
                       "stages.2", "stages.3"),             base_lr=None,  llrd_decay=None, label_smoothing=0.0),
    "S2": dict(freeze=("stem", "stages.0", "stages.1"),     base_lr=1e-4,  llrd_decay=None, label_smoothing=0.0),
    "S3": dict(freeze=("stem", "stages.0", "stages.1"),     base_lr=1e-4,  llrd_decay=0.7,  label_smoothing=0.0),
    "S4": dict(freeze=("stem", "stages.0", "stages.1"),     base_lr=1e-4,  llrd_decay=0.7,  label_smoothing=0.1),
}

# Urutan level ConvNeXt-Tiny dari kepala ke lapisan awal — untuk LLRD.
# head mendapat base_lr, tiap langkah ke bawah dikali llrd_decay.
_LLRD_ORDER = ("head", "stages.3", "stages.2", "stages.1", "stages.0", "stem")


def _prefix_of(param_name: str) -> str:
    """Kembalikan prefix level LLRD untuk sebuah nama parameter timm ConvNeXt."""
    for pfx in _LLRD_ORDER:
        if param_name.startswith(pfx):
            return pfx
    # norm_pre / lain-lain: perlakukan seperti head (LR penuh)
    return "head"


def freeze_layers(model, strategy: str = "S0") -> None:
    """Set requires_grad sesuai strategi. Aman dipanggil untuk semua strategi."""
    cfg = STRATEGIES[strategy]
    for name, p in model.named_parameters():
        p.requires_grad = not name.startswith(cfg["freeze"])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[finetune] {strategy}: trainable {n_train/1e6:.1f}M / {n_total/1e6:.1f}M param")


def build_param_groups(model, strategy: str = "S0", base_lr: float = 3e-4):
    """
    Kembalikan param_groups untuk AdamW.
      - S0/S1: satu grup, LR = base_lr (perilaku baseline).
      - S2   : satu grup, LR = 1e-4.
      - S3/S4: grup per level dengan LLRD (head 1e-4; stages.3 x0.7; stages.2 x0.49; dst).
    Hanya parameter requires_grad=True yang dimasukkan.
    """
    cfg = STRATEGIES[strategy]
    lr0 = cfg["base_lr"] if cfg["base_lr"] is not None else base_lr

    if cfg["llrd_decay"] is None:
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "lr": lr0}]

    decay = cfg["llrd_decay"]
    lr_of = {pfx: lr0 * (decay ** i) for i, pfx in enumerate(_LLRD_ORDER)}
    buckets = {pfx: [] for pfx in _LLRD_ORDER}
    for name, p in model.named_parameters():
        if p.requires_grad:
            buckets[_prefix_of(name)].append(p)

    groups = [{"params": ps, "lr": lr_of[pfx]} for pfx, ps in buckets.items() if ps]
    print("[finetune] LLRD LR:", {pfx: f"{lr_of[pfx]:.1e}" for pfx, ps in buckets.items() if ps})
    return groups


# ---------------------------------------------------------------------------
# Uji cepat tanpa data:  python -m src.cvl.finetune
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import timm, torch

    for strat in STRATEGIES:
        model = timm.create_model("convnext_tiny", pretrained=False, num_classes=308)
        freeze_layers(model, strat)
        groups = build_param_groups(model, strat, base_lr=3e-4)
        opt = torch.optim.AdamW(groups, weight_decay=0.05)
        x = torch.randn(2, 3, 224, 224)
        loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([1, 2]))
        loss.backward(); opt.step()
        print(f"  {strat}: OK ({len(groups)} param group)\n")
    print("Semua strategi lolos smoke-test.")
