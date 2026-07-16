# Panduan Eksperimen Fine-Tuning ConvNeXt-Tiny pada L1

Paket: `finetune.py` (sudah lolos smoke-test S0–S4) + patch 3 baris di `train.py`.

## 1. Tujuan
Menguji apakah **strategi fine-tuning yang tepat** menaikkan akurasi ConvNeXt di L1
(1 halaman/penulis) tanpa menambah data. **Baseline pembanding: Top-1 = 0,768.**

## 2. Strategi yang diuji

| Kode | Strategi | Freeze | LR | LLRD | Label smoothing |
|------|----------|--------|----|------|-----------------|
| S0 | Baseline (kode asli) | — | 3e-4 | — | — |
| S1 | Feature extraction | semua backbone | 3e-4 (head) | — | — |
| S2 | Selective | stem + stages 0–1 | 1e-4 | — | — |
| S3 | Selective + LLRD | stem + stages 0–1 | 1e-4 | 0,7/stage | — |
| S4 | S3 + smoothing | stem + stages 0–1 | 1e-4 | 0,7/stage | 0,1 |

LLRD (S3/S4): head 1,0e-4 → stages.3 7,0e-5 → stages.2 4,9e-5 (terverifikasi smoke-test).

## 3. Integrasi (kerja rekan ±10 menit)

1. Salin `finetune.py` → `src/cvl/finetune.py`
2. Di `src/cvl/train.py`, ganti baris optimizer:

```python
from .finetune import freeze_layers, build_param_groups, STRATEGIES

strategy = getattr(rc, "finetune_strategy", "S0")
freeze_layers(model, strategy)
opt = torch.optim.AdamW(
    build_param_groups(model, strategy, base_lr=rc.lr),
    weight_decay=rc.weight_decay,
)
# khusus S4:
# crit = torch.nn.CrossEntropyLoss(label_smoothing=STRATEGIES[strategy]["label_smoothing"])
```

3. Tambah `finetune_strategy: "S0"` di `configs/default.yaml` (ubah per run).
4. **Jangan ubah**: scheduler (warmup+cosine), early stopping (patience 8), AMP, augmentasi, val_frac 0,1 — supaya perbandingan adil.

## 4. Matriks run (fokus, hemat)

- Arsitektur: **convnext_tiny saja** · Level: **L1 saja** · Seed: **0, 1, 2**
- Jalur cepat (±1 jam GPU): S0 & S3 × seed 0,1
- Jalur penuh (±3–4 jam GPU): S0–S4 × seed 0,1,2 (15 run)

Catat per run: Top-1, Top-5, Macro-F1, waktu latih, epoch berhenti (early stop).

## 5. Catatan hardware

| Mesin | Penyesuaian |
|-------|-------------|
| RunPod RTX 4000 Ada ($0.26/jam) | tanpa perubahan; batch 64 aman (20GB) |
| RTX 5060 8GB | **PyTorch terbaru CUDA 12.8+** wajib; `batch_size: 32` |
| Windows | jika DataLoader error → `num_workers: 2` atau 0 |

Smoke-test tanpa data: `python -m src.cvl.finetune` (harus cetak "Semua strategi lolos").

## 6. Membaca hasil (anti-overclaim)

- **Naik** jika Top-1 rata-rata > 0,768 dan selisihnya melebihi simpangan baku antar seed.
- Jika **tidak naik**: tetap temuan valid — "strategi fine-tuning tidak mengatasi
  keterbatasan data di L1; jumlah data adalah kendala utama."
- Laporkan selalu rata-rata ± std (3 seed), jangan angka seed tunggal terbaik.
