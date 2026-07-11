# Ensiklopedia

Referensi ringkas untuk tesis perbandingan arsitektur *writer identification* pada dataset CVL. Tiap file menjelaskan satu arsitektur (apa itu, sejarah, algoritma, dan bagaimana ia muncul di kode repo ini), plus analisis dataset.

## Isi

| File | Topik |
|---|---|
| [resnet-50.md](resnet-50.md) | ResNet-50 — CNN residual (baseline klasik) |
| [convnext-tiny.md](convnext-tiny.md) | ConvNeXt-Tiny — CNN modern ala transformer (2022) |
| [efficientnetv2-s.md](efficientnetv2-s.md) | EfficientNetV2-S — CNN efisien, cepat dilatih |
| [vit-small.md](vit-small.md) | ViT-Small — Vision Transformer murni |
| [swin-tiny.md](swin-tiny.md) | Swin-Tiny — Transformer hierarkis berjendela |
| [dataset.md](dataset.md) | Analisis dataset CVL (jumlah, ukuran, split train/val/test) |

## Konteks eksperimen

Kelima arsitektur dibandingkan pada dua sumbu:

- **Pretrained vs from-scratch** — pakai bobot ImageNet vs inisialisasi acak.
- **Data terbatas (ablasi)** — jumlah halaman latih per penulis dibatasi 1/2/3/4/penuh.

Grid penuh: `5 arsitektur × 5 level × 2 mode × 3 seed = 150 run`. Semua arsitektur dibangun lewat [`timm`](https://github.com/huggingface/pytorch-image-models) di `src/cvl/models.py`, dengan input **224×224** dan kepala klasifikasi **308 kelas** (jumlah penulis). Angka parameter & throughput aktual tiap arsitektur tercatat di `results/results.csv` (`n_params`, `throughput_img_s`).
