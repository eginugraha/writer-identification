# Riset Literatur: Perbandingan Arsitektur untuk Writer Identification

> Dibuat: 2026-07-07 (sesi 2). Hasil 3 agen riset paralel + verifikasi web.
> Tujuan: cek duplikasi & petakan celah untuk skripsi S1 (perbandingan arsitektur + kondisi menantang).

## RINGKASAN EKSEKUTIF (yang mengubah keputusan)

1. **CVL benar-benar jenuh** (~99% Top-1 / ~98% mAP) — dikonfirmasi. Raven & Fink 2024 bahkan capai **98.6% mAP di CVL TANPA fine-tuning** (ViT self-supervised dilatih di data lain). Jadi "banding arsitektur di CVL bersih" → semua mentok ~99%, margin tipis, kurang menarik.

2. **Perbandingan CNN vs Transformer untuk writer-ID SUDAH ADA sebagian** — jadi harus hati-hati duplikasi:
   - **Swin vs ResNeSt-50 di CVL** (ELCVIA 2024): Swin 98.5% vs ResNeSt 96.6%. ← sudah ada di CVL!
   - **SwinV2 vs ResNet18 vs EfficientNetV2** (Beyond the Pipeline, 2025): tapi di dataset **historis**, bukan CVL. ConvNeXt tidak diikutkan.
   - **VGG/ResNet50/MobileNet/Xception/EfficientNet** di IAM & CVL (JOIV 2023): semua CNN, **tanpa ViT**.
   - **ConvNeXt belum pernah** dibanding head-to-head vs ResNet/ViT untuk writer-ID → celah nyata.

3. **KOREKSI PENTING soal cross-dataset CVL↔IAM:** ini TIDAK sekuat dugaan awal.
   - CVL & IAM sama-sama **modern, domain mirip** (layout CVL meniru IAM). Transfer fitur antar keduanya **bagus, drop kecil** (bukti: ViT zero-shot 98.6% di CVL).
   - Jadi "train CVL → test IAM" **tidak akan menunjukkan penurunan besar** → kondisi "menantang"-nya jadi kurang menantang.
   - Drop besar justru di **modern → historis** (CVL/IAM → Historical-WI/HisIR19: mAP 98% → 83–95%).
   - Cross-dataset **wajib retrieval/verification** (identitas penulis beda antar dataset → tak bisa klasifikasi closed-set). CVL & IAM **tidak overlap penulis** (dikoleksi terpisah: Wina vs Bern).

4. **CELAH TERKUAT yang muncul — data terbatas + perbandingan arsitektur:**
   - **Belum ada** studi writer-ID yang secara sistematis **mengurangi jumlah sampel per penulis** (5→3→2→1) sambil membanding CNN vs Transformer head-to-head.
   - Ada **ketegangan literatur** yang bisa diuji: ViT menang di writer-ID **hanya dengan pretraining besar/self-supervised**; kalau dilatih from-scratch di data kecil, **CNN/ConvNeXt lebih unggul** (ViT "data-hungry" — banyak bukti umum).
   - Ini **feasible S1** (dataset publik, arsitektur pretrained di timm/torchvision) dan menjawab tantangan yang eksplisit disebut survei.

## IMPLIKASI ARAH SKRIPSI

- Sudut "cross-dataset CVL↔IAM" saja → **lemah** (drop kecil). Perlu diperkuat/diganti.
- Sudut **"data terbatas + perbandingan arsitektur (termasuk ConvNeXt) + peran pretraining"** → **celah nyata, orisinal, feasible**.
- Kombinasi kuat: banding ResNet vs ConvNeXt vs ViT/Swin di CVL, **dengan ablasi data terbatas** sebagai kondisi menantang utama, cross-dataset (ke IAM dan/atau historis) sebagai uji generalisasi sekunder.

## PAPER KUNCI (untuk Bab 2)

| Paper | Tahun | Arsitektur | Dataset | Hasil | URL |
|---|---|---|---|---|---|
| Beyond the Pipeline (Rasyidi & Farazi) | 2025 | SwinV2 vs ResNet18 vs EffNetV2 | ICDAR2013/2017/2019 (historis) | Swin ArcFace 97.4% Top-1; ResNet18 generalisasi buruk | https://arxiv.org/abs/2510.18671 |
| Swin vs ResNeSt-50 | 2024 | Swin vs ResNeSt-50 | **CVL** | Swin 98.5% (patch) vs ResNeSt 96.6% (page) | https://elcvia.cvc.uab.cat/article/view/1787 |
| Suteddy dkk. (comparative) | 2023 | VGG/ResNet50/MobileNet/Xception/EffNet | IAM & CVL | Xception paling stabil (semua CNN) | https://doi.org/10.30630/joiv.7.1.1293 |
| Self-Supervised ViT for Writer Retrieval (Raven, Matei, Fink) | 2024 | ViT + VLAD (SSL) | CVL/Historical/HisIR19 | **CVL 98.6% mAP tanpa fine-tune**; Hist-WI 83.1%; HisIR19 95.0% | https://arxiv.org/abs/2409.00751 |
| Unsupervised Feature Learning (Christlein dkk.) | 2017 | SIFT+ResNet surrogate | CVL | ~99.4% Top-1 | https://arxiv.org/abs/1705.09369 |
| Towards Influence of Text Quantity (Peer dkk.) | 2025 | ResNet20+VLAD/NetVLAD | CVL/IAM | CVL 99.2% Top-1 / 97.4% mAP | https://arxiv.org/html/2506.07566 |
| Attention End-to-end word-level (Kumar & Sundaram) | 2024 | Attention CNN | IAM/CVL/CERUG | IAM 93.8%, CVL 92.3%; drop utk kata <4 char | https://arxiv.org/abs/2404.07602 |
| Contrastive Dissimilarity (Pignelli dkk.) | 2025 | EfficientNetV2 metric-learning | subset IAM+CVL | framing dissimilarity utk lintas-dataset | http://www.din.uem.br/yandre/Pignelli_IWSSIP_2025.pdf |
| Fiel & Sablatnig (cross-database) | 2015 | CNN | latih IAM → uji ICDAR2013/CVL | contoh klasik cross-database | https://link.springer.com/chapter/10.1007/978-3-319-23117-4_3 |

### Bukti "ViT data-hungry" (untuk argumen data terbatas)
- Zhu dkk. 2023 "Why ViT Trains Badly on Small Datasets": ViT < ResNet-18 pada data kecil — https://arxiv.org/abs/2302.03751
- ConvNeXt vs ViT small data: ConvNeXt jenuh ~12.8k sampel, ViT butuh jauh lebih banyak — https://arxiv.org/html/2505.08259v1
- Liu dkk. NeurIPS 2021 "Efficient Training of ViT with Small Datasets" — https://proceedings.neurips.cc/paper/2021/file/c81e155d85dae5430a8cee6f2242e82c-Paper.pdf

## CATATAN KEJUJURAN (verifikasi manual)
- mAP eksak Christlein 2017 di CVL: PDF tak terekstrak penuh, hanya Top-1 ~99.4% yang muncul → verifikasi langsung sebelum dikutip.
- Tidak ada satu paper pun dengan tabel bersih "train CVL → test IAM" drop closed-set (karena struktural: label beda). Ini menegaskan cross-dataset = retrieval, bukan klasifikasi.
- Beberapa angka >99% berasal dari setup patch-level/closed-set berbeda → tidak selalu apple-to-apple. Protokol seragam = nilai tambah skripsi.

## Referensi internal
- `04-topik-alternatif-resnet-klasifikasi.md` — kerangka topik.
- `03-catatan-diskusi.md` — konteks pivot.
