# Metodologi Eksperimen

## 1. Dataset dan Kohort

Penelitian menggunakan **CVL Database (v1.1)** pada tingkat **citra baris**
tulisan tangan (`lines`). Setiap berkas baris dinamai dengan pola
`writer-page-line.tif`, sehingga identitas penulis, halaman, dan nomor baris
diperoleh langsung dari nama berkas.

Kohort disaring dengan dua aturan: (i) dua penulis dikeluarkan secara eksplisit
(ID `0431` dan `0161`), dan (ii) hanya penulis dengan **minimal 5 halaman** yang
dipertahankan. Setelah penyaringan, kohort final terdiri atas **308 penulis
(kelas)** dengan total **13.440 citra baris**; jumlah halaman per penulis
berkisar 5–7 (rerata 5,2).

## 2. Protokol Split dan Ablasi Ukuran Data

Untuk setiap penulis, **satu halaman terakhir** dijadikan data uji (total
**2.490 baris uji**, tetap identik di seluruh kondisi agar perbandingan adil).
Sisa halaman menjadi *pool* pelatihan. Ukuran data latih divariasikan sebagai
**level ablasi** berdasarkan jumlah halaman latih per penulis:

| Level | Halaman latih/penulis | Baris latih | Baris validasi | Baris uji |
|-------|----------------------:|------------:|---------------:|----------:|
| L1    | 1                     | 2.285       | 310            | 2.490     |
| L2    | 2                     | 4.665       | 545            | 2.490     |
| L3    | 3                     | 7.113       | 765            | 2.490     |
| L4    | 4                     | 9.455       | 1.056          | 2.490     |
| full  | semua (5–6)           | 9.852       | 1.098          | 2.490     |

Sebanyak **10%** baris latih disisihkan sebagai validasi (minimal 1 baris).
Halaman latih dipilih secara acak per *seed*. Karena ukuran `full` (9.852)
praktis identik dengan L4 (9.455, selisih ~4%) dan menghasilkan akurasi yang
setara, **ablasi dilaporkan pada L1–L4** dan `full` digunakan hanya untuk
analisis pelatihan dari *scratch*.

## 3. Praproses dan Augmentasi

Seluruh citra dikonversi ke *grayscale* lalu direplikasi menjadi tiga kanal,
diubah ukuran ke sisi 224 piksel, dan dinormalisasi dengan statistik ImageNet.
Pada data latih diterapkan augmentasi: `RandomAffine` (rotasi $\pm$3°,
translasi 2%, skala 0,95–1,05), `RandomResizedCrop` 224 (skala 0,8–1,0), dan
`ColorJitter` (brightness/contrast 0,2). Pada data validasi/uji hanya dilakukan
`Resize` + `CenterCrop` 224 tanpa augmentasi.

## 4. Arsitektur yang Dibandingkan

Lima arsitektur dievaluasi melalui pustaka **`timm`**, dengan lapisan
klasifikasi disesuaikan menjadi 308 kelas:

| Kunci | Model `timm` | Params (juta) |
|-------|--------------|--------------:|
| ResNet-50        | `resnet50`                     | 24,1 |
| EfficientNetV2-S | `tf_efficientnetv2_s`          | 20,6 |
| ViT-S/16         | `vit_small_patch16_224`        | 21,8 |
| Swin-T           | `swin_tiny_patch4_window7_224` | 27,8 |
| ConvNeXt-T       | `convnext_tiny`                | 28,1 |

Setiap arsitektur dilatih dalam dua mode: **pretrained** (inisialisasi bobot
ImageNet) dan **scratch** (inisialisasi acak).

## 5. Konfigurasi Pelatihan

- **Optimizer:** AdamW, *weight decay* 0,05.
- **Learning rate:** 3e-4 (basis). Khusus ConvNeXt-T mode *pretrained*, LR
  diturunkan ke **1e-4** karena divergen pada LR basis.
- **Penjadwal LR:** *warmup* linear **3 epoch** (faktor awal 0,01) diikuti
  *cosine annealing*. *Warmup* ini krusial: tanpanya ConvNeXt-T dan Swin-T
  divergen di epoch awal dan kolaps.
- **Fungsi rugi:** *cross-entropy*.
- **Batch size:** 64; *mixed precision* (AMP) diaktifkan pada GPU.
- **Epoch maksimum:** 40 (*pretrained*), 150 (*scratch*).
- **Early stopping:** *patience* 8 epoch berdasarkan akurasi validasi
  (level baris); *checkpoint* dengan akurasi validasi terbaik disimpan.

## 6. Protokol Evaluasi

Model dievaluasi pada data uji tingkat baris, lalu prediksi **diagregasi ke
tingkat halaman** dengan merata-ratakan probabilitas *softmax* seluruh baris
dalam satu halaman (identitas halaman = `writer|page`). Metrik yang dilaporkan:

- **Klasifikasi (tingkat halaman):** Top-1, Top-5, dan Macro-F1.
- **Retrieval (tingkat baris):** *mean Average Precision* (mAP) dan Top-1
  *retrieval*, dihitung dari kemiripan kosinus fitur (representasi
  `forward_features` ternormalisasi), dengan menghapus kecocokan diri sendiri.
- **Efisiensi:** jumlah parameter dan *throughput* (citra/detik) saat inferensi.

## 7. Desain Grid dan Reproduksibilitas

Eksperimen penuh mencakup **5 arsitektur × 5 level × 2 mode × 3 seed = 150
*run***. *Seed* (0, 1, 2) mengontrol pembentukan split maupun inisialisasi
(NumPy dan PyTorch), sehingga hasil dapat direproduksi. Setiap kombinasi
dijalankan melalui skrip terparameterisasi (variabel lingkungan `CVL_*`), dan
hasil metrik dicatat per *run* ke `results/results.csv`.
