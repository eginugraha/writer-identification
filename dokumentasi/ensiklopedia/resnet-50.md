# ResNet-50

## Apa itu ResNet-50

**ResNet** (Residual Network) adalah arsitektur *Convolutional Neural Network* (CNN) yang memperkenalkan **residual connection** (koneksi lompat / *skip connection*). ResNet-50 adalah varian dengan **50 lapisan berbobot**, tersusun dari blok *bottleneck*. Ia menjadi *baseline* de-facto untuk hampir semua tugas visi komputer selama bertahun-tahun.

Dalam tesis ini ResNet-50 berperan sebagai **arsitektur pembanding klasik** — titik acuan untuk menilai apakah arsitektur yang lebih baru benar-benar lebih unggul pada tugas *writer identification* dengan data terbatas.

## Kapan ditemukan

- **Makalah:** *"Deep Residual Learning for Image Recognition"* — Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (**Microsoft Research**).
- **Tahun:** dipublikasikan **2015** (arXiv), memenangkan **ILSVRC 2015** & **COCO 2015**, terbit di **CVPR 2016**.
- **Dampak:** salah satu makalah paling banyak disitasi dalam sejarah *deep learning*; membuat pelatihan jaringan sangat dalam (ratusan lapisan) menjadi praktis.

## Algoritmanya seperti apa

**Masalah yang dipecahkan — degradasi.** Sebelum ResNet, menambah lapisan pada jaringan biasa justru **memperburuk** akurasi (bukan sekadar overfitting), karena gradien sulit merambat mundur ke lapisan awal (*vanishing gradient*).

**Ide inti — residual learning.** Alih-alih memaksa setiap blok mempelajari transformasi `H(x)` secara langsung, ResNet membuatnya mempelajari **selisih (residu)** `F(x) = H(x) − x`, lalu keluarannya ditambahkan kembali ke input:

```
keluaran = F(x) + x        (skip connection)
```

Kalau lapisan tak perlu mengubah apa-apa, ia cukup membuat `F(x) ≈ 0` — jauh lebih mudah dipelajari. Penjumlahan `+ x` juga membuka "jalan tol" bagi gradien untuk mengalir langsung ke lapisan awal.

**Blok bottleneck (dipakai ResNet-50).** Tiap blok berisi 3 konvolusi berurutan:
1. `1×1` — memampatkan jumlah kanal (mengurangi komputasi),
2. `3×3` — konvolusi utama,
3. `1×1` — mengembalikan jumlah kanal,

dengan *skip connection* mengelilingi ketiganya. ResNet-50 menumpuk 16 blok semacam ini dalam 4 tahap, ditutup *global average pooling* + satu lapisan *fully-connected* sebagai klasifikator. Total ~**25 juta parameter**.

## Yang ada di kode

- **Pemanggilan (`src/cvl/config.py`):**
  ```python
  ALL_ARCHITECTURES = { "resnet50": "resnet50", ... }
  ```
  Kunci internal `resnet50` dipetakan ke nama model `timm` `"resnet50"`.

- **Pembangunan model (`src/cvl/models.py`):**
  ```python
  timm.create_model("resnet50", pretrained=(mode=="pretrained"), num_classes=308)
  ```
  - `pretrained=True` → bobot **ImageNet** (mode *pretrained*); `False` → inisialisasi acak (mode *scratch*).
  - `num_classes=308` → kepala klasifikasi diganti agar keluarannya 308 penulis.

- **Input:** gambar baris tulisan di-*grayscale* lalu digandakan ke 3 kanal, di-*resize* ke **224×224**, dinormalisasi memakai mean/std ImageNet (`src/cvl/dataset.py`).

- **Fitur untuk retrieval (mAP):** `forward_features()` diikuti `forward_head(pre_logits=True)` mengambil vektor fitur sebelum lapisan klasifikasi (*global average pooling* ResNet), dipakai menghitung `map_line` (`src/cvl/metrics.py`).

- **Peran dalam grid:** dilatih pada 5 level ablasi × 2 mode × 3 seed, sama seperti arsitektur lain. Jumlah parameter aktual tercatat sebagai `n_params` di `results/results.csv`.
