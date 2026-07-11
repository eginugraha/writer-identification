# EfficientNetV2-S

## Apa itu EfficientNetV2-S

**EfficientNetV2** adalah keluarga CNN yang dirancang untuk **efisiensi parameter** sekaligus **kecepatan pelatihan** yang tinggi. Varian **-S** (*Small*) adalah yang terkecil (~**21 juta parameter**) namun tetap akurat. Ciri khasnya: mencapai akurasi tinggi dengan parameter jauh lebih sedikit dibanding model lain sekelasnya.

Dalam tesis ini, EfficientNetV2-S mewakili **CNN hemat & cepat** — relevan untuk pertanyaan efisiensi (akurasi per-parameter, throughput) yang juga dilaporkan di hasil.

## Kapan ditemukan

- **Makalah:** *"EfficientNetV2: Smaller Models and Faster Training"* — Mingxing Tan, Quoc V. Le (**Google Research / Google Brain**).
- **Tahun:** **2021** (arXiv April 2021), terbit di **ICML 2021**.
- **Pendahulu:** EfficientNet (V1) tahun 2019 dari penulis yang sama, yang memperkenalkan **compound scaling**. V2 memperbaiki kelemahan V1 pada kecepatan latih.

## Algoritmanya seperti apa

**1. Compound scaling (warisan V1).** Alih-alih memperbesar jaringan hanya di satu dimensi, EfficientNet menskalakan **kedalaman (depth)**, **lebar (width)**, dan **resolusi input** secara **serentak dan seimbang** menurut satu koefisien. Ini menghasilkan keluarga model (B0…B7 / S, M, L) yang efisien di berbagai anggaran komputasi.

**2. MBConv + Fused-MBConv.** Blok dasarnya adalah **MBConv** (*Mobile Inverted Bottleneck Convolution*) dari MobileNet: melebar dengan `1×1`, konvolusi *depthwise*, lalu menyempit lagi, dilengkapi **Squeeze-and-Excitation** (mekanisme atensi antar-kanal yang ringan). Inovasi V2: pada tahap-tahap awal, MBConv diganti **Fused-MBConv** (menggabungkan konvolusi ekspansi `1×1` dan *depthwise* `3×3` menjadi satu konvolusi `3×3` biasa) — lebih cepat di perangkat modern.

**3. Training-aware NAS + progressive learning.** Arsitektur V2 dicari lewat *Neural Architecture Search* yang **memperhitungkan kecepatan latih**, bukan hanya akurasi. V2 juga memakai **progressive learning** (ukuran gambar & regularisasi dinaikkan bertahap selama pelatihan) untuk mempercepat konvergensi — meski di repo ini pelatihan memakai ukuran tetap 224.

Gabungan ini membuat EfficientNetV2-S berlatih lebih cepat dan lebih hemat parameter dibanding banyak model dengan akurasi setara.

## Yang ada di kode

- **Pemanggilan (`src/cvl/config.py`):**
  ```python
  "efficientnetv2_s": "tf_efficientnetv2_s",
  ```
  Kunci internal `efficientnetv2_s` dipetakan ke model `timm` **`tf_efficientnetv2_s`** (implementasi hasil port dari bobot TensorFlow resmi).

- **Pembangunan model (`src/cvl/models.py`):**
  ```python
  timm.create_model("tf_efficientnetv2_s", pretrained=(mode=="pretrained"), num_classes=308)
  ```
  Bobot ImageNet untuk *pretrained*, acak untuk *scratch*; kepala klasifikasi 308 penulis.

- **Input:** baris tulisan *grayscale* → 3 kanal, *resize* **224×224**, normalisasi ImageNet.

- **Catatan efisiensi:** karena jumlah parameternya paling kecil di antara kelima arsitektur, EfficientNetV2-S sering jadi pembanding menarik pada kolom `n_params` dan `throughput_img_s` di `results/results.csv` (akurasi tinggi dengan model kecil). Namun blok *depthwise* + SE bisa membuat throughput per-gambar tidak selalu paling cepat, meski parameternya sedikit.

- **Fitur retrieval:** `forward_features()` + `forward_head(pre_logits=True)` → vektor untuk `map_line`.
