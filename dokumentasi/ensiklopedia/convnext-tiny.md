# ConvNeXt-Tiny

## Apa itu ConvNeXt-Tiny

**ConvNeXt** adalah CNN "murni" yang dimodernisasi dengan meniru keputusan desain **Vision Transformer** dan **Swin Transformer**, tanpa memakai *self-attention* sama sekali. Tujuannya membuktikan bahwa CNN yang dirancang ulang dengan cermat bisa **menyaingi transformer** pada tugas visi modern. **ConvNeXt-Tiny** adalah varian terkecil (~**28 juta parameter**), setara kelas dengan Swin-Tiny.

Dalam tesis ini, ConvNeXt-Tiny mewakili **CNN generasi terbaru** — untuk melihat apakah "CNN modern" mengungguli ResNet-50 klasik pada *writer identification*.

## Kapan ditemukan

- **Makalah:** *"A ConvNet for the 2020s"* — Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie (**Facebook AI Research / Meta AI**, bekerja sama dengan UC Berkeley).
- **Tahun:** **2022** (arXiv Januari 2022), terbit di **CVPR 2022**.
- **Konteks:** muncul sebagai "jawaban CNN" setelah gelombang transformer (ViT 2020, Swin 2021) mendominasi. Menunjukkan bahwa sebagian besar keunggulan transformer sebenarnya berasal dari *training recipe* dan pilihan desain, bukan dari *attention* itu sendiri.

## Algoritmanya seperti apa

ConvNeXt berangkat dari ResNet-50 lalu **memodernisasinya langkah demi langkah**, meniru transformer. Perubahan utamanya:

1. **Patchify stem** — lapisan awal memakai konvolusi `4×4` *stride* 4 (seperti "patch" ViT), bukan konvolusi + max-pool ala ResNet.
2. **Depthwise convolution 7×7** — kernel besar per-kanal, meniru cakupan spasial (*receptive field*) luas milik *self-attention*.
3. **Inverted bottleneck** — melebar di tengah lalu menyempit (pola dari MobileNet/Transformer MLP), kebalikan bottleneck ResNet.
4. **Lebih sedikit aktivasi & normalisasi** — hanya satu **GELU** dan satu **LayerNorm** per blok (ResNet memakai banyak ReLU + BatchNorm).
5. **Rasio tahap ala Swin** — distribusi jumlah blok antar-tahap dibuat mirip Swin-T (mis. 3:3:9:3).

Hasilnya tetap arsitektur **hierarkis** (resolusi mengecil, kanal membesar antar-tahap) seperti CNN pada umumnya, sehingga efisien dan mudah dilatih, namun akurasinya menyaingi transformer. Ditutup *global average pooling* + LayerNorm + lapisan klasifikasi.

## Yang ada di kode

- **Pemanggilan (`src/cvl/config.py`):**
  ```python
  "convnext_tiny": "convnext_tiny",
  ```
  Dipetakan ke model `timm` `"convnext_tiny"`.

- **Pembangunan model (`src/cvl/models.py`):**
  ```python
  timm.create_model("convnext_tiny", pretrained=(mode=="pretrained"), num_classes=308)
  ```
  Bobot ImageNet untuk mode *pretrained*, acak untuk *scratch*; kepala klasifikasi 308 penulis.

- **Input:** sama seperti arsitektur lain — baris tulisan *grayscale* → 3 kanal, *resize* **224×224**, normalisasi ImageNet (`src/cvl/dataset.py`).

- **Fitur retrieval:** `forward_features()` + `forward_head(pre_logits=True)` mengambil vektor sesudah *global pooling* ConvNeXt untuk metrik `map_line`.

- **Peran dalam grid:** dijalankan pada 5 level × 2 mode × 3 seed. Karena arsitektur modern biasanya sangat bergantung pada *pretraining*, perbandingan *pretrained vs scratch*-nya menarik di kondisi data terbatas.
