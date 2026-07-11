# Swin-Tiny

## Apa itu Swin-Tiny

**Swin Transformer** (*Shifted Window Transformer*) adalah Vision Transformer **hierarkis** yang menghitung *self-attention* hanya di dalam **jendela lokal** (bukan global seperti ViT), lalu "menggeser" jendela antar-lapisan agar informasi tetap mengalir antar-wilayah. Rancangan ini membuat kompleksitasnya **linear** terhadap ukuran gambar dan menghasilkan peta fitur bertingkat seperti CNN. **Swin-Tiny** adalah varian terkecil (~**28 juta parameter**).

Dalam tesis ini, Swin-Tiny mewakili **transformer hierarkis modern** — jembatan antara ViT murni dan CNN. Berkat *inductive bias* lokal-nya, ia diperkirakan **lebih tahan** pada data terbatas dibanding ViT murni.

## Kapan ditemukan

- **Makalah:** *"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"* — Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo (**Microsoft Research Asia**).
- **Tahun:** **2021** (arXiv Maret 2021), terbit di **ICCV 2021** dan memenangkan **Marr Prize (Best Paper)**.
- **Konteks:** hadir setahun setelah ViT, menjawab dua kelemahan ViT — biaya atensi global yang mahal untuk gambar besar, dan ketiadaan struktur multi-skala yang dibutuhkan tugas deteksi/segmentasi.

## Algoritmanya seperti apa

**1. Patch kecil + struktur hierarkis.** Gambar dipecah jadi patch `4×4` (lebih halus dari ViT). Melalui beberapa tahap, patch-patch tetangga digabung (*patch merging*) sehingga resolusi mengecil dan kanal membesar — persis pola piramida CNN (mis. 56×56 → 28×28 → 14×14 → 7×7).

**2. Window-based self-attention (W-MSA).** Atensi tidak dihitung antar-semua-token, melainkan **hanya di dalam jendela lokal** berukuran `7×7` patch. Ini memangkas biaya dari kuadratik menjadi **linear** terhadap jumlah patch — jauh lebih hemat daripada ViT.

**3. Shifted window (SW-MSA) — inti Swin.** Jika jendela selalu di posisi yang sama, informasi tak pernah menyeberang batas jendela. Maka pada lapisan berikutnya, susunan jendela **digeser** setengah jendela, sehingga token yang tadinya terpisah kini berada dalam jendela yang sama. Selang-seling **W-MSA → SW-MSA** inilah yang memberi jangkauan global secara bertahap namun tetap murah.

**4. Klasifikasi.** Setelah tahap terakhir, dilakukan *global average pooling* atas token lalu satu lapisan klasifikasi.

**Konsekuensi:** karena atensi lokal + hierarki memberi *inductive bias* mirip CNN (lokalitas, multi-skala), Swin umumnya **lebih mudah dilatih pada data lebih sedikit** dibanding ViT murni, sambil tetap menikmati keunggulan mekanisme atensi.

## Yang ada di kode

- **Pemanggilan (`src/cvl/config.py`):**
  ```python
  "swin_tiny": "swin_tiny_patch4_window7_224",
  ```
  Dipetakan ke model `timm` **`swin_tiny_patch4_window7_224`** — Swin-T, patch 4, jendela 7, input 224.

- **Pembangunan model (`src/cvl/models.py`):**
  ```python
  timm.create_model("swin_tiny_patch4_window7_224", pretrained=(mode=="pretrained"), num_classes=308)
  ```
  Bobot ImageNet untuk *pretrained*, acak untuk *scratch*; kepala klasifikasi 308 penulis.

- **Input:** **224×224** (ukuran jendela & tahap sudah dikalibrasi untuk resolusi ini). Baris tulisan *grayscale* → 3 kanal, normalisasi ImageNet (`src/cvl/dataset.py`).

- **Fitur retrieval:** `forward_features()` + `forward_head(pre_logits=True)` mengambil vektor terkumpul sesudah *pooling* untuk `map_line`.

- **Hipotesis yang diuji:** pada data terbatas, Swin-Tiny diharapkan berada **di antara** ViT-Small (paling data-hungry) dan CNN — bukti bahwa *inductive bias* lokal membantu ketahanan terhadap keterbatasan data.
