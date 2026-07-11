# ViT-Small (ViT-S/16)

## Apa itu ViT-Small

**ViT** (*Vision Transformer*) adalah arsitektur yang menerapkan **Transformer** — model yang semula dari pemrosesan bahasa (NLP) — langsung pada gambar, **tanpa konvolusi sama sekali**. Gambar dipotong menjadi "patch", tiap patch diperlakukan seperti sebuah "kata". **ViT-Small dengan patch 16×16** (ViT-S/16) adalah varian kecil (~**22 juta parameter**) yang dilatih pada input 224×224.

Dalam tesis ini, ViT-Small mewakili **transformer visi murni** — dan menjadi kandidat utama untuk menunjukkan fenomena **"data-hungry"**: transformer biasanya butuh banyak data / *pretraining* kuat, sehingga diduga paling terpukul pada kondisi data terbatas.

## Kapan ditemukan

- **Makalah:** *"An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale"* — Alexey Dosovitskiy dkk. (**Google Research / Brain Team**).
- **Tahun:** **2020** (arXiv Oktober 2020), terbit di **ICLR 2021**.
- **Varian "Small":** ukuran ViT-S dipopulerkan oleh karya lanjutan seperti **DeiT** (Touvron dkk., 2021) dan *"How to train your ViT"* (Steiner dkk., 2021), yang membuat ViT bisa dilatih efektif tanpa dataset raksasa.

## Algoritmanya seperti apa

**1. Patchify + embedding.** Gambar `224×224` dibagi menjadi patch `16×16` yang tidak tumpang tindih → `14×14 = 196` patch. Tiap patch diratakan dan diproyeksikan linear menjadi vektor (*token*).

**2. Class token + positional embedding.** Sebuah token khusus `[CLS]` ditambahkan di depan untuk merangkum keseluruhan gambar. Karena Transformer tidak punya notion urutan/posisi, ditambahkan **positional embedding** agar model tahu letak tiap patch.

**3. Transformer encoder.** Rangkaian token melewati beberapa blok Transformer identik, tiap blok berisi:
   - **Multi-Head Self-Attention (MHSA)** — tiap token "melihat" **semua** token lain sekaligus (atensi **global**), menimbang mana yang relevan;
   - **MLP** (dua lapisan + GELU);
   - **LayerNorm** dan *residual connection* di sekeliling keduanya.

**4. Klasifikasi.** Keluaran token `[CLS]` (atau rata-rata semua token) dilewatkan ke satu lapisan klasifikasi.

**Konsekuensi penting:** karena atensi bersifat global sejak lapisan pertama dan tidak punya *inductive bias* lokal seperti CNN (lokalitas, translasi), ViT **butuh lebih banyak data** untuk belajar pola dasar — inilah alasan ia rawan buruk saat dilatih *from-scratch* pada data sedikit.

## Yang ada di kode

- **Pemanggilan (`src/cvl/config.py`):**
  ```python
  "vit_small": "vit_small_patch16_224",
  ```
  Dipetakan ke model `timm` **`vit_small_patch16_224`** — ViT-S, patch 16, input 224.

- **Pembangunan model (`src/cvl/models.py`):**
  ```python
  timm.create_model("vit_small_patch16_224", pretrained=(mode=="pretrained"), num_classes=308)
  ```
  Bobot ImageNet untuk *pretrained*, acak untuk *scratch*; kepala klasifikasi 308 penulis.

- **Input:** ukuran **224×224 wajib** (positional embedding terikat pada 196 patch). Baris tulisan *grayscale* → 3 kanal, normalisasi ImageNet (`src/cvl/dataset.py`).

- **Fitur retrieval:** `forward_features()` mengembalikan token; `forward_head(pre_logits=True)` mengambil representasi terkumpul (token `[CLS]` / *pooled*) sebagai vektor untuk `map_line` (`src/cvl/models.py` menangani perbedaan bentuk keluaran antar-arsitektur secara seragam).

- **Hipotesis yang diuji:** pada mode *scratch* + level data kecil (L1–L2), ViT-Small diperkirakan turun paling tajam — kurva `acc_vs_n` untuknya menjadi bukti visual "ViT butuh banyak data".
