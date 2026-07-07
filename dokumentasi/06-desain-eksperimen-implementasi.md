# Desain Eksperimen & Implementasi — Perbandingan Arsitektur untuk Klasifikasi Penulis (CVL)

> Dibuat: 2026-07-07 (sesi 3). Spec desain hasil brainstorming.
> Status: **spec disetujui arah, menunggu review final user → lanjut ke rencana implementasi.**
> Referensi: `04-topik-alternatif-resnet-klasifikasi.md` (kerangka), `05-riset-perbandingan-arsitektur.md` (celah literatur), `01-analisis-dataset-cvl.md` (dataset).

## 1. Ringkasan & Tujuan

Membandingkan **5 arsitektur deep learning** (CNN klasik, CNN modern, transformer) untuk **klasifikasi penulis (writer identification)** pada dataset **CVL**, dengan **ablasi data terbatas** sebagai kondisi menantang utama. Karena CVL jenuh (~99% saat data penuh), perbedaan antararsitektur baru terlihat saat jumlah sampel latih dikurangi. Hipotesis inti (dari literatur): **transformer (ViT) "data-hungry"** — unggul saat data/pretraining besar, tetapi runtuh lebih tajam daripada CNN/ConvNeXt saat data sedikit, terutama tanpa pretraining.

**Pertanyaan penelitian:**
1. Bagaimana peringkat akurasi & efisiensi kelima arsitektur pada klasifikasi penulis CVL?
2. Bagaimana masing-masing arsitektur menurun ketika jumlah halaman latih per penulis dikurangi (4→1)?
3. Seberapa besar peran pretraining ImageNet (pretrained vs from-scratch) terhadap ketahanan tiap arsitektur pada data terbatas?

## 2. Dataset & Preprocessing

- **Sumber:** CVL Handwriting Database v1.1 (`~/personal/thesis/cvl-database-1-1/`), potongan **baris** dari folder `lines/` (subfolder per penulis; nama file `{writer}-{page}-{line}.tif`).
- **Exclude:** writer **0431** (dihapus di protokol resmi) & **0161** (hanya 1 halaman). Sisa target kelas.
- **Kohor penulis:** hanya penulis dengan **≥5 halaman** dipakai (agar tersedia 1 halaman test + hingga 4 halaman latih yang seragam antar-level). Penulis dengan <5 halaman di-exclude dan **jumlahnya dilaporkan**. Jumlah kelas final = jumlah penulis lolos syarat (target ~308, dikonfirmasi saat `data_prep`).
- **Preprocessing baris:** grayscale→3 kanal; resize aspek-terjaga ke tinggi 224, lalu pad/crop lebar ke **224×224**; normalisasi statistik ImageNet. (Saat training, lebar boleh random-crop; saat test, center.)

## 3. Protokol Split (per-penulis, closed-set)

- **Klasifikasi closed-set**: model memprediksi 1 dari K kelas penulis.
- **Test tetap:** untuk tiap penulis, **1 halaman dikunci sebagai test** (dipilih deterministik — indeks halaman terakhir), **tidak pernah** dipakai latih di level mana pun. Semua baris halaman ini jadi data test.
- **Kolam latih:** halaman sisa penulis (4–6 tergantung penulis).
- Semua pemilihan halaman/baris memakai **seed tetap** agar reproducible.

## 4. Variabel Utama — Ablasi Data Terbatas

Jumlah **halaman latih per penulis** N ∈ **{1, 2, 3, 4, penuh}**:
- N=1..4 → subsampel N halaman dari kolam latih (deterministik per seed).
- **penuh** → semua halaman kolam latih (4–6; berbeda dari N=4 untuk 27 penulis "power" yang punya 6 halaman latih).
- Semua level memakai **test yang sama** → penurunan akurasi murni akibat pengurangan data latih.
- Dari N halaman terpilih, **semua baris** halaman itu menjadi sampel latih.

## 5. Arsitektur yang Dibandingkan

Dipilih agar **jumlah parameter setara (~21–28M)** untuk perbandingan adil; semua tersedia pretrained di `timm`.

| Kelompok | Model (`timm`) | ~Params |
|---|---|---|
| CNN klasik (baseline) | `resnet50` | 25M |
| CNN modern | `convnext_tiny` | 28M |
| CNN efisien | `tf_efficientnetv2_s` | 21M |
| Transformer polos | `vit_small_patch16_224` | 22M |
| Transformer hierarkis | `swin_tiny_patch4_window7_224` | 28M |

**Sumbu pretraining (dimasukkan sekarang):** tiap arsitektur dilatih dalam **2 mode**:
- **pretrained** — bobot ImageNet, lalu fine-tune (skenario realistis).
- **from-scratch** — inisialisasi acak (menguji langsung tesis "data-hungry"). Butuh lebih banyak epoch.

## 6. Metrik (Pendekatan Hybrid: klasifikasi + retrieval)

- **Klasifikasi** (agregasi prediksi baris → halaman/penulis via rata-rata softmax):
  - **Top-1**, **Top-5**, **macro-F1**.
- **Retrieval** (dari fitur pra-head, "gratis" tanpa training baru): tiap **baris test** jadi query, gallery = semua baris test lain, relevan = penulis sama → **mAP** & **Top-1 retrieval**. (Feasible karena 1 halaman test punya banyak baris/penulis.)
- **Efisiensi:** jumlah parameter, GFLOPs, waktu latih (GPU-hours), throughput inferensi (img/s).

## 7. Recipe Training (SAMA untuk semua arsitektur = kunci keadilan)

- Framework: **PyTorch + `timm`**; optimizer **AdamW**, scheduler **cosine**, **AMP mixed-precision**, **early stopping** pada validasi.
- Validasi: sisihkan sebagian baris kolam latih (mis. 10%) sebagai val untuk early stopping; **jangan** sentuh halaman test.
- **Augmentasi (tidak merusak gaya):** random resized-crop ringan, rotasi/affine kecil, jitter kontras/brightness. **TANPA horizontal flip** (membalik tulisan merusak identitas penulis).
- Epoch: pretrained ~30–50 (early stop); from-scratch lebih panjang (~100–200, early stop). Recipe konsisten dalam tiap mode, sama antararsitektur.
- **3 seed** per (arsitektur × level × mode) → laporkan **mean ± std** (variansi tinggi saat N=1).
- Checkpoint terbaik (val) disimpan ke volume RunPod.

## 8. Matriks Eksperimen & Biaya

- Grid penuh: **5 arsitektur × 5 level × 2 mode × 3 seed = 150 run**.
- Set latih kecil (N=1 ≈ ribuan baris, penuh ≈ belasan ribu) → run pretrained cepat (menit di RTX 4090). Run **from-scratch + data penuh** paling lama.
- Estimasi total ~**15–30 GPU-hours** di RunPod.
- **Knob penghemat (opsional bila jam GPU jadi kendala):** kurangi from-scratch jadi **1 seed**, atau jalankan from-scratch hanya pada level N∈{1,4,penuh}. Keputusan ditunda sampai lihat durasi run pertama.

## 9. Struktur Kode (config-driven, ramah RunPod)

```
src/
  data_prep.py       # parse CVL lines, exclude 0431/0161 & penulis <5 hal,
                     # bangun manifest split per-penulis untuk tiap level ablasi (CSV/JSON)
  dataset.py         # Dataset/DataLoader baris + augmentasi + preprocessing 224x224
  models.py          # factory timm (5 arsitektur, mode pretrained/scratch, head K kelas)
  train.py           # 1 run: (arch, level, mode, seed) -> checkpoint + log metrik
  evaluate.py        # agregasi baris->halaman, Top-1/Top-5/F1, mAP retrieval, efisiensi
  run_experiments.py # loop grid 150 run, tulis results.csv (resume-able, skip run selesai)
  report.py          # bangun tabel + grafik dari results.csv
configs/             # YAML: daftar arsitektur, level, mode, seed, hyperparam
results/
  manifests/         # split per-penulis per-level (reproducible)
  checkpoints/       # bobot terbaik (volume RunPod)
  results.csv        # 1 baris per run: semua metrik + metadata
  figures/           # grafik akurasi-vs-N, dll.
```
- `run_experiments.py` **resume-able**: lewati kombinasi yang sudah ada di `results.csv` → aman kalau sesi RunPod putus.
- Logging: CSV wajib; Weights & Biases opsional.

## 10. Deliverable Akhir (Laporan)

Markdown di `dokumentasi/` berisi:
- **Tabel hasil** arsitektur × level (Top-1 & mAP, mean±std) untuk mode pretrained & from-scratch.
- **Grafik kurva** akurasi-vs-N per arsitektur (memvisualkan "data-hungry").
- **Tabel efisiensi** (params, GFLOPs, waktu latih, throughput) vs akurasi.
- **Analisis temuan**: peringkat arsitektur, titik di mana transformer runtuh, peran pretraining.
- (Opsional) **Grad-CAM** — fitur mana yang membedakan penulis.

## 11. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| Semua arsitektur mentok ~99% pada data penuh (jenuh) | Justru diharapkan; kontribusi ada di level N rendah. Level penuh = pembanding atas/kontrol. |
| Variansi tinggi saat N=1 | 3 seed, laporkan mean±std, sampling halaman deterministik. |
| Distorsi gambar akibat resize baris lebar→224² | Aspek-terjaga + pad; evaluasi juga level halaman (agregasi) meredam noise per-baris. |
| Sesi RunPod putus | `run_experiments.py` resume-able, checkpoint ke volume. |
| From-scratch lambat/mahal | Knob: kurangi seed/level from-scratch (Bagian 8). |
| Ketidakadilan perbandingan | Params setara ~21–28M, recipe identik antararsitektur, preprocessing sama. |

## 12. Yang TIDAK dikerjakan (YAGNI)

- Cross-dataset CVL↔IAM (drop kecil, lemah — lihat `05`). IAM tidak dipakai.
- Personalisasi / TrOCR / QLoRA / HITL (judul lama, digugurkan).
- Metric-learning (ArcFace/triplet) sebagai training — retrieval cukup dari fitur klasifikasi (Hybrid C).
