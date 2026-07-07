# Riset Mendalam: Arsitektur Writer Identification Terbaru (CVL Database)

> Hasil deep-research terverifikasi (103 agen, 20 sumber primer, 25 klaim diverifikasi adversarial — 23 terkonfirmasi, 2 ditolak).
> Dibuat: 2026-07-07
>
> ⚠️ **CATATAN KONTEKS:** Riset ini menjawab tugas **Writer Identification** ("siapa penulisnya"). Judul thesis final adalah **HTR Personalization** ("apa isi tulisannya") menggunakan TrOCR + QLoRA + HITL — tugas yang **berbeda**. Dokumen ini tetap disimpan karena: (a) analisis dataset tetap berlaku, (b) writer identification bisa menjadi **sub-komponen opsional** (router pemilih adapter QLoRA per penulis). Untuk metode inti thesis, riset ini **tidak** menjadi acuan utama.

## Temuan Utama

> **CVL adalah benchmark "jenuh".** Metode klasik, CNN, maupun Transformer semuanya mencapai **~99% Top-1** dan **~98% mAP**. Perbedaan antar-arsitektur sangat kecil; **mAP lebih diskriminatif daripada Top-1**. Klaim "SOTA di CVL" harus dinyatakan hati-hati.

> **Hanya SATU arsitektur ViT/DINO 2023–2025 yang benar-benar melaporkan angka di CVL:** Raven, Matei & Fink (ICDAR 2024). Dua metode Transformer terbaru lain (SAGHOG, NetRVLAD+SGR) **tidak** diuji di CVL — fokus ke dataset historis.

## Arsitektur "Terbaru" (SOTA writer retrieval)

**Self-Supervised Vision Transformers for Writer Retrieval**
- **Penulis:** Tim Raven, Arthur Matei, Gernot A. Fink (TU Dortmund)
- **Venue:** ICDAR 2024 — `arXiv:2409.00751`
- **Arsitektur:** ViT-small/16 (input 224) dilatih self-supervised dengan **AttMask** (adaptasi *iBOT* yang menanamkan Masked Image Modeling ke framework **DINO**). Token *foreground* di-encode dengan **VLAD**, reranking **kRNN**.
- **Hasil di CVL (tanpa fine-tuning):**

| Konfigurasi | mAP | Top-1 |
|---|---|---|
| VLAD foreground tokens (tanpa reranking) | 97.1% | 99.4% |
| + kRNN reranking | **98.6%** | **99.4%** |

## Peta Perbandingan Lengkap

| Metode | Arsitektur | Penulis / Venue | CVL mAP | CVL Top-1 |
|---|---|---|---|---|
| **— Transformer / Self-Supervised —** |
| Self-Supervised ViT ⭐ | ViT-small/16 + AttMask(DINO) + VLAD + kRNN | Raven, Matei & Fink — ICDAR 2024 (`2409.00751`) | **98.6%** | **99.4%** |
| CCT7 + NetMVLAD | Compact Convolutional Transformer | Peer, Kleber & Sablatnig — ICPR 2022 | 96.5% | — |
| SAGHOG | ViT masked-autoencoder (rekonstruksi HOG) | Peer, Kleber & Sablatnig — ICDAR 2024 (`2404.17221`) | ❌ tidak diuji CVL | — |
| NetRVLAD + SGR | ResNet + NetRVLAD + graph reranking | Peer, Kleber & Sablatnig — ICDAR 2023 (`2305.05358`) | ❌ tidak diuji CVL | — |
| **— CNN —** |
| ResNet-20 + NetVLAD + krNN-QE | CNN + NetVLAD | Rasoulzadeh & BabaAli — IET Biometrics 2022 (`2012.06186`) | 98.6% | 99.2% |
| WriterINet | Dual-stream ResNet-50 + DenseNet-201 | IJDAR 2023 (DOI 10.1007/s10032-022-00418-3) | kompetitif (paywall) | — |
| CNN activation features | CNN + GMM supervector | Christlein et al. — GCPR 2015 | 97.8% | 99.4% |
| Codebook VLAD/E-SVM | CNN penultimate + VLAD + Exemplar-SVM | Christlein & Maier — DAS 2018 (`1712.07923`) | — (jangan sitasi angka presisi) | ~99.5% |
| **— Klasik (pra-deep-learning) —** |
| GMM Supervector + Exemplar-SVM | RootSIFT + GMM + E-SVM | Christlein et al. — Pattern Recognition 2017 | 98.4% | **99.5%** |

⭐ = SOTA writer retrieval. Christlein PR 2017 memegang **Top-1 tertinggi (99.5%)** — bukti metode klasik masih kompetitif di CVL.

## ⚠️ Peringatan Akurasi Sitasi (KRUSIAL)

1. **DOI `10.1007/978-3-319-24947-6_45`** = paper **Christlein et al. GCPR 2015** (CNN activation features), **BUKAN** Fiel & Sablatnig CAIP. Kesalahan sitasi umum.
2. **Jangan sitasi angka presisi** Top-1 99.6/mAP 98.0 untuk Christlein & Maier DAS 2018 — **ditolak** verifikasi (vote 0-3). Sitasi metodologinya saja.
3. **WriterINet** — angka CVL di balik paywall; jangan mengarang angka.
4. **Rasoulzadeh (2022), CCT7 (2022)** secara teknis sedikit di luar jendela 2023–2025 — sebutkan konteks tahunnya.

## Klaim yang DITOLAK verifikasi (jangan dipakai)

- "VLAD+E-SVM di CVL: Top-1 99.6 / mAP 98.0" (Christlein & Maier 2018) — vote 0-3.
- "Prior GMM-supervector CNN: Top-1 99.5 / mAP 97.2" — vote 1-2.

## Sumber Kunci (semua primer)

- `arXiv:2409.00751` — Raven, Matei & Fink, ICDAR 2024 (SOTA writer retrieval)
- `arXiv:2012.06186` / IET Biometrics 2022 — Rasoulzadeh & BabaAli (baseline CNN)
- Pattern Recognition 63:258-267, 2017 — Christlein et al. (baseline klasik)
- `arXiv:2404.17221` (SAGHOG) & `arXiv:2305.05358` (NetRVLAD+SGR) — ViT terbaru (tanpa angka CVL)
- `arXiv:1705.09369` — Christlein et al. ICDAR 2017 (unsupervised feature learning, fondasi metodologi)
- `arXiv:1712.07923` — Christlein & Maier DAS 2018

## Pertanyaan Terbuka

- Adakah **DINOv2** murni yang diuji di CVL (2024–2025)? Belum ditemukan — potensi celah kontribusi.
- Angka Top-1/mAP WriterINet yang persis (butuh full-text IJDAR).
- Apakah keunggulan self-supervised ViT lebih jelas di **cross-dataset / data terbatas** ketimbang di CVL yang jenuh?

## Catatan: Kompleksitas Self-Supervised ViT (kenapa "tinggi")

1. **SSL pretraining berat** — DINO teacher–student + momentum encoder (EMA), Masked Image Modeling (AttMask), multi-crop augmentation, temperature/centering untuk cegah collapse; butuh banyak data + GPU-hours; sensitif hyperparameter.
2. **ViT haus data & O(n²)** attention → memori/komputasi besar.
3. **Pipeline bertingkat**: ViT → foreground tokens → VLAD → PCA → kRNN reranking.

→ Jika writer ID hanya sub-komponen (memilih adapter), **overkill**. ResNet-20+NetVLAD (sedang) atau SIFT+VLAD (ringan, tanpa training) sudah ~99% di CVL.
