# Catatan Diskusi — Arah Thesis

> Ringkasan percakapan terakhir. Dibuat: 2026-07-07

## Judul Thesis (final)

**"Personalisasi Pengenalan Tulisan Tangan menggunakan TrOCR dengan Adaptasi QLoRA dan Pembelajaran Inkremental Human-in-the-Loop"**

## Klarifikasi Krusial: HTR, bukan Writer Identification

- Judul = **HTR (Handwritten Text Recognition)** → *"apa isi tulisannya?"* (output: teks; metrik: **CER/WER**).
- **BUKAN Writer Identification** → *"siapa penulisnya?"* (output: ID penulis; metrik: mAP/Top-1).
- Riset deep-search awal (arsitektur ViT/DINO writer ID) menjawab tugas yang **berbeda** → hanya berguna sebagian (analisis dataset + writer ID sebagai sub-komponen opsional). Lihat `02-riset-writer-identification.md`.

## Penilaian Komponen Judul (semua current & koheren, 2025)

| Komponen | Status | Catatan |
|---|---|---|
| **TrOCR** | ✅ Relevan | Model HTR Transformer standar (Microsoft, AAAI 2023). Baseline kuat. |
| **QLoRA** | ✅ Tepat | Personalisasi hemat parameter: adapter mungil per-penulis, base 4-bit di-freeze. |
| **HITL Incremental Learning** | ✅ Relevan | Belajar dari koreksi bertahap. LoRA/QLoRA membantu kurangi *catastrophic forgetting* (base beku). |

## Dataset CVL — masih relevan?

**Ya**, cocok karena terorganisir per-penulis + ada transkripsi + gambar level baris.
**Keterbatasan:** vocabulary sempit (~312 token), aliran HITL harus disimulasikan.
**Rekomendasi:** pertimbangkan **IAM (utama) + CVL (personalisasi per-writer)**. Detail: `01-analisis-dataset-cvl.md`.

## Posisi Writer Identification dalam thesis

Bukan tugas utama. Bisa muncul sebagai **sub-komponen opsional** (router):
```
Tulisan masuk → [SIAPA penulisnya?] → pilih QLoRA adapter → TrOCR baca teks
```
- Jika identitas penulis **sudah diketahui** (user login) → tidak perlu writer ID sama sekali.
- Jika **tidak diketahui** → butuh modul writer ID ringan (BUKAN Self-Supervised ViT yang overkill).

**Opsi writer ID jika diperlukan (semua ~99% di CVL):**
| Opsi | Kompleksitas | Kapan dipakai |
|---|---|---|
| Self-Supervised ViT | 🔴 Tinggi | Hanya jika writer ID = fokus utama |
| ResNet-20 + NetVLAD | 🟡 Sedang | Baseline deep-learning solid & ringan |
| SIFT/RootSIFT + VLAD | 🟢 Rendah | Router cepat/ringan, tanpa training |

## Tentang "SOTA"

- **SOTA = State-of-the-Art** (hasil terbaik saat ini pada suatu tugas + benchmark).
- SOTA **tergantung tugas**:
  - Writer Identification → Self-Supervised ViT (Raven 2024), 98.6% mAP CVL.
  - **HTR (tugas Anda)** → diukur CER/WER (biasanya di IAM). **Belum diriset terverifikasi.**
- Dua SOTA relevan untuk thesis: (a) SOTA HTR umum sebagai baseline; (b) **SOTA personalisasi/adaptasi HTR** (PEFT/LoRA, continual/active learning) — inilah yang menentukan **novelty** Anda.

## Langkah Berikutnya (belum dijalankan)

1. **Riset mendalam baru** dengan target benar: TrOCR & SOTA HTR di IAM (CER/WER) + PEFT/LoRA/QLoRA untuk HTR + personalisasi/writer-adaptation + continual/active/HITL learning, 2023–2025, bersitasi. → untuk baseline + memetakan novelty.
2. Keputusan dataset final (CVL saja vs CVL + IAM).
3. Rancang pipeline implementasi (preprocessing → TrOCR + QLoRA → HITL loop → evaluasi CER/WER).

## Pertanyaan terbuka untuk dikonfirmasi

- Apakah writer identification masuk scope (router otomatis) atau tidak (identitas diketahui)?
- Fokus kontribusi utama: efisiensi QLoRA / anti-forgetting / active learning HITL / peningkatan CER-WER?
- Dataset: CVL saja atau CVL + IAM?
