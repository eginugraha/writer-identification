# Topik Alternatif: ResNet (atau Arsitektur Lebih Baru) untuk Klasifikasi Penulis di CVL

> Masukan/asesmen untuk arah topik yang sedang diminati. Dibuat: 2026-07-07
> Status: **sedang dipertimbangkan** — diskusi dilanjutkan besok.
> Ketertarikan awal: **sudut "Studi Perbandingan Arsitektur"**.

## Interpretasi Topik

CVL tidak punya label gender/usia/jenis huruf. Yang bisa "diklasifikasikan" pada dasarnya adalah **penulisnya** (310 penulis). Jadi:

> **"Klasifikasi tulisan tangan di CVL" = Writer Classification / Writer Identification** — input gambar tulisan → tebak **penulis mana** (kelas), menggunakan ResNet / arsitektur lain.

⚠️ Ini **berbeda** dari judul HTR/TrOCR/QLoRA. Ini praktis kembali ke Writer Identification, dibingkai sebagai klasifikasi.

## Penilaian Jujur

**Kelebihan**
- ✅ Jauh lebih sederhana & feasible daripada TrOCR + QLoRA + HITL.
- ✅ Well-supported (banyak kode/tutorial/dataset siap). Risiko teknis rendah.
- ✅ Bagus untuk fondasi deep learning.

**Kelemahan (WAJIB diwaspadai)**
- ⚠️ **CVL sudah "jenuh"** — semua metode (klasik, CNN, ViT) sudah ~99% Top-1 / ~98% mAP (lihat `02-riset-writer-identification.md`). "ResNet klasifikasi CVL saja" akan ~99% juga → **tanpa ruang kontribusi**.
- ⚠️ **Novelty rendah** jika berhenti di "ResNet + CVL". Sudah banyak dikerjakan.

## Cara Membuatnya Layak Jadi Penelitian (Sudut Kontribusi)

Jangan berhenti di "ResNet + CVL". Tambahkan minimal satu sudut:

| Sudut kontribusi | Contoh |
|---|---|
| **Studi perbandingan arsitektur** ⭐ (diminati) | ResNet **vs** arsitektur lebih baru (**ViT, ConvNeXt, Swin Transformer, EfficientNet**) — bandingkan akurasi, biaya, kecepatan. Inilah sudut "arsitektur lebih baru". |
| **Kondisi menantang** | Uji pada **data terbatas** (few samples/writer), **cross-dataset** (latih CVL, uji IAM), atau **robustness** (noise, augmentasi) — di sinilah beda arsitektur baru terlihat, bukan di CVL bersih. |
| **Efisiensi** | Bandingkan ukuran model / FLOPs / waktu inferensi vs akurasi — relevan untuk deployment. |
| **Analisis interpretability** | Visualisasi (Grad-CAM) fitur mana yang membedakan penulis. |

⭐ = sudut yang menarik minat. **Saran kuat:** gabungkan **Studi perbandingan arsitektur + minimal satu "kondisi menantang"** (mis. data terbatas atau cross-dataset). Alasannya: di CVL bersih semua arsitektur mentok ~99% sehingga perbandingannya jadi tidak menarik; perbedaan ResNet vs ViT/ConvNeXt baru **terlihat jelas** justru pada kondisi menantang. Ini yang mengubah "sekadar benchmarking" menjadi temuan penelitian.

## Perbandingan dengan Judul Saat Ini

| Aspek | ResNet klasifikasi CVL (writer ID) | TrOCR + QLoRA + HITL (judul sekarang) |
|---|---|---|
| Tugas | Writer Identification | HTR Personalization |
| Kesulitan | 🟢 Rendah–Sedang | 🔴 Tinggi |
| Novelty di CVL | ⚠️ Rendah (benchmark jenuh) | ✅ Tinggi (kombinasi baru) |
| Risiko teknis | Rendah | Tinggi |
| "Kekinian" | Sedang | Sangat tinggi |

## Kandidat Arsitektur untuk Studi Perbandingan

| Kelompok | Arsitektur | Peran dalam studi |
|---|---|---|
| CNN klasik | **ResNet-50** (atau ResNet-18/20) | Baseline utama |
| CNN modern | **ConvNeXt**, EfficientNet | "CNN generasi baru" |
| Transformer | **ViT**, **Swin Transformer** | "Arsitektur lebih baru" |
| (opsional) Hybrid | CvT / CoAtNet | Jembatan CNN–Transformer |

Metrik: **Top-1 accuracy, mAP** (+ FLOPs, jumlah parameter, waktu inferensi jika ambil sudut efisiensi).
Catatan: pakai protokol writer-independent CVL; **exclude writer 0431 & 0161** (lihat `01-analisis-dataset-cvl.md`).

## Yang Perlu Diputuskan Besok (Pertanyaan Terbuka)

1. **Jenjang studi?** (S1 / S2 / S3) — menentukan kedalaman kontribusi yang dituntut.
   - S1: "ResNet vs arsitektur baru + 1 kondisi menantang" umumnya memadai.
   - S2/S3: butuh sudut lebih kuat (cross-dataset, kontribusi metode, dsb).
2. **Alasan pivot** dari judul TrOCR/QLoRA? (kesulitan teknis / arahan dosen / minat / masih eksplorasi)
3. **Apakah benar-benar ganti judul**, atau writer ID ini jadi bagian/pembanding dari topik HTR?
4. **Dataset:** CVL saja, atau CVL + IAM (untuk cross-dataset)?

## Rencana Langkah Berikutnya (untuk sesi besok)

1. Konfirmasi 4 pertanyaan di atas.
2. Jika lanjut arah ini → **riset mendalam terverifikasi**: "writer classification/identification dengan ResNet vs ViT/ConvNeXt/Swin, benchmark CVL & cross-dataset, 2022–2025, bersitasi" → memetakan siapa yang sudah melakukan perbandingan ini & menemukan celah.
3. Rumuskan **rumusan masalah + kontribusi** yang tajam (hindari jebakan "benchmark jenuh").
4. Rancang **desain eksperimen** (arsitektur, split, metrik, kondisi menantang).

## Referensi Internal
- `01-analisis-dataset-cvl.md` — detail dataset & anomali (0431, 0161).
- `02-riset-writer-identification.md` — SOTA writer ID + bukti "benchmark jenuh".
- `03-catatan-diskusi.md` — konteks judul & pergeseran arah.
