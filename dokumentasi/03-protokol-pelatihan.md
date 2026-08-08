# 03 — Protokol pelatihan: scratch dan pretrained

Kedua mode memakai **resep yang sama persis**. Satu-satunya perbedaan adalah
`pretrained=True/False` saat model dibuat, dan anggaran epoch. Ini disengaja:
kalau resepnya berbeda, selisih hasilnya tidak bisa dikaitkan ke pretraining.

## Hiperparameter bersama

Dari `configs/default.yaml`:

| Parameter | Nilai | Catatan |
|---|---|---|
| `image_size` | 224 | masukan persegi untuk semua arsitektur |
| `batch_size` | 64 | VRAM puncak 5,3 GB dari 25 GB — sangat lapang |
| `lr` | 3e-4 | AdamW |
| `weight_decay` | 0,05 | |
| `warmup_epochs` | 3 | LinearLR dari 1% lalu cosine |
| `early_stop_patience` | 8 | epoch tanpa perbaikan val |
| `amp` | true | mixed precision |
| `val_frac` | 0,1 | 10% baris halaman latih |

**Anggaran epoch berbeda**: `pretrained_epochs = 40`, `scratch_epochs = 150`.
Scratch diberi anggaran hampir empat kali lipat karena harus mempelajari fitur
dari nol. Ini membuat perbandingannya konservatif ke arah yang benar — kalau
scratch tetap kalah, itu bukan karena kekurangan anggaran.

### Satu pengecualian LR

`LR_OVERRIDES` di `src/cvl/run_experiments.py` menurunkan LR **ConvNeXt-Tiny
mode pretrained** ke `1e-4`, karena pada 3e-4 ia divergen. Semua kombinasi lain
memakai 3e-4. Override ini tercatat di kolom `lr` tiap baris CSV, jadi bisa
diperiksa ulang dari artefaknya.

## Jadwal laju belajar

`LinearLR(start_factor=0.01, total_iters=3)` diikuti `CosineAnnealingLR`,
disambung dengan `SequentialLR`.

**Warmup ini bukan hiasan.** Tanpa warmup, ConvNeXt dan Swin sering divergen di
epoch-epoch awal lalu kolaps ke satu kelas. Menambahkan warmup memperbaiki
kolaps pada mode **pretrained** sepenuhnya — tetapi tidak menyelesaikan masalah
pada mode scratch, yang tetap kolaps pada 36 dari 100 run. Lihat
[05-hasil-eksperimen-scratch.md](05-hasil-eksperimen-scratch.md).

## Loop pelatihan

`train_one_run` di `src/cvl/train.py`:

1. AdamW atas seluruh parameter (kecuali skenario dengan `freeze_strategy`).
2. Autocast + `GradScaler` bila AMP aktif dan device bukan CPU.
3. Setelah tiap epoch: evaluasi akurasi validasi **per baris**.
4. Checkpoint disimpan saat akurasi validasi membaik; state disalin ke CPU.
5. Berhenti bila 8 epoch berturut-turut tidak membaik.
6. `best.pt` yang disimpan adalah **state terbaik**, bukan state terakhir.

Yang dikembalikan hanya `best_val_acc`, `train_time_s`, `epochs_ran`. Metrik
hasil dihitung terpisah oleh `evaluate_checkpoint` atas `best.pt`, di split
**test** — tidak pernah menyentuh data validasi.

## Augmentasi

Baseline, hanya saat latih (`src/cvl/dataset.py`):

- `Grayscale(num_output_channels=3)` — CVL abu-abu, tapi bobot pretrained
  mengharapkan 3 kanal
- `Resize(224)` (sisi pendek)
- `RandomAffine(degrees=3, translate=(0.02,0.02), scale=(0.95,1.05))`
- `RandomResizedCrop(224, scale=(0.8,1.0), ratio=(0.9,1.1))`
- `ColorJitter(brightness=0.2, contrast=0.2)`
- `ToTensor` + normalisasi statistik ImageNet

Saat evaluasi: `Resize(224)` + `CenterCrop(224)` saja, tanpa augmentasi.

Augmentasinya sengaja ringan — tidak ada `RandomErasing`, tidak ada shear, rotasi
hanya 3°. Rotasi besar pada tulisan tangan berisiko menghapus kemiringan tulisan,
yang justru salah satu penanda identitas penulis paling kuat.

**Batasan penting**: kombinasi `Resize` sisi-pendek dan `RandomResizedCrop` pada
citra 12:1 membuat model hanya melihat strip tengah selebar 246 piksel dari
~3.284 piksel. Penjelasan lengkap di
[01-dataset.md](01-dataset.md#model-hanya-melihat-75-dari-tiap-baris).

## Evaluasi

Lima metrik, dihitung di `src/cvl/metrics.py`:

| Metrik | Tingkat | Cara hitung |
|---|---|---|
| `top1_page`, `top5_page` | halaman | probabilitas baris satu halaman dirata-ratakan, lalu argmax; 308 keputusan |
| `macro_f1_page` | halaman | F1 makro atas prediksi halaman |
| `map_line`, `top1_retrieval` | baris | kemiripan kosinus antar 2.490 fitur baris uji, diri sendiri dibuang |

`top1_page` adalah metrik utama. Metrik retrieval mengukur hal yang berbeda —
apakah fiturnya berguna tanpa kepala klasifikasi — dan angkanya jauh lebih rendah
karena satu baris membawa bukti jauh lebih sedikit daripada satu halaman.

## Kriteria kolaps

`top1_page < 0,05` berarti model memprediksi kira-kira satu kelas dari 308.
Ambang ini dipatok di muka. Run yang kolaps **tidak boleh** dirata-ratakan
bersama run sehat: keduanya berasal dari populasi berbeda, dan rata-rata
gabungannya tidak mewakili satu pun.

## Desain eksekusi

Grid dijalankan di **dua pod GPU dengan model kartu yang sama** — server 1
mengerjakan scratch, server 2 mengerjakan pretrained lalu Studi 2. Kesamaan
kartu bukan soal kecepatan: AMP aktif, dan perilaku TF32/bf16 berbeda antar
generasi GPU. Temuan utama mode scratch adalah klaim *stabilitas*, dan stabilitas
optimisasi paling peka terhadap presisi numerik. Kedua server tercatat memakai
`NVIDIA RTX PRO 4000 Blackwell`, `torch 2.13.0+cu130`, `timm 1.0.28` di setiap
baris CSV, sehingga klaim itu bisa diverifikasi dari artefaknya.
