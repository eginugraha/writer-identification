# 01 — Dataset

## Sumber

**CVL Database 1-1**, korpus tulisan tangan dari TU Wien, diunduh dari Zenodo
(rekaman 1492267). Yang dipakai hanya folder `lines/`: potongan **baris**
tulisan tangan yang sudah tersegmentasi, bukan halaman utuh atau kata tunggal.

Nama berkas mengikuti pola `{penulis}-{halaman}-{baris}.tif`, diurai oleh
`parse_line_filename` di `src/cvl/data_prep.py`. Pemindaian menemukan **13.473**
berkas `.tif` di bawah folder `lines/`.

Citra baris berbentuk sangat memanjang — rasio lebar:tinggi sekitar 12:1. Angka
ini menentukan hampir semua keputusan pra-pemrosesan di berkas
[03-protokol-pelatihan.md](03-protokol-pelatihan.md), dan menjadi sumber
keterbatasan utama eksperimen ini.

## Kohor: dari seluruh penulis ke 308

Tidak semua penulis dipakai. `filter_cohort` menerapkan dua saringan:

1. **Dua penulis dibuang manual** — `0431` dan `0161` (`EXCLUDE_WRITERS` di
   `src/cvl/config.py`).
2. **Penulis dengan kurang dari 5 halaman dibuang** (`MIN_PAGES = 5`). Alasannya
   struktural: level ablasi tertinggi memakai 4 halaman latih per penulis dan 1
   halaman uji, jadi penulis dengan 4 halaman tidak bisa mengisi L4 tanpa
   mengorbankan halaman ujinya.

Hasilnya **308 penulis**, dan angka itu menjadi jumlah kelas di seluruh
eksperimen. Label dipetakan dari `sorted(writer)` sehingga stabil lintas mesin.

## Pembagian train / val / test

Dibangun oleh `build_manifest` di `src/cvl/data_prep.py`, per penulis:

1. **Halaman uji**: satu halaman terakhir setelah pengurutan (`test_pages=1`).
   Dipilih deterministik, bukan acak — semua seed memakai halaman uji yang sama.
2. **Halaman latih**: `n_train_pages` halaman diambil acak dari sisanya. Angka
   inilah yang menjadi level ablasi.
3. **Validasi**: 10% baris dari halaman latih (`val_frac=0.1`, minimal 1 baris).

Halaman yang tidak terpilih ditandai `unused` dan dibuang dari manifest.

### Empat level ablasi

| Level | Halaman latih / penulis | Baris latih | Baris val | Baris uji |
|---|---|---|---|---|
| L1 | 1 | 2.285 | 310 | 2.490 |
| L2 | 2 | 4.665 | 545 | 2.490 |
| L3 | 3 | 7.113 | 765 | 2.490 |
| L4 | 4 | 9.455 | 1.056 | 2.490 |

Semua level memakai **308 penulis** dan **halaman uji yang sama** (308 halaman,
2.490 baris). Yang berubah hanya banyaknya data latih. Ini yang membuat L1–L4
sebanding: satu-satunya variabel adalah ukuran data.

Level `full` (semua halaman non-uji) pernah ada tapi dibuang dari grid: ukurannya
9.852 baris, hanya ~4% di atas L4, dan akurasinya setara — redundan.

### Manifest disimpan per seed

`results/manifests/seed{S}_L{N}.parquet`, 20 berkas (5 seed × 4 level). Kolom
`path` menyimpan path absolut mesin, jadi manifest **tidak bisa disalin antar
server**. Ini aman: pembagiannya terbukti identik lintas mesin untuk seed yang
sama, karena `groupby("writer")` dan `sorted(pages)` menormalkan urutan sebelum
RNG dipakai.

## Dua hal yang harus disebut saat melapor

### Validasi berbagi halaman dengan latih

Pada L1, tiap penulis hanya punya **satu** halaman latih. Baris validasi diambil
dari halaman itu juga — jadi val dan train berbagi pena, kertas, sesi menulis,
dan kondisi pemindaian yang sama. Baris validasinya memang tidak ikut dilatih,
tapi ia bukan sampel independen.

Konsekuensinya: `best_val_acc` pada L1 (~0,54–0,62 untuk Swin) **tidak sebanding**
dengan `top1_page` uji (~0,81). Keduanya mengukur hal berbeda. `best_val_acc`
hanya sah dipakai untuk memilih checkpoint, bukan dilaporkan sebagai hasil.

### Model hanya melihat 7,5% dari tiap baris

Dengan `geometry="center"` (dipakai seluruh Studi 1 dan enam dari tujuh skenario
Studi 2), `T.Resize(224)` menyetel **sisi pendek** sehingga baris 12:1 melebar
jadi sekitar 3.284 × 224 piksel. Lalu:

- **saat latih**, `RandomResizedCrop(224, ratio=(0.9,1.1))` tidak bisa memenuhi
  batasan rasionya pada citra selebar itu dan jatuh ke fallback torchvision
  `w = round(h × ratio_maks)` = **246 piksel**. "Crop acak" itu praktis selalu
  mengambil strip tengah yang sama;
- **saat uji**, `CenterCrop(224)` mengambil 224 piksel di tengah.

Artinya sekitar **92,5% dari setiap baris tidak pernah dilihat model**. Seluruh
angka Studi 1 dicapai hanya dari potongan tengah baris.

Ini bukan kelalaian yang tidak disadari — skenario `FT1` di
[04-skenario-fine-tuning.md](04-skenario-fine-tuning.md) dirancang khusus untuk
mengukur dampaknya, dan hasilnya ada di
[07-hasil-eksperimen-finetune.md](07-hasil-eksperimen-finetune.md): +14,2 poin
`top1_page`.

Yang mengejutkan datang dari `FT5`: **dua pertiga kerugian itu bisa ditebus
tanpa melatih ulang apa pun**, cukup dengan berhenti menilai satu penulis dari
satu potongan tengah saat inferensi (+9,42 dari +14,22 poin). `FT6` melengkapi
gambarannya dari sisi sebaliknya: melatih dengan jendela acak tanpa mengubah
protokol uji memberi +12,01 poin. Keduanya tidak menjumlah — mereka menutup
kekurangan yang sama dari ujung berbeda. Jadi 92,5% yang terbuang adalah soal
apa yang dilihat model saat *latih* **dan** saat *uji*, dan menutup salah
satunya saja sudah memulihkan sebagian besarnya.
