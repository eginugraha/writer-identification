# Analisis Dataset — CVL Database 1.1

## Sumber & lisensi

- **Nama:** CVL-Database (versi 1.1, rilis 12 September 2013).
- **Pembuat:** Computer Vision Lab, **Vienna University of Technology**.
- **Makalah rujukan:** F. Kleber, S. Fiel, M. Diem, R. Sablatnig, *"CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting"*, ICDAR 2013, hlm. 560–564.
- **Lisensi:** Creative Commons Attribution-NonCommercial 3.0 (riset non-komersial).

## Apa isinya

Dataset berisi **tulisan tangan** dari banyak penulis yang **menyalin teks yang sama**. Karena isi teksnya sama antar-penulis, model dipaksa membedakan penulis dari **gaya tulisan tangan**, bukan dari isi kata — inilah yang membuat tugas *writer identification* valid.

Tiap penulis menyalin kutipan dari: Flatland (E. A. Abbott), Macbeth (Shakespeare), Wikipedia "Mailüfterl", Origin of Species (Darwin), Faust (Goethe), The Picture of Dorian Gray (Wilde), dan The Fall of the House of Usher (Poe).

### Identitas penulis
Penulis **anonim**, hanya diberi **ID numerik** (`0001`–`0310`), bukan nama. ID diambil dari nama file/folder — lihat [pola penamaan](#struktur--pola-nama-file).

### Level anotasi
Dataset menyediakan beberapa level: `pages/` (halaman utuh), `lines/` (baris), `words/` (kata), `xml/` (koordinat). **Pipeline ini hanya memakai `lines/`** — folder lain diabaikan (`src/cvl/data_prep.py`, filter `"lines" in path`).

## Angka mentah (sebelum filter)

| Bagian | Baris | Penulis (folder) |
|---|---:|---:|
| `trainset/lines` | 1.624 | 27 |
| `testset/lines` | 11.849 | 283 |
| **Total** | **13.473** | **310** |

> **Penting — split asli CVL TIDAK dipakai apa adanya.** Pembagian `trainset` (27 penulis) vs `testset` (283 penulis) di dataset asli adalah protokol **writer retrieval** dengan penulis yang **berbeda** (disjoint). Sesuai desain tesis (*klasifikasi* penulis, *closed-set*), pipeline **menggabungkan semua baris dari kedua folder** (`scan_lines` memindai seluruh `cvl-database-1-1/`), lalu **membuat split train/test sendiri per-halaman** (lihat di bawah). Angka "27 menulis 7 teks, 283 menulis 5 teks" dari `readme.txt` menjelaskan kenapa jumlah halaman per penulis bervariasi 5–7.

## Filtering kohort

Diterapkan di `filter_cohort()` (`src/cvl/data_prep.py`) dengan parameter dari `config.py`:

- **Exclude** penulis `0431` & `0161` (`EXCLUDE_WRITERS`) — `0431` memang dihapus dari evaluasi writer-ID resmi (dua halaman kosong).
- **Minimal 5 halaman** per penulis (`MIN_PAGES = 5`).

**Hasil setelah filter:**

| Metrik | Nilai |
|---|---:|
| Penulis dipakai (**= jumlah kelas**) | **308** |
| Total baris dipakai | 13.440 |
| Penulis terbuang karena `<5` halaman | 0 (semua ≥5) |
| Halaman per penulis | 5–7 (rata-rata 5,2) |
| Baris per penulis | 31–75 (rata-rata 43,6) |

> **Untuk bab metodologi:** jumlah kelas final = **308 penulis**. (Angka ini juga tercetak sebagai `n_kept_writers` saat `prep_manifests.py` dijalankan.)

## Ukuran gambar

Gambar baris berbentuk **sangat memanjang** (satu baris teks):

| Dimensi | Rentang | Median |
|---|---|---:|
| Lebar | ~810 – 2.161 px | ~1.772 px |
| Tinggi | ~72 – 217 px | ~125 px |
| Ukuran file | — | ~100 KB (.tif) |

Di pipeline, semua gambar di-**grayscale** (tulisan tangan) lalu digandakan ke 3 kanal, di-**resize ke 224×224**, dan dinormalisasi dengan mean/std ImageNet (`src/cvl/dataset.py`). Karena selalu diseragamkan ke 224×224, ukuran sumber tidak memengaruhi beban komputasi model.

## Struktur & pola nama file

```
cvl-database-1-1/
  trainset/lines/<penulis>/<penulis>-<halaman>-<baris>.tif
  testset/lines/<penulis>/<penulis>-<halaman>-<baris>.tif
```

Contoh: `0050-8-4.tif` → penulis `0050`, halaman `8`, baris `4`. Regex di `data_prep.py`:
```python
_LINE_RE = re.compile(r"^(\d+)-(\d+)-(\d+)\.tif$")   # (writer, page, line)
```

## Split train / val / test (dibuat pipeline)

Dibuat di `build_manifest()` (`src/cvl/data_prep.py`), **per penulis**, berbasis **halaman** agar adil (baris dari halaman yang sama tak bocor antar-split):

1. **Test** — **1 halaman terakhir** tiap penulis disisihkan (`test_pages=1`). Tetap sama di semua level → **2.490 baris** konsisten.
2. **Train** — dari sisa halaman, diambil sebanyak **level ablasi** (1/2/3/4 halaman, atau semua untuk `full`), dipilih acak menurut `seed`.
3. **Val** — **10%** dari baris train (`val_frac=0.1`) untuk *early-stopping*.

Karena train dibatasi per **level ablasi**, ukurannya berbeda tiap level (test selalu tetap):

| Level | Train (baris) | Val (baris) | Test (baris) |
|---|---:|---:|---:|
| L1 (1 halaman) | 2.285 | 310 | 2.490 |
| L2 (2 halaman) | 4.665 | 545 | 2.490 |
| L3 (3 halaman) | 7.113 | 765 | 2.490 |
| L4 (4 halaman) | 9.455 | 1.056 | 2.490 |
| Lfull (semua) | 9.852 | 1.098 | 2.490 |

*(angka untuk seed 0; seed lain mirip karena pemilihan halaman acak)*

## Bagaimana dataset dipakai saat evaluasi

- **Klasifikasi** diagregasi ke **level halaman**: softmax semua baris dalam satu `penulis|halaman` dirata-ratakan (`aggregate_by_group`), lalu diukur Top-1/Top-5/macro-F1. Inilah kenapa `top1_page` bisa jauh lebih tinggi daripada akurasi per-baris.
- **Retrieval (mAP)** dihitung di **level baris** pada set test (kemiripan kosinus antar-fitur, *leave-one-out*, diri sendiri dikecualikan) — `retrieval_map` di `src/cvl/metrics.py`.

## Ringkasan angka kunci

| Item | Nilai |
|---|---:|
| Penulis total (mentah) | 310 |
| Penulis dipakai (kelas) | **308** |
| Total baris dipakai | 13.440 |
| Baris test (semua level) | 2.490 |
| Halaman/penulis | 5–7 (μ 5,2) |
| Baris/penulis | 31–75 (μ 43,6) |
| Resolusi input model | 224×224 |
