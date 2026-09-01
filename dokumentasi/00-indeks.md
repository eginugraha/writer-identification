# Dokumentasi Eksperimen — Writer-ID CVL

Perbandingan lima arsitektur untuk identifikasi penulis pada dataset CVL, dengan
ablasi data terbatas dan dua studi terpisah.

| # | Berkas | Isi | Sumber |
|---|---|---|---|
| 01 | [Dataset](01-dataset.md) | CVL, kohor 308 penulis, pembagian train/val/test | tulisan tangan |
| 02 | [Arsitektur](02-arsitektur.md) | Lima arsitektur yang dibandingkan | tulisan tangan |
| 03 | [Protokol pelatihan](03-protokol-pelatihan.md) | Resep scratch dan pretrained | tulisan tangan |
| 04 | [Skenario fine-tuning](04-skenario-fine-tuning.md) | Tujuh skenario Studi 2 | tulisan tangan |
| 05 | [Hasil scratch](05-hasil-eksperimen-scratch.md) | Tabel hasil mode scratch | **dibangkitkan** |
| 06 | [Hasil pretrained](06-hasil-eksperimen-pretrained.md) | Tabel hasil mode pretrained | **dibangkitkan** |
| 07 | [Hasil fine-tuning](07-hasil-eksperimen-finetune.md) | Studi 2, termasuk pembelahan FT1 lewat FT5 | tulisan tangan |
| 08 | [Hipotesis dan kesimpulan](08-hipotesis-dan-kesimpulan.md) | Sintesis ketiga eksperimen | tulisan tangan |

**Berkas 05 dan 06 dibangkitkan ulang oleh `scripts/make_report.py`** — jangan
disunting manual, suntingan Anda akan hilang saat laporan dibuat ulang:

```bash
python scripts/make_report.py --results results/results-scratch.csv    --date scratch
python scripts/make_report.py --results results/results-pretrained.csv --date pretrained
python scripts/make_figures.py
```

Berkas lain ditulis tangan dan aman disunting.

## Status per 2026-09-01

| Eksperimen | Run | Status |
|---|---|---|
| Studi 1 — scratch | 100/100 | selesai |
| Studi 1 — pretrained | 100/100 | selesai |
| Studi 2 — fine-tuning Swin-Tiny | 25/25 | selesai |
| Studi 2 — FT5 (eval-only, tanpa latih) | 5/5 | selesai |

Semuanya lengkap 5 seed, jadi seluruh angka di berkas 05–08 sudah final.
FT5 dijalankan belakangan (2026-09-01) tanpa pelatihan: ia mengevaluasi ulang
checkpoint FT0 dengan protokol uji FT1.
Pekerjaan lanjutan yang belum dijalankan didaftar di
[08-hipotesis-dan-kesimpulan.md](08-hipotesis-dan-kesimpulan.md#yang-belum-dikerjakan).

## Aturan pelaporan yang berlaku di semua berkas

**Run kolaps tidak boleh dirata-ratakan bersama run sehat.** Kriteria kolaps:
`top1_page < 0,05`, yaitu model memprediksi ~1 kelas dari 308. Pada mode
scratch, 36 dari 100 run kolaps — merata-ratakannya bersama run sehat
menghasilkan angka yang tidak mewakili keduanya. Laporkan sebagai
"kolaps 4/5; rata-rata run sehat 0,83".

**`top1_page` adalah metrik utama**, dihitung pada tingkat halaman: probabilitas
semua baris dalam satu halaman uji dirata-ratakan dulu, baru diambil argmax.
308 halaman uji = 308 keputusan. `map_line` dan `top1_retrieval` bekerja pada
tingkat baris (2.490 baris uji) sehingga angkanya jauh lebih rendah — keduanya
tidak sebanding satu sama lain.

**`best_val_acc` bukan hasil.** Ia dipakai untuk memilih checkpoint, dan pada L1
diukur pada baris yang berasal dari halaman yang sama dengan data latih. Lihat
[01-dataset.md](01-dataset.md#validasi-berbagi-halaman-dengan-latih).
