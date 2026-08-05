# Writer-ID Architecture Comparison (CVL)

Perbandingan 5 arsitektur (ResNet-50, ConvNeXt-Tiny, EfficientNetV2-S, ViT-Small, Swin-Tiny) untuk **klasifikasi penulis** pada dataset **CVL**, dengan ablasi data terbatas (halaman latih per penulis = 1–4) dan sumbu pretrained vs from-scratch.

Desain lengkap: [`docs/superpowers/specs/2026-08-05-grid-5seed-dan-finetune-convnext-design.md`](docs/superpowers/specs/2026-08-05-grid-5seed-dan-finetune-convnext-design.md)

## Rencana eksperimen

| | Studi 1 — Grid utama | Studi 2 — Fine-tuning ConvNeXt |
|---|---|---|
| **Isi** | 5 arsitektur × 4 level × 2 mode × 5 seed | 6 skenario × 5 seed, ConvNeXt-Tiny di L1 |
| **Jumlah run** | 200 | 25 baru (baseline diambil dari Studi 1) |
| **Perkiraan** | ±41 jam GPU | ±2,4 jam GPU |
| **Dijalankan** | 2 server paralel | server 2, setelah Studi 1 selesai |

Enam skenario Studi 2 — masing-masing mengubah **satu** mekanisme saja: `FT0` baseline · `FT1` geometri input · `FT2` drop_path + label smoothing · `FT3` freeze parsial + LLRD · `FT4` head ArcFace · `AUG` augmentasi kuat.

Hasil dipisah jadi tiga berkas: `results-scratch.csv`, `results-pretrained.csv`, `results-finetune.csv`.

> **Status implementasi.** Studi 1 butuh dua perubahan kecil (`ALL_SEEDS` jadi 5 seed, kolom `gpu_name` di CSV) yang **belum diterapkan**. Studi 2 butuh `src/cvl/scenarios.py` dan `scripts/run_scenarios.py` yang **belum dibuat**. Lihat §5 spec.

## Struktur

```
src/cvl/          # pipeline: data_prep, dataset, models, metrics, train, evaluate,
                  #           run_experiments, finetune, report
scripts/          # entry-point: prep_manifests.py, run_all.py, make_report.py
configs/          # default.yaml (hyperparameter)
tests/            # pytest, 26 test, jalan di CPU tanpa dataset
cvl-database-1-1/ # DATASET — tidak di-commit, taruh manual
results/          # manifests, checkpoints, CSV, figures (dibuat otomatis)
```

---

# Eksekusi di cloud

Butuh **dua pod GPU**. Server 1 mengerjakan mode scratch, server 2 mengerjakan pretrained lalu Studi 2.

> **Kedua pod wajib memakai model GPU yang sama.** Bukan soal kecepatan — AMP aktif dan perilaku TF32/bf16 berbeda antar generasi GPU. Temuan utama mode scratch adalah klaim *stabilitas* ("kolaps x/5 seed"), dan stabilitas optimisasi paling peka terhadap presisi numerik. Kartu berbeda membuat temuan itu tidak bisa dipertahankan, dan menyamakannya tidak menambah biaya.

## Langkah 1 — Siapkan server 1

```bash
git clone <url-repo> thesis && cd thesis
git checkout research

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Butuh Python ≥3.10. Semua perintah `python scripts/...` dijalankan **dari root repo**.

Kunci versinya, lalu pakai file ini untuk memasang server 2:

```bash
pip freeze > requirements.lock.txt
```

Ini penting: `requirements.txt` tidak mengunci versi apa pun, dan timm sesekali mengubah tag bobot pretrained bawaan antar rilis. Dua pod yang dibuat selang beberapa hari bisa dapat versi berbeda.

## Langkah 2 — Taruh dataset

```
thesis/cvl-database-1-1/
  trainset/lines/<writer>/<writer>-<page>-<line>.tif
  testset/lines/<writer>/<writer>-<page>-<line>.tif
```

Hanya gambar di bawah folder `lines/` yang dibaca (`words/`, `pages/`, `xml/` diabaikan). Upload via `runpodctl`, `scp`, atau `rsync` — atau pakai Network Volume agar persist antar-sesi.

## Langkah 3 — Bangun manifest

```bash
python scripts/prep_manifests.py
```

- Membuang writer **0431 & 0161**, memakai penulis dengan **≥5 halaman** → 308 kelas.
- Menulis `results/manifests/seed{S}_L{tag}.parquet` untuk tiap seed × level.
- Catat `n_kept_writers` yang tercetak — angka ini masuk bab metodologi.

**Tiap server membangun manifest-nya sendiri.** Kolom `path` menyimpan path absolut mesin, jadi manifest tidak bisa disalin antar-server. Ini aman: split-nya terbukti identik lintas mesin untuk seed yang sama, karena `groupby("writer")` dan `sorted(pages)` menormalkan urutan sebelum RNG dipakai.

## Langkah 4 — Siapkan server 2

Ulangi Langkah 1–3 di pod kedua, tapi pasang dependensinya dari file terkunci:

```bash
pip install -r requirements.lock.txt
```

## Langkah 5 — Jalankan Studi 1

Server 1:

```bash
CVL_MODES=scratch python scripts/run_all.py --date scratch
```

Server 2:

```bash
CVL_MODES=pretrained python scripts/run_all.py --date pretrained
```

Masing-masing menulis `results/results-<tag>.csv` dan `results/checkpoints-<tag>/`. Tiap run mencetak progres per epoch dan baris `done <run_id>: top1=... map=...` saat selesai.

**Kalau pod putus:** ulangi perintah yang **sama persis**. Resume bekerja per file CSV — run yang sudah tercatat di CSV itu dilewati. Mengganti tag `--date` berarti mulai dari nol.

## Langkah 6 — Jalankan Studi 2

Hanya di server 2, **setelah** Studi 1 selesai:

```bash
python scripts/run_scenarios.py --date finetune
```

Baseline `FT0` disalin dari `results-pretrained.csv`, bukan dijalankan ulang — sah karena berada di mesin, versi library, dan manifest yang sama. Kalau Studi 2 terpaksa pindah mesin, `FT0` harus dijalankan ulang di sana.

## Langkah 7 — Laporan

```bash
python scripts/make_report.py --results results/results-pretrained.csv --date pretrained
```

Menghasilkan tabel per arsitektur × level plus grafik akurasi-vs-N di `results/figures/`.

---

## Aturan pelaporan hasil

Tiga aturan yang mempengaruhi cara angka dibaca — rinciannya di §6 spec.

**Run kolaps tidak masuk rata-rata.** Kriteria kolaps: `top1_page < 0,05` (model memprediksi ~1 kelas). Laporkan `kolaps 2/5; rata-rata run sehat 0,82 ± 0,03`, bukan rata-rata gabungan yang mencampur keduanya. Pada grid sebelumnya 31 dari 75 run scratch kolaps meski warmup aktif — ini temuan, bukan bug.

**Throughput hanya dari satu server.** `throughput_img_s` diukur saat inferensi sehingga hanya bergantung pada arsitektur dan hardware, bukan pada mode. Ambil angkanya dari server 2 saja; kolom yang sama dari server 1 diabaikan. Untuk biaya latih pakai `epochs_ran` yang bebas hardware, dan sebutkan GPU-jam per server terpisah beserta nama kartunya.

**Selisih di bawah 3,7 poin tidak terdeteksi.** Std antar-seed pada L1 adalah 0,0295; dengan uji-t berpasangan 5 seed, itu ambang deteksi minimumnya. Selisih 1–2 poin dilaporkan sebagai "tidak terdeteksi", bukan "sedikit lebih baik". Jangan pakai Wilcoxon signed-rank — pada n=5 nilai p terkecil yang mungkin adalah 0,0625, jadi hasilnya mustahil signifikan.

---

## Verifikasi lokal (tanpa GPU/dataset)

```bash
.venv/bin/pytest -q      # 26 test
```

## Konfigurasi

`configs/default.yaml` — batch size, epoch, learning rate, weight decay, patience, num_workers, AMP. Katalog arsitektur/level/seed ada di `src/cvl/config.py`.

Untuk mempersempit grid tanpa mengedit kode, buat `.env` di root repo (`cp .env.example .env`):

| Variabel | Arti | Contoh |
|---|---|---|
| `CVL_ARCHS` | subset arsitektur | `resnet50` |
| `CVL_LEVELS` | level ablasi | `1,4` |
| `CVL_SEEDS` | seed | `0` |
| `CVL_MODES` | mode | `pretrained` |
| `CVL_MAX_WRITERS` | batasi jumlah penulis | `10` |
| `CVL_PRETRAINED_EPOCHS` / `CVL_SCRATCH_EPOCHS` | override epoch | `2` |
| `CVL_BATCH_SIZE` | override batch size | `32` |

Baris yang diisi = subset; dikosongkan atau dihapus = nilai penuh. Environment variable menang atas isi `.env`, jadi bisa override sekali jalan seperti pada Langkah 5. `.env` tidak di-commit.

`CVL_MAX_WRITERS` mempengaruhi manifest — ubah nilainya lalu jalankan ulang `prep_manifests.py`.

## Catatan

- Metrik retrieval (mAP) dihitung di level baris pada set test (leave-one-out, self dikecualikan); metrik klasifikasi diagregasi ke level halaman (rata-rata softmax per `writer|page`).
- Augmentasi sengaja **tanpa horizontal flip** — membalik tulisan merusak identitas penulis.
- **Citra baris berasio ~12:1, dan pipeline saat ini hanya melihat 7,5% bagian tengahnya.** `Resize(224)` menyetel sisi pendek, lalu `RandomResizedCrop` dengan `ratio=(0.9,1.1)` tidak pernah bisa dipenuhi sehingga jatuh ke center-crop deterministik — artinya crop itu juga bukan augmentasi. Skenario `FT1` menguji perbaikannya secara terkendali; grid utama sengaja dibiarkan apa adanya. Lihat §2 spec.
- GFLOPs belum dihitung (efisiensi dilaporkan lewat jumlah parameter + throughput).
