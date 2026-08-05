# Writer-ID Architecture Comparison (CVL)

Perbandingan 5 arsitektur deep learning (ResNet-50, ConvNeXt-Tiny, EfficientNetV2-S, ViT-Small, Swin-Tiny) untuk **klasifikasi penulis** pada dataset **CVL**, dengan **ablasi data terbatas** (halaman latih/penulis = 1,2,3,4,penuh) dan sumbu **pretrained vs from-scratch**.

Desain lengkap: `dokumentasi/06-desain-eksperimen-implementasi.md` · Rencana implementasi: `dokumentasi/07-rencana-implementasi.md`.

---

## Struktur

```
src/cvl/          # package pipeline (data_prep, dataset, models, metrics, train, evaluate, run_experiments, report)
scripts/          # entry-point: prep_manifests.py, run_all.py, make_report.py
configs/          # default.yaml (hyperparameter)
.env.example      # contoh subset grid / smoke test (salin ke .env)
tests/            # pytest (26 test, jalan di CPU)
cvl-database-1-1/ # DATASET (tidak di-commit — taruh manual)
results/          # manifests, checkpoints, results.csv, figures (dibuat otomatis)
```

---

## Menjalankan di RunPod (GPU)

### 1. Siapkan environment

Sewa pod GPU (mis. RTX 4090 / A100), lalu:

```bash
# clone repo
git clone <url-repo-anda> thesis && cd thesis

# environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Butuh Python ≥3.10. Dependensi utama: torch, timm, pandas, pyarrow, scikit-learn, Pillow, pyyaml, matplotlib.
>
> Jalankan semua perintah `python scripts/...` **dari root repo** (`thesis/`). Skrip otomatis menemukan package `src.cvl` sendiri, tapi path data (`configs/`, `results/`, `cvl-database-1-1/`) relatif terhadap direktori kerja.

### 2. Taruh dataset

Salin folder dataset ke root repo sehingga strukturnya:

```
thesis/cvl-database-1-1/
  trainset/lines/<writer>/<writer>-<page>-<line>.tif
  testset/lines/<writer>/<writer>-<page>-<line>.tif
  ...
```

Pipeline hanya membaca gambar di bawah folder bernama `lines/` (folder `words/`, `pages/`, `xml/` diabaikan).

Upload cepat via `runpodctl`, `scp`, atau `rsync` — atau taruh sebagai RunPod Network Volume agar persist antar-sesi.

### 3. Bangun manifest split

```bash
python scripts/prep_manifests.py
```

- Men-scan CVL, membuang writer **0431 & 0161**, dan hanya memakai penulis dengan **≥5 halaman**.
- Mencetak `n_kept_writers` dan daftar `dropped_writers` — **catat angka ini** (jumlah kelas final = n_kept_writers; masuk ke Bab metodologi).
- Menulis `results/manifests/seed{S}_L{tag}.parquet` untuk tiap seed × level.

### 4. Jalankan grid eksperimen

```bash
python scripts/run_all.py
```

- Grid penuh: **5 arsitektur × 4 level × 2 mode × 3 seed = 120 run**. (Level `full` di-drop — ukurannya ≈ L4, lihat `src/cvl/config.py`.)
- **Resume-able**: aman diulang kalau sesi RunPod putus — run yang sudah ada di `results/results.csv` dilewati. Cukup jalankan ulang perintah yang sama.
- Output: `results/results.csv` (1 baris/run: Top-1/Top-5/macro-F1 halaman, mAP & Top-1 retrieval, n_params, throughput, waktu latih), checkpoint terbaik di `results/checkpoints/<run_id>/best.pt`.
- Pantau progres: tiap run mencetak `done <run_id>: top1=... map=...`.

**Menghemat jam GPU / smoke test** (opsional) — atur lewat file `.env` di root repo, **tanpa** ngedit kode (lihat [Konfigurasi lewat `.env`](#konfigurasi-lewat-env)). Mis. `CVL_MODES=pretrained` melewati semua from-scratch (120 → 60 run).

Run **from-scratch + data penuh** yang paling lama; pretrained + N kecil sangat cepat.

### 5. Bangun laporan

```bash
python scripts/make_report.py
```

Menghasilkan `dokumentasi/08-hasil-eksperimen.md` (tabel Top-1, Top-5, macro-F1, mAP, dan efisiensi per arsitektur × level, untuk mode pretrained & scratch) + grafik `results/figures/acc_vs_n_*.png` (kurva akurasi-vs-N — visual "ViT data-hungry").

**Arsipkan laporan per tanggal** (biar hasil run sebelumnya tidak ketimpa):

```bash
python scripts/make_report.py --date                     # 08-hasil-eksperimen-2026-08-05.md
python scripts/make_report.py --date rerun-warmup        # 08-hasil-eksperimen-rerun-warmup.md
python scripts/make_report.py --out /path/laporan.md     # path bebas
```

`--date` menstempel **laporan dan figure-nya sekaligus** (`acc_vs_n_pretrained-2026-08-05.png`), jadi laporan lama tetap menunjuk grafik yang benar. Tanpa flag, output tetap nama kanonik seperti biasa — dokumen lain (`09-pembahasan-hasil.md`, `11-alur-kode-training.md`) merujuk nama itu.

> Di RunPod jam sistem UTC. Kalau mau tanggal WIB: `TZ=Asia/Jakarta python scripts/make_report.py --date`.

---

## Verifikasi lokal (tanpa GPU / dataset)

Seluruh logika pipeline diuji via smoke test dengan fixture kecil di CPU:

```bash
.venv/bin/pytest -q      # 26 test
```

---

## Konfigurasi

`configs/default.yaml` — batch size, jumlah epoch (pretrained/scratch), learning rate, weight decay, patience early-stopping, num_workers, AMP. Katalog penuh arsitektur, level ablasi, dan seed didefinisikan di `src/cvl/config.py` (`ALL_ARCHITECTURES`, `ALL_ABLATION_LEVELS`, `ALL_SEEDS`).

### Konfigurasi lewat `.env`

Untuk **mempersempit grid** (smoke test / hemat GPU) tanpa mengedit kode, buat file `.env` di root repo (dibaca otomatis oleh `src/cvl/config.py`). Salin dari contoh:

```bash
cp .env.example .env
```

| Variabel | Arti | Contoh | Penuh (default bila kosong) |
|---|---|---|---|
| `CVL_ARCHS` | subset arsitektur | `resnet50` | `resnet50,convnext_tiny,efficientnetv2_s,vit_small,swin_tiny` |
| `CVL_LEVELS` | level ablasi (halaman latih/penulis) | `1,4` | `1,2,3,4` |
| `CVL_SEEDS` | seed | `0` | `0,1,2` |
| `CVL_MODES` | mode | `pretrained` | `pretrained,scratch` |
| `CVL_MAX_WRITERS` | batasi jumlah penulis | `10` | semua penulis |
| `CVL_PRETRAINED_EPOCHS` / `CVL_SCRATCH_EPOCHS` | override epoch | `2` | dari `default.yaml` |
| `CVL_BATCH_SIZE` | override batch size | `32` | dari `default.yaml` |

Aturannya: **baris yang diisi = subset; dikosongkan/dihapus = pakai nilai penuh.** Hapus `.env` (atau kosongkan semua) → otomatis kembali ke grid penuh 120 run.

> `CVL_MAX_WRITERS` memengaruhi manifest, jadi ubah nilainya lalu **jalankan ulang `prep_manifests.py`** sebelum `run_all.py`.
>
> Environment variable menang atas isi `.env`, jadi bisa override sekali jalan: `CVL_MODES=pretrained python scripts/run_all.py`.
>
> `.env` **tidak** di-commit (masuk `.gitignore`); hanya `.env.example` yang ikut repo.

## Catatan

- Metrik retrieval (mAP) dihitung pada level baris di set test (leave-one-out, self dikecualikan); metrik klasifikasi diagregasi ke level halaman (rata-rata softmax per `writer|page`).
- Augmentasi sengaja **tanpa horizontal flip** (membalik tulisan merusak identitas penulis).
- GFLOPs belum dihitung (efisiensi dilaporkan via jumlah parameter + throughput); bisa ditambahkan bila diperlukan.
