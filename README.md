# Writer-ID Architecture Comparison (CVL)

Perbandingan 5 arsitektur deep learning (ResNet-50, ConvNeXt-Tiny, EfficientNetV2-S, ViT-Small, Swin-Tiny) untuk **klasifikasi penulis** pada dataset **CVL**, dengan **ablasi data terbatas** (halaman latih/penulis = 1,2,3,4,penuh) dan sumbu **pretrained vs from-scratch**.

Desain lengkap: `dokumentasi/06-desain-eksperimen-implementasi.md` · Rencana implementasi: `dokumentasi/07-rencana-implementasi.md`.

---

## Struktur

```
src/cvl/          # package pipeline (data_prep, dataset, models, metrics, train, evaluate, run_experiments, report)
scripts/          # entry-point: prep_manifests.py, run_all.py, make_report.py
configs/          # default.yaml (hyperparameter)
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

> Butuh Python ≥3.10. Dependensi utama: torch, timm, pandas, scikit-learn, Pillow, pyyaml, matplotlib.

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

- Grid penuh: **5 arsitektur × 5 level × 2 mode × 3 seed = 150 run**.
- **Resume-able**: aman diulang kalau sesi RunPod putus — run yang sudah ada di `results/results.csv` dilewati. Cukup jalankan ulang perintah yang sama.
- Output: `results/results.csv` (1 baris/run: Top-1/Top-5/macro-F1 halaman, mAP & Top-1 retrieval, n_params, throughput, waktu latih), checkpoint terbaik di `results/checkpoints/<run_id>/best.pt`.
- Pantau progres: tiap run mencetak `done <run_id>: top1=... map=...`.

**Menghemat jam GPU** (opsional, kalau biaya jadi kendala) — edit `scripts/run_all.py`:
- `modes=["pretrained"]` → lewati semua from-scratch (dari 150 → 75 run), atau
- kurangi `SEEDS` untuk mode scratch, atau jalankan from-scratch hanya pada sebagian level.

Run **from-scratch + data penuh** yang paling lama; pretrained + N kecil sangat cepat.

### 5. Bangun laporan

```bash
python scripts/make_report.py
```

Menghasilkan `dokumentasi/08-hasil-eksperimen.md` (tabel Top-1, Top-5, macro-F1, mAP, dan efisiensi per arsitektur × level, untuk mode pretrained & scratch) + grafik `results/figures/acc_vs_n_*.png` (kurva akurasi-vs-N — visual "ViT data-hungry").

---

## Verifikasi lokal (tanpa GPU / dataset)

Seluruh logika pipeline diuji via smoke test dengan fixture kecil di CPU:

```bash
.venv/bin/pytest -q      # 26 test
```

---

## Konfigurasi

`configs/default.yaml` — batch size, jumlah epoch (pretrained/scratch), learning rate, weight decay, patience early-stopping, num_workers, AMP. Arsitektur, level ablasi, dan seed didefinisikan di `src/cvl/config.py` (`ARCHITECTURES`, `ABLATION_LEVELS`, `SEEDS`).

## Catatan

- Metrik retrieval (mAP) dihitung pada level baris di set test (leave-one-out, self dikecualikan); metrik klasifikasi diagregasi ke level halaman (rata-rata softmax per `writer|page`).
- Augmentasi sengaja **tanpa horizontal flip** (membalik tulisan merusak identitas penulis).
- GFLOPs belum dihitung (efisiensi dilaporkan via jumlah parameter + throughput); bisa ditambahkan bila diperlukan.
