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

## Struktur

```
src/cvl/          # pipeline: data_prep, dataset, models, arcface, metrics, train,
                  #           evaluate, run_experiments, run_scenarios, scenarios,
                  #           finetune, env_info, report
scripts/          # entry-point: preflight.py, prep_manifests.py, run_all.py,
                  #              run_scenarios.py, make_report.py, make_figures.py,
                  #              progress.py
configs/          # default.yaml (hyperparameter)
tests/            # pytest, 97 test, jalan di CPU tanpa dataset
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
python -m pip install -r requirements.txt
```

Butuh Python ≥3.10. Semua perintah `python scripts/...` dijalankan **dari root repo**.

> **Pakai `python -m pip`, bukan `pip` polos.** Pada image RunPod, `pip` sering menunjuk ke Python lain meski prompt sudah menampilkan `(.venv)`. Gejalanya menyesatkan: pemasangan melaporkan sukses, tapi `import` tetap gagal dengan `ModuleNotFoundError`. Bentuk `python -m pip` menjamin paket masuk ke interpreter yang sama dengan yang menjalankan skrip.
>
> Kalau instalasi gagal karena kehabisan ruang — `torch` sekitar 2,5 GB plus cache seukuran itu, dan container disk RunPod jauh lebih kecil daripada volume `/workspace` — arahkan cache-nya ke volume:
> ```bash
> TMPDIR=/workspace/tmp python -m pip install --cache-dir /workspace/.pipcache -r requirements.txt
> ```

Verifikasi paketnya benar-benar masuk sebelum lanjut:

```bash
python -m pip list | grep -Ei "torch|timm|pandas|pyarrow"
```

Kunci versinya, lalu pakai file ini untuk memasang server 2:

```bash
python -m pip freeze > requirements.lock.txt
```

Ini penting: `requirements.txt` tidak mengunci versi apa pun, dan timm sesekali mengubah tag bobot pretrained bawaan antar rilis. Dua pod yang dibuat selang beberapa hari bisa dapat versi berbeda.

## Langkah 1b — Cek pra-terbang

Jalankan sekali di tiap pod, **sebelum** menyalin dataset:

```bash
python scripts/preflight.py
```

Ia memeriksa tiga hal yang kalau salah baru ketahuan berjam-jam kemudian:

1. **Build PyTorch punya kernel untuk kartu ini.** Kartu generasi Blackwell (RTX PRO 4000/4500/6000) butuh CUDA 12.8+ dan kernel `sm_120`. Tanpa itu prosesnya mati pada peluncuran kernel pertama — bukan pada `torch.cuda.is_available()`, yang tetap melaporkan `True`. Karena itu cek ini menjalankan forward **dan** backward sungguhan, bukan sekadar menanyakan ketersediaan CUDA. Kalau baris kernel ditandai `!!`, pasang ulang dari indeks CUDA yang sesuai:
   ```bash
   python -m pip install --force-reinstall torch torchvision \
     --index-url https://download.pytorch.org/whl/cu128
   ```
   Lalu jalankan `python -m pip freeze > requirements.lock.txt` **ulang**, supaya pod kedua mendapat build yang sama.
2. **VRAM puncak sebenarnya** pada `batch_size` dari `configs/default.yaml`, diukur dengan EfficientNetV2-S — model paling rakus aktivasi di katalog. Batch 64 sudah terbukti aman di 20 GB, tapi angka pastinya lebih baik dilihat daripada dipercaya.
3. **vCPU, dan apakah `/dev/shm` cukup untuk antrean prefetch.** Docker membatasi shared memory ke 64 MB secara bawaan; gejala kalau sempit adalah `DataLoader worker killed by signal: Bus error` beberapa menit setelah mulai — mudah disalahartikan sebagai masalah GPU.

Setiap baris bertanda `!!` harus dibereskan sebelum melanjutkan. Keluaran nyata dari pod yang dipakai (RTX PRO 4000 Blackwell):

```
== versi ==
  torch 2.13.0+cu130 | CUDA 13.0 | timm 1.0.28

== GPU ==
  NVIDIA RTX PRO 4000 Blackwell | compute capability 12.0 | VRAM 25 GB
  kernel tersedia di build ini: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']

== forward+backward tf_efficientnetv2_s, batch 64 ==
  OK — VRAM puncak 5.3 GB dari 25 GB (21%)

== CPU, RAM, shared memory ==
  vCPU efektif: 10 (dari kuota cgroup v2; core host 48) -> saran num_workers = 8 (sekarang 8)
  antrean prefetch: 2 loader x 8 worker x 2 = 1.2 GB di shared memory
  /dev/shm tersedia: 15.0 GB
```

Dua hal yang layak diperhatikan dari keluaran itu:

- **VRAM puncak cuma 21%.** Batch 64 sangat lapang di kartu 24 GB; tidak ada risiko OOM di seluruh grid. Jangan tergoda menaikkan `batch_size` — itu parameter eksperimen, mengubahnya mengubah hasil dan memutus kesebandingan dengan baseline.
- **`vCPU efektif` (10) jauh di bawah `core host` (48).** Angka yang mengikat adalah kuota kontainer. Menyetel `num_workers` dari 48 akan membuat puluhan proses berebut 10 CPU — lebih lambat daripada bawaan, plus antrean prefetch membengkak beberapa GB.

## Langkah 2 — Taruh dataset

```
thesis/cvl-database-1-1/
  trainset/lines/<writer>/<writer>-<page>-<line>.tif
  testset/lines/<writer>/<writer>-<page>-<line>.tif
```

Hanya gambar di bawah folder `lines/` yang dibaca (`words/`, `pages/`, `xml/` diabaikan). Upload via `runpodctl`, `scp`, atau `rsync` — atau pakai Network Volume agar persist antar-sesi.

**Unggah `lines/` saja.** Dataset penuh 5,1 GB, tapi folder `lines/` cuma **1,3 GB** (13.473 berkas) — sisanya 3,8 GB berisi `words/`, `pages/`, dan `xml/` yang tidak pernah dibaca pipeline. Menyalin semuanya membuang tiga perempat waktu unggah Anda:

```bash
rsync -a --include='*/' --include='*/lines/***' --exclude='*' \
  cvl-database-1-1/ <pod>:/workspace/writer-identification/cvl-database-1-1/
```

**Kebutuhan disk pod** (volume `/workspace`):

| Isi | Server 1 (scratch) | Server 2 (pretrained + Studi 2) |
|---|---|---|
| Dataset (`lines/` saja) | 1,3 GB | 1,3 GB |
| venv dengan torch CUDA | ~7 GB | ~7 GB |
| Checkpoint | 9,8 GB (100 run × ~98 MB) | 12,3 GB (125 run) |
| Cache pip | ~2,5 GB | ~2,5 GB |
| **Total** | **~21 GB** | **~23 GB** |

Volume 50 GB lapang. Checkpoint adalah penyumbang terbesar dan tumbuh sepanjang run — satu `best.pt` berkisar 82 MB (EfficientNetV2-S) sampai 112 MB (ConvNeXt-T). Kalau suatu saat mepet, `python -m pip cache purge` membebaskan ~2,5 GB tanpa efek samping.

## Langkah 3 — Bangun manifest

```bash
python scripts/prep_manifests.py
```

- Membuang writer **0431 & 0161**, memakai penulis dengan **≥5 halaman** → 308 kelas.
- Menulis `results/manifests/seed{S}_L{tag}.parquet` untuk tiap seed × level.
- Catat `n_kept_writers` yang tercetak — angka ini masuk bab metodologi.

**Tiap server membangun manifest-nya sendiri.** Kolom `path` menyimpan path absolut mesin, jadi manifest tidak bisa disalin antar-server. Ini aman: split-nya terbukti identik lintas mesin untuk seed yang sama, karena `groupby("writer")` dan `sorted(pages)` menormalkan urutan sebelum RNG dipakai.

> Kalau di Langkah 5 muncul `FileNotFoundError` pada path yang bukan milik mesin ini, artinya `results/manifests/` berisi manifest dari mesin lain (ikut terbawa `rsync`). Jalankan ulang langkah ini — ia menimpa berkas yang ada.

## Langkah 4 — Siapkan server 2 & setel `num_workers`

Ulangi Langkah 1–3 di pod kedua, tapi pasang dependensinya dari file terkunci:

```bash
python -m pip install -r requirements.lock.txt
```

**`num_workers` adalah pengungkit kecepatan terbesar yang Anda punya, bukan pilihan GPU.** Beban ini terbatas oleh CPU: ia berjalan pada ~4,8 TFLOPS efektif, jauh di bawah kemampuan kartu mana pun yang layak dipakai. Yang menghabiskan waktu adalah decode TIF, resize ke ~3284×224, dan `RandomAffine` — semuanya di CPU.

`configs/default.yaml` sudah disetel untuk pod dengan **kuota 10 vCPU**:

```yaml
num_workers: 8          # kuota cgroup - 2
prefetch_factor: 2      # 2 loader × 8 worker × 2 = 1,2 GB shared memory
```

Kalau pod Anda berbeda, pakai angka `vCPU efektif` dari Langkah 1b dikurangi 2 — **bukan** `nproc`, yang melaporkan core host dan bisa berkali-kali lipat lebih besar daripada kuota yang Anda bayar. Perkiraan dampaknya pada Studi 1 (latih + evaluasi, per server):

| vCPU | worker | Per server |
|---|---|---|
| 10 (pod ini) | 8 | ~17 j |
| 20 | 18 | ~11 j |
| 28 | 26 | ~7 j |

`persistent_workers` menyala otomatis begitu `num_workers > 0`. Tanpanya PyTorch menyalakan dan mematikan seluruh proses pekerja **dua kali setiap epoch** — biaya tetap ~3,8 detik yang di level data terkecil memakan 38% waktu tiap epoch, dan justru membesar seiring jumlah worker.

`prefetch_factor` menentukan pemakaian shared memory. Naikkan ke 4 hanya kalau Langkah 1b menunjukkan `/dev/shm` lapang.

> Angka di tabel itu proyeksi dari model biaya yang dicocokkan ke dua titik data grid lama, bukan hasil pengukuran — rentangnya lebar karena penskalaan worker tidak bisa saya pastikan. Arah dan urutan prioritasnya bisa dipercaya; angka persisnya jangan. Verifikasi dengan membandingkan `train_time_s` run L1 pertama terhadap patokan lama 341 detik (pretrained) / 241 detik (scratch).

## Langkah 5 — Jalankan Studi 1

Run ini memakan sekitar 20 jam per server, jadi jalankan di background dengan log — jangan ditunggu di foreground, karena sesi SSH yang putus akan membunuh prosesnya.

Server 1:

```bash
CVL_MODES=scratch nohup python -u scripts/run_all.py --date scratch > run-scratch.log 2>&1 &
echo $! > run-scratch.pid
```

Server 2:

```bash
CVL_MODES=pretrained nohup python -u scripts/run_all.py --date pretrained > run-pretrained.log 2>&1 &
echo $! > run-pretrained.pid
```

Flag **`-u` wajib**. Tanpa itu Python menahan output di buffer dan log Anda tetap kosong berjam-jam meski prosesnya berjalan normal — mustahil dibedakan dari proses yang macet.

`echo $! > *.pid` menyimpan PID-nya supaya nanti bisa dicek atau dihentikan:

```bash
ps -p $(cat run-scratch.pid)       # masih hidup?
kill $(cat run-scratch.pid)        # hentikan
```

Alternatif dengan `tmux`, kalau Anda ingin bisa melihat outputnya secara langsung:

```bash
tmux new -s scratch
CVL_MODES=scratch python -u scripts/run_all.py --date scratch 2>&1 | tee run-scratch.log
# lepas: Ctrl+B lalu D   ·   sambung lagi: tmux attach -t scratch
```

> ⚠️ **Periksa baris pertama log sebelum meninggalkannya jalan.** Perintah di atas mencetak cakupan grid sebelum melatih apa pun:
>
> ```
> grid: archs=['resnet50', 'convnext_tiny', 'efficientnetv2_s', 'vit_small', 'swin_tiny'] levels=[1, 2, 3, 4] modes=['scratch'] seeds=[0, 1, 2, 3, 4]
> output: results/results-scratch.csv (0 run sudah ada -> di-skip) | ckpt: results/checkpoints-scratch
> ```
>
> Kalau daftarnya lebih pendek dari itu, ada file `.env` yang mempersempit grid — hapus atau ganti namanya, lalu ulangi. `.env` tidak ikut `git clone` (masuk `.gitignore`), tapi **ikut kalau repo disalin dengan `rsync`/`scp`**. Ini penting karena `.env` smoke-test bisa memangkas 200 run jadi 2 tanpa satu pun pesan error, dan `CVL_MAX_WRITERS` bahkan mengubah manifest di Langkah 3 sehingga jumlah kelasnya bukan 308.

Masing-masing menulis `results/results-<tag>.csv` dan `results/checkpoints-<tag>/`. Tiap run mencetak progres per epoch dan baris `done <run_id>: top1=... map=...` saat selesai.

**Kalau pod putus:** ulangi perintah yang **sama persis**. Resume bekerja per file CSV — run yang sudah tercatat di CSV itu dilewati. Mengganti tag `--date` berarti mulai dari nol.

## Memantau progres

Dua alat yang saling melengkapi. **`tail`** untuk melihat apa yang sedang dikerjakan detik ini:

```bash
tail -f run-scratch.log
```

**`progress.py`** untuk menjawab "sudah berapa persen, sisa berapa jam":

```bash
CVL_MODES=scratch    python scripts/progress.py --date scratch      # server 1
CVL_MODES=pretrained python scripts/progress.py --date pretrained   # server 2
```

Contoh keluarannya:

```
cakupan: archs=['resnet50', 'convnext_tiny', 'efficientnetv2_s', 'vit_small', 'swin_tiny'] levels=[1, 2, 3, 4] modes=['scratch'] seeds=[0, 1, 2, 3, 4] -> 100 run
sumber : results/results-scratch.csv

=== progres: 37/100 run selesai (37%) ===
         count   mean
mode
scratch     37  742.0
total waktu terpakai: 7.6 jam

=== sisa ===
scratch: 63 run x 742s = 13.0 jam

Estimasi sisa waktu: 13.0 jam (~0.5 hari)
```

Tiga hal yang perlu Anda tahu saat membacanya:

- **Prefiks `CVL_MODES=` wajib.** Tanpa itu `progress.py` menghitung kedua mode dan melaporkan 200 run total, sehingga estimasi sisanya jadi dua kali lipat — separuhnya milik server yang lain.
- **Baris `cakupan` adalah pemeriksaan `.env` yang sama** seperti di Langkah 5. Kalau daftarnya lebih pendek dari contoh di atas, grid Anda sedang dipersempit.
- **Estimasinya optimis di awal.** Ia memakai rata-rata run yang sudah selesai, sedangkan grid berjalan dari level kecil ke besar — L4 memakan sekitar tiga kali waktu L1. Angka yang keluar di 10% pertama akan meleset jauh ke bawah; baru mendekati kenyataan setelah setengah jalan.

## Langkah 6 — Jalankan Studi 2

Hanya di server 2, **setelah** Studi 1 selesai:

```bash
nohup python -u scripts/run_scenarios.py --date finetune > run-finetune.log 2>&1 &
echo $! > run-finetune.pid
```

Sekitar 2,4 jam. Pantau dengan `tail -f run-finetune.log` — `progress.py` tidak berlaku di sini karena ia menghitung grid arsitektur × level, bukan skenario. Untuk Studi 2, hitung barisnya langsung: `wc -l results/results-finetune.csv` (targetnya 26 baris — 25 run plus header).

Baseline `FT0` disalin dari `results-pretrained.csv`, bukan dijalankan ulang — sah karena berada di mesin, versi library, dan manifest yang sama. Kalau Studi 2 terpaksa pindah mesin, `FT0` harus dijalankan ulang di sana.

## Langkah 7 — Laporan

```bash
python scripts/make_report.py --results results/results-pretrained.csv --date pretrained
```

Menghasilkan tabel per arsitektur × level plus grafik akurasi-vs-N di `results/figures/`.

**Jangan kutip langsung angka dari perintah ini ke skripsi.** `make_report.py` merata-ratakan run kolaps dan run sehat jadi satu — lihat "Aturan pelaporan hasil" di bawah untuk kriteria kolaps dan cara memisahkannya sebelum melapor.

## Langkah 8 — Rangkai perbandingan Studi 2

Setelah `run_scenarios.py` menulis `results-finetune.csv`, jangan berhenti di situ:

- Baris baseline `FT0` **tidak ada** di `results-finetune.csv` (sengaja dilewati). Ambil dari `results-pretrained.csv`, dengan `run_id` berpola `convnext_tiny_L1_pretrained_s*` — bukan `FT0_s*`.
- `make_report.py` buta terhadap kolom `scenario`: ia mengelompokkan hanya berdasarkan `(arch, level, mode)`, jadi kalau dijalankan atas `results-finetune.csv` keenam skenario akan tercampur jadi satu baris. Jangan pakai untuk Studi 2.
- Perbandingan enam skenario dan uji-t berpasangan di §6 spec dilakukan **manual** dari kedua CSV di atas (gabungkan `FT0` dari `results-pretrained.csv` dengan `FT1`–`AUG` dari `results-finetune.csv`, lalu ikuti aturan kolaps yang sama seperti Studi 1).

---

## Aturan pelaporan hasil

Tiga aturan yang mempengaruhi cara angka dibaca — rinciannya di §6 spec.

**Run kolaps tidak masuk rata-rata.** Kriteria kolaps: `top1_page < 0,05` (model memprediksi ~1 kelas). Laporkan `kolaps 2/5; rata-rata run sehat 0,82 ± 0,03`, bukan rata-rata gabungan yang mencampur keduanya. Pada grid sebelumnya 31 dari 75 run scratch kolaps meski warmup aktif — ini temuan, bukan bug.

**Throughput hanya dari satu server.** `throughput_img_s` diukur saat inferensi sehingga hanya bergantung pada arsitektur dan hardware, bukan pada mode. Ambil angkanya dari server 2 saja; kolom yang sama dari server 1 diabaikan. Untuk biaya latih pakai `epochs_ran` yang bebas hardware, dan sebutkan GPU-jam per server terpisah beserta nama kartunya.

**Selisih di bawah 3,7 poin tidak terdeteksi.** Std antar-seed pada L1 adalah 0,0295; dengan uji-t berpasangan 5 seed, itu ambang deteksi minimumnya. Selisih 1–2 poin dilaporkan sebagai "tidak terdeteksi", bukan "sedikit lebih baik". Jangan pakai Wilcoxon signed-rank — pada n=5 nilai p terkecil yang mungkin adalah 0,0625, jadi hasilnya mustahil signifikan.

---

## Verifikasi lokal (tanpa GPU/dataset)

```bash
.venv/bin/pytest -q      # 97 test
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
