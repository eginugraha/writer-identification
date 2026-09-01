# Writer-ID Architecture Comparison (CVL)

Perbandingan 5 arsitektur (ResNet-50, ConvNeXt-Tiny, EfficientNetV2-S, ViT-Small, Swin-Tiny) untuk **klasifikasi penulis** pada dataset **CVL**, dengan ablasi data terbatas (halaman latih per penulis = 1–4) dan sumbu pretrained vs from-scratch.

Desain lengkap: [`docs/superpowers/specs/2026-08-05-grid-5seed-dan-finetune-convnext-design.md`](docs/superpowers/specs/2026-08-05-grid-5seed-dan-finetune-convnext-design.md)

## Rencana eksperimen

| | Studi 1 — Grid utama | Studi 2 — Fine-tuning |
|---|---|---|
| **Isi** | 5 arsitektur × 4 level × 2 mode × 5 seed | 6 skenario × 5 seed, Swin-Tiny di L1 |
| **Jumlah run** | 200 | 25 baru (baseline diambil dari Studi 1) |
| **Perkiraan** | ±41 jam GPU | ±5 jam GPU |
| **Dijalankan** | 2 server paralel | server 2, setelah pretrained selesai |

Tujuh skenario Studi 2 — masing-masing mengubah **satu** mekanisme saja: `FT0` baseline · `FT1` geometri input · `FT2` drop_path + label smoothing · `FT3` freeze parsial + LLRD · `FT4` head ArcFace · `FT5` 9-crop saat uji saja (tanpa latih) · `AUG` augmentasi kuat.

Hasil dipisah jadi tiga berkas: `results-scratch.csv`, `results-pretrained.csv`, `results-finetune-swin.csv`.

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

Kunci versinya untuk dipakai server 2 — **saring ke paket proyek saja**:

```bash
python -m pip freeze --local \
  | grep -iE "^(torch|torchvision|timm|numpy|pandas|pyarrow|scikit-learn|pillow|pyyaml|tqdm|matplotlib|pytest)==" \
  > requirements.lock.txt
cat requirements.lock.txt
```

Ini penting: `requirements.txt` tidak mengunci versi apa pun, dan timm sesekali mengubah tag bobot pretrained bawaan antar rilis. Dua pod yang dibuat selang beberapa hari bisa dapat versi berbeda.

> **Jangan pakai `pip freeze` polos.** Ia menangkap seluruh isi environment termasuk paket sistem bawaan image pod — `dbus-python`, `PyGObject`, `python-apt`. Di pod kedua paket-paket itu akan dicoba dibangun dari source dan gagal dengan pesan seperti `Did not find pkg-config` atau `Run-time dependency dbus-1 found: NO`. Padahal tidak satu pun dipakai proyek ini. Flag `--local` membuang paket di luar venv, dan saringan `grep` membatasi ke dependensi yang benar-benar dideklarasikan.

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

Unduh langsung di pod dari Zenodo — jauh lebih cepat daripada mengunggah dari laptop:

```bash
wget -O cvl-database-1-1.zip \
  "https://zenodo.org/records/1492267/files/cvl-database-1-1.zip?download=1"
```

Ekstraksi memakan beberapa menit, jadi jalankan di background:

```bash
nohup unzip cvl-database-1-1.zip > unzip.log 2>&1 &
```

Pantau dan pastikan selesai bersih:

```bash
tail -f unzip.log          # Ctrl+C untuk berhenti memantau
find cvl-database-1-1 -path "*/lines/*" -name "*.tif" | wc -l   # harus 13473
```

Struktur akhir yang dibaca pipeline:

```
cvl-database-1-1/
  trainset/lines/<writer>/<writer>-<page>-<line>.tif
  testset/lines/<writer>/<writer>-<page>-<line>.tif
```

**Bereskan ruangnya setelah ekstraksi.** Hanya folder `lines/` yang pernah dibaca; `words/`, `pages/`, dan `xml/` sama sekali tidak disentuh dan memakan 3,8 GB dari 5,1 GB dataset. Bersama berkas zip-nya, itu ruang yang lebih baik dipakai checkpoint:

```bash
rm cvl-database-1-1.zip
rm -rf cvl-database-1-1/*/words cvl-database-1-1/*/pages cvl-database-1-1/*/xml
du -sh cvl-database-1-1        # tersisa ~1,3 GB
```

> Selama ekstraksi, zip dan hasilnya sama-sama ada di disk — puncaknya sekitar 11 GB. Aman di volume 50 GB, tapi perhitungkan kalau volume Anda lebih kecil.

Alternatif kalau dataset sudah ada di laptop dan Anda lebih suka menyalinnya, ambil `lines/` saja:

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

Ulangi Langkah 1–3 di pod kedua, tapi pasang dependensinya dari file terkunci. **Torch dipasang lebih dulu dari indeks CUDA yang sama** — versi seperti `2.13.0+cu130` tidak ada di PyPI, jadi memasangnya langsung dari lock file akan gagal:

```bash
# 1. torch dari indeks CUDA yang sama dengan server 1 (sesuaikan cuXXX
#    dengan yang tercetak di Langkah 1b server 1)
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch torchvision

# 2. sisanya dari lock file
python -m pip install -r requirements.lock.txt

# 3. wajib: pastikan versinya benar-benar sama dengan server 1
python -m pip list | grep -Ei "^(torch|torchvision|timm|numpy) "
```

Langkah 3 bukan formalitas. Syarat "kedua pod identik" bertumpu pada versi torch dan timm yang sama; kalau berbeda, temuan stabilitas mode scratch tidak bisa dipertahankan. Bandingkan keluarannya dengan server 1 baris per baris sebelum melanjutkan.

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

Hanya di server 2, setelah bagian **pretrained** Studi 1 selesai di server itu. Tidak perlu menunggu mode scratch di server 1: Studi 2 tidak membaca `results-scratch.csv` sama sekali, jadi menunggu hanya membuat GPU menganggur.

Grid utama menempatkan Swin-Tiny di posisi teratas pada keempat level, jadi Studi 2 dijalankan di sana:

```bash
nohup python -u scripts/run_scenarios.py --arch swin_tiny --date finetune-swin \
  > run-finetune.log 2>&1 &
echo $! > run-finetune.pid
```

`--arch` **wajib disebut** dan hanya menerima arsitektur yang punya peta lapisan di `LAYER_MAP` (`src/cvl/finetune.py`), saat ini `convnext_tiny` dan `swin_tiny`. Dua batasan itu disengaja: FT3 memakai freeze + LLRD yang perlu tahu nama modul tiap keluarga arsitektur, dan flag yang terlupa berarti berjam-jam GPU di arsitektur yang salah. `--level` tersedia dan defaultnya 1.

**Perkiraan ±5 jam.** Dasarnya `train_time_s` baseline FT0 di `results-pretrained.csv` (`swin_tiny_L1_pretrained_s*`): rata-rata 711 detik per run, sebaran 570–823. Dikalikan 25 run = 4,9 jam, ditambah ~11 menit untuk evaluasi FT1 yang memakai 9 jendela atas 2.490 baris uji di 5 seed. Angka ini mengasumsikan tiap skenario sama mahalnya dengan baseline — asumsi yang tidak persis benar (FT3 lebih murah karena stem + 2 stage beku; AUG sedikit lebih mahal di sisi CPU), tapi keduanya berlawanan arah sehingga cenderung saling meniadakan. Rentang wajarnya 4,5–6 jam.

Baris pertama log mengonfirmasi setelannya:

```
skenario: ['FT1', 'FT2', 'FT3', 'FT4', 'AUG'] | seeds=[0, 1, 2, 3, 4] | arch=swin_tiny L1 | lr=0.0003 | device=cuda
baseline FT0: ambil dari results-pretrained.csv, pola swin_tiny_L1_pretrained_s*
```

`lr=0.0003` benar untuk Swin — `LR_OVERRIDES` hanya menurunkan LR ConvNeXt ke 1e-4, dan grid utama melatih Swin di 3e-4. Kalau Anda menjalankan `--arch convnext_tiny`, angkanya harus `lr=0.0001`.

Baseline `FT0` disalin dari `results-pretrained.csv`, bukan dijalankan ulang — sah karena berada di mesin, versi library, dan manifest yang sama. Kalau Studi 2 terpaksa pindah mesin, `FT0` harus dijalankan ulang di sana.

`run_id` Studi 2 memuat arsitektur dan level (`swin_tiny_L1_FT1_s0`), mengikuti pola grid utama. Dua arsitektur karena itu bisa menulis ke CSV yang sama tanpa saling melewati — tapi tetap pakai `--date` berbeda supaya folder checkpoint-nya terpisah.

### Memantau Studi 2

> **`progress.py` tidak berlaku di sini, dan tidak akan memberi tahu Anda soal itu.** Ia membangun daftar `run_id` yang diharapkan dari `{arch}_L{level}_{mode}_s{seed}` atas seluruh grid (`scripts/progress.py:22-31`). `run_id` Studi 2 menaruh nama skenario di posisi mode, jadi tidak satu pun cocok — laporannya berbunyi `0/200 run selesai (0%)` meski 25 run sudah beres. Bukan error, hanya salah.

```bash
tail -f run-finetune.log                       # jalannya run demi run
ps -p $(cat run-finetune.pid)                  # masih hidup?
wc -l results/results-finetune-swin.csv        # target 26 (25 run + header)
```

Ringkasan per skenario — lebih berguna daripada `wc -l` karena langsung memperlihatkan `top1_page` tiap skenario terhadap baseline FT0 (0,8084) tanpa menunggu semuanya selesai:

```bash
python -c "
import pandas as pd
d = pd.read_csv('results/results-finetune-swin.csv')
print(f'{len(d)}/25 run selesai ({100*len(d)/25:.0f}%) | terpakai {d.train_time_s.sum()/3600:.1f} jam | sisa ~{(25-len(d))*d.train_time_s.mean()/3600:.1f} jam')
print(d.groupby('scenario').agg(n=('run_id','size'), top1=('top1_page','mean'), detik=('train_time_s','mean')).round(3))
"
```

Dua hal saat membacanya:

- **Estimasi sisanya bias ke bawah di awal.** Urutan eksekusinya seed-di-luar (seed 0 → FT1..AUG, lalu seed 1), jadi rata-rata dari beberapa run pertama didominasi skenario yang kebetulan murah. FT3 paling terasa: `patch_embed` + dua stage awal beku membuatnya jauh lebih cepat dari yang lain.
- **`top1` per skenario baru bisa dibandingkan setelah kelima seed-nya lengkap.** Sebaran antar-seed di L1 sekitar ±0,005–0,011, jadi rata-rata dari 1–2 seed belum berarti apa-apa.

### Langkah 6b — FT5: 9-crop saat uji, tanpa latih ulang

FT1 mengubah geometri latih **dan** protokol uji sekaligus, jadi +14,2 poin `top1_page`-nya tidak bisa dibagi antara keduanya. FT5 memisahkannya: bobot FT0 apa adanya, dievaluasi dengan sembilan jendela yang dirata-rata — protokol uji yang sama persis dengan FT1, tanpa satu langkah latih pun.

```bash
python scripts/eval_only.py --arch swin_tiny \
  --src-ckpt-root results/checkpoints-pretrained --date evalonly-swin
```

**±11 menit**, bukan jam: yang dikerjakan hanya evaluasi 9 jendela atas 2.490 baris uji di 5 seed — biaya yang sama dengan porsi evaluasi FT1 pada perkiraan Langkah 6.

`--src-ckpt-root` **wajib disebut** dan menunjuk folder checkpoint **grid utama**, bukan folder Studi 2. Grid itu punya `--date` sendiri, jadi menebaknya dari `--date` di sini akan diam-diam menunjuk folder yang salah. Hasilnya ditulis ke CSV terpisah dengan kolom `source_run_id`; menulis ke `results-finetune-swin.csv` ditolak karena kolomnya beda dan `_append_row` hanya menulis header saat berkasnya belum ada.

> **Prasyarat yang mudah hilang.** `results/checkpoints*/` ada di `.gitignore` dan tidak ikut ke mana-mana. Kalau pod sudah dihapus atau folder itu sudah dibersihkan, `swin_tiny_L1_pretrained_s*/best.pt` tidak ada lagi dan FT5 mustahil dijalankan apa adanya — runner-nya berhenti menyebut path yang hilang. Jalan keluarnya: latih ulang FT0 lima seed di mesin dengan versi torch/timm yang sama seperti tercatat di `results-pretrained.csv`, lalu **cocokkan dulu** angka FT0 hasil latih ulang dengan yang lama sebelum FT5 dipakai sebagai pembanding.

`--source` mengisi posisi mode pada `run_id` sumber, jadi `--source FT1 --src-ckpt-root results/checkpoints-finetune-swin` mengukur kebalikannya: bobot FT1, tapi diuji satu potongan. Itu melengkapi tabel 2×2-nya (lihat [dokumentasi/04](dokumentasi/04-skenario-fine-tuning.md)), meski bukan bagian dari permintaan awal.

## Langkah 7 — Laporan

Satu perintah per berkas hasil — dijalankan di server yang memilikinya:

```bash
# server 2
python scripts/make_report.py --results results/results-pretrained.csv --date pretrained
# server 1
python scripts/make_report.py --results results/results-scratch.csv --date scratch
```

`make_report.py` membaca kolom `mode` dan hanya menulis bagian yang datanya ada. CSV scratch-only menghasilkan laporan scratch saja, dengan figure `acc_vs_n_scratch-scratch.png`; baris terakhirnya memberi tahu bagian mana yang dilewati:

```
report written to dokumentasi/08-hasil-eksperimen-scratch.md
figure written to results/figures/acc_vs_n_scratch-scratch.png
(mode pretrained tidak ada di CSV ini -> bagiannya dilewati)
```

Kalau Anda menggabungkan kedua CSV jadi satu berkas, laporannya memuat kedua bagian sekaligus — tidak perlu flag tambahan.

**Jangan kutip langsung angka dari tabel Top-1 ke skripsi.** `pivot_markdown` merata-ratakan run kolaps dan run sehat jadi satu, jadi rerata mode scratch di tabel itu mencampur dua populasi yang berbeda. Yang boleh dikutip adalah tabel **per-seed @ L4** dan kolom **kolaps** di bawahnya — lihat "Aturan pelaporan hasil".

Kalimat penutup bagian scratch dibangkitkan dari data, termasuk jumlah seed sebenarnya. Sebelumnya kalimat itu dipaku dari grid 3-seed yang lama, sehingga laporan 5-seed menutup dengan "kolaps 3/3" yang bertentangan dengan tabel di atasnya. Kalau Anda pernah menghasilkan laporan sebelum perbaikan ini, buang dan bangun ulang.

### Grafik tambahan

Dua bar chart untuk bab hasil — leaderboard pretrained dan trainability scratch:

```bash
python scripts/make_figures.py
```

Tanpa argumen ia membaca `results-pretrained.csv` dan `results-scratch.csv` sekaligus; pakai `--results PATH` (boleh diulang) untuk sumber lain. Mode yang tidak ada di CSV manapun dilewati dengan pesan, bukan digambar kosong. Jumlah seed pada label dan batas sumbu dihitung dari data, jadi grafiknya ikut benar saat jumlah seed berubah.

**Baca `scratch_trainability.png` dengan hati-hati.** Warna menandai kolaps, tinggi batang menandai rerata — dan keduanya tidak sejalan. ResNet-50 hijau (kolaps 0/5) tapi reratanya 0,486, di bawah ConvNeXt-T yang merah (0,614). Sebabnya ResNet tidak pernah kolaps total tapi sangat tidak stabil antar-seed (0,263–0,782 di L4). "Tidak kolaps" bukan berarti "berhasil dilatih"; sebutkan sebarannya saat mengutip grafik ini.

## Langkah 8 — Rangkai perbandingan Studi 2

Setelah `run_scenarios.py` menulis `results-finetune-swin.csv`, jangan berhenti di situ:

- Baris baseline `FT0` **tidak ada** di `results-finetune-swin.csv` (sengaja dilewati). Ambil dari `results-pretrained.csv`, dengan `run_id` berpola `swin_tiny_L1_pretrained_s*` — sesuaikan dengan `--arch` yang Anda pakai.
- `make_report.py` buta terhadap kolom `scenario`: ia mengelompokkan hanya berdasarkan `(arch, level, mode)`, jadi kalau dijalankan atas `results-finetune-swin.csv` keenam skenario akan tercampur jadi satu baris. Jangan pakai untuk Studi 2.
- Perbandingan enam skenario dan uji-t berpasangan di §6 spec dilakukan **manual** dari kedua CSV di atas (gabungkan `FT0` dari `results-pretrained.csv` dengan `FT1`–`AUG` dari `results-finetune-swin.csv`, lalu ikuti aturan kolaps yang sama seperti Studi 1).

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
