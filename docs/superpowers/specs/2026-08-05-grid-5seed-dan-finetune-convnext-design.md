# Desain: Grid 5 Seed + Studi Fine-Tuning ConvNeXt-Tiny

Tanggal: 2026-08-05 · Branch: `research` · Status: disetujui, siap masuk rencana implementasi

## 1. Latar & Tujuan

Grid 150-run sebelumnya (5 arsitektur × 5 level × 2 mode × 3 seed) sudah selesai dan
menghasilkan bab hasil, tetapi tiga hal membuatnya perlu diulang:

1. **3 seed terlalu sedikit untuk menyimpulkan apa pun tentang selisih antar-perlakuan.**
   Std antar-seed ConvNeXt-Tiny pada L1 pretrained adalah 0,0295. Dengan uji-t
   berpasangan α=0,05 dan n=3, selisih terkecil yang bisa dideteksi adalah **7,3 poin**.
   Dengan n=5 turun jadi **3,7 poin**.
2. **Model hanya melihat 7,5% dari setiap baris** (lihat §2). Temuan ini baru ditemukan
   setelah grid lama selesai.
3. **Model GPU tidak pernah tercatat** di `results.csv` maupun log mana pun, sehingga
   klaim reproduktibilitas dan angka efisiensi tidak bisa diverifikasi.

Pekerjaan dibagi jadi dua studi berurutan:

- **Studi 1** — grid utama diulang dengan 5 seed, dijalankan paralel di dua server.
- **Studi 2** — studi skenario fine-tuning + augmentasi pada ConvNeXt-Tiny di level L1,
  dijalankan setelah Studi 1 selesai.

## 2. Temuan yang mendasari desain: 92,5% tiap baris dibuang

Citra baris CVL berasio ekstrem — median **1739 × 137 px** (≈12:1, p5 7,1 / p95 16,0).
Pipeline saat ini di `src/cvl/dataset.py:9-16`:

```
Grayscale → Resize(224) → RandomAffine → RandomResizedCrop(224, scale=(0.8,1.0), ratio=(0.9,1.1)) → ColorJitter → ToTensor → Normalize
```

`T.Resize(224)` dengan argumen skalar menyetel **sisi pendek** ke 224, sehingga citra
1745×119 menjadi **3284×224**. `RandomResizedCrop` lalu diminta mencari potongan seluas
80–100% area dengan rasio 0,9–1,1; potongan seperti itu menuntut tinggi 680–840 px pada
citra yang tingginya hanya 224 px. Batasan itu **tidak pernah terpenuhi**, dan torchvision
jatuh ke *fallback* center-crop deterministik.

Diverifikasi secara empiris: 10 pemanggilan `get_params` berturut-turut mengembalikan
potongan yang identik, `(i=0, j=1519, h=224, w=246)`.

Dua akibatnya:

1. **92,5% tiap baris tidak pernah masuk ke model.** Yang terlihat hanya ~1–2 kata di
   tengah baris.
2. **`RandomResizedCrop` bukan augmentasi sama sekali** — potongannya selalu sama persis.

Mengubah parameter `scale` (mis. ke 0,6–1,0) **tidak berefek apa pun**; sudah diuji dan
mengembalikan potongan yang identik. Ini menjelaskan mengapa kenaikan pada eksperimen
`S8_strongaug` milik Indra berasal dari rotasi/shear/jitter/erasing, bukan dari crop.

Temuan ini mengenai kelima arsitektur di grid utama. **Keputusan: Studi 1 tetap memakai
geometri lama.** Memperbaikinya di grid utama akan mengubah seluruh angka bab hasil dan
memaksa penulisan ulang bab pembahasan. Perbaikan geometri diuji secara terkendali sebagai
satu skenario di Studi 2 (FT1), sehingga dampaknya terukur tanpa membatalkan grid.

## 3. Studi 1 — Grid utama, 5 seed, dua server

### 3.1 Ruang lingkup

| Sumbu | Nilai | Jumlah |
|---|---|---|
| Arsitektur | resnet50, convnext_tiny, efficientnetv2_s, vit_small, swin_tiny | 5 |
| Level ablasi | 1, 2, 3, 4 (`full` tetap di-drop, ≈L4) | 4 |
| Mode | pretrained, scratch | 2 |
| Seed | 0, 1, 2, 3, 4 | 5 |

Total **200 run**, proyeksi **±41 jam GPU** (pretrained 20,3 jam; scratch 20,5 jam),
diekstrapolasi dari waktu latih grid lama. Proyeksi ini memakai waktu latih dari GPU grid
lama yang **modelnya tidak pernah tercatat**; bila kartu yang dipakai kali ini berbeda,
angkanya bergeser proporsional. Ini salah satu alasan kolom `gpu_name` ditambahkan (§5.1). Semua hyperparameter lain tidak berubah dari
grid sebelumnya, termasuk warmup 3 epoch dan override LR ConvNeXt pretrained ke 1e-4
(`src/cvl/run_experiments.py:11`).

### 3.2 Pembagian dua server

| Server | Perintah | Keluaran |
|---|---|---|
| 1 | `CVL_MODES=scratch python scripts/run_all.py --date scratch` | `results/results-scratch.csv`, `results/checkpoints-scratch/` |
| 2 | `CVL_MODES=pretrained python scripts/run_all.py --date pretrained` | `results/results-pretrained.csv`, `results/checkpoints-pretrained/` |

Pemisahan mode sudah didukung native lewat `CVL_MODES` (`src/cvl/config.py:106`) dan
penamaan keluaran lewat `--date` yang sudah ada. Tidak perlu kode baru.

### 3.3 Keabsahan lintas server

**Split data reproducible — terverifikasi.** `scan_lines` (`src/cvl/data_prep.py:18`)
memakai `rglob` yang urutannya bergantung filesystem, dan urutan itu masuk ke
`rng.permutation` saat memilih halaman latih dan baris validasi. Diuji dengan mengacak
urutan hasil scan tiga kali dan membandingkan identitas fisik tiap baris beserta split-nya:
**identik 3 dari 3**. Penyebabnya `groupby("writer")` dan `sorted(pages)` menormalkan
urutan sebelum RNG dipakai. Karena kolom `path` menyimpan path absolut mesin, tiap server
**wajib** membangun manifest-nya sendiri.

**Metrik efisiensi tidak sebanding lintas server.** `train_time_s` dan
`throughput_img_s` mengukur mesin yang berbeda. `throughput_img_s` khususnya menyesatkan:
ia diukur saat inferensi sehingga hanya bergantung pada arsitektur dan hardware, bukan pada
mode — secara fisik ConvNeXt scratch dan ConvNeXt pretrained punya throughput identik.
Aturan pelaporan:

- **Throughput** dilaporkan **hanya dari server 2**, satu angka per arsitektur. Kolom
  throughput dari server 1 diabaikan.
- **Biaya latih** dilaporkan memakai **`epochs_ran`** sebagai ukuran utama (bebas
  hardware, sekaligus menangkap early stopping). GPU-jam disebutkan terpisah per server
  beserta nama GPU-nya, tidak dijumlahkan menjadi satu angka.

**Dua hal yang wajib dikunci sebelum mulai:**

1. **Model GPU harus sama di kedua pod.** Bukan soal kecepatan, tapi numerik: AMP aktif
   (`configs/default.yaml`) dan perilaku TF32/bf16 berbeda antar generasi GPU. Temuan
   utama mode scratch adalah klaim **stabilitas** ("kolaps x/5"), dan stabilitas optimisasi
   adalah hal yang paling peka terhadap presisi numerik. Menyewa dua pod bertipe sama tidak
   menambah biaya dan menghapus keberatan "apakah kolaps itu properti arsitektur atau
   properti GPU".
2. **Versi library dipin.** `requirements.txt` saat ini tidak mengunci apa pun
   (`torch`, `torchvision`, `numpy` polos; `timm>=1.0.0` terbuka ke atas), dan timm
   sesekali mengubah tag bobot pretrained bawaan antar rilis. Jalankan
   `pip freeze > requirements.lock.txt` di server pertama, pasang server kedua dari file itu.

## 4. Studi 2 — Skenario fine-tuning ConvNeXt-Tiny di L1

### 4.1 Pemilihan arsitektur dan level

**Arsitektur: ConvNeXt-Tiny, dikunci di muka.** Di grid lama ConvNeXt-Tiny (0,7684) dan
Swin-Tiny (0,7608) berselisih 0,0076 pada L1 dengan std antar-seed 0,03 — selisihnya empat
kali lebih kecil dari derau, dan sama rapatnya di L4 (0,9773 vs 0,9762). Selisih sebesar itu
tidak akan pernah jadi klaim yang bisa dipertahankan ke arah mana pun, jadi pemilihannya
dinyatakan terbuka di metodologi berdasarkan alasan yang bisa disebutkan: unggul tipis di L1
*dan* L4, punya resep regularisasi bawaan yang matang (`drop_path`), dan `src/cvl/finetune.py`
sudah ditulis untuk arsitekturnya.

**Level: L1 saja.** Hanya L1 yang punya ruang gerak nyata (0,7684; jarak 15 poin ke L2).
L3 (0,9556) dan L4 (0,9773) sudah mepet plafon dengan std antar-seed 0,019 di L3, sehingga
kenaikan 1–2 poin pun tidak akan terdeteksi.

Skala data L1: **2.285 baris latih, 310 baris validasi, 2.490 baris uji, 308 kelas** —
rata-rata **7,4 contoh latih per kelas** (min 3, maks 15), dilatih dengan model 28 juta
parameter.

### 4.2 Enam skenario

Semua skenario memakai konfigurasi dasar identik dan hanya mengubah **satu mekanisme**:
ConvNeXt-Tiny pretrained, L1, LR 1e-4, weight decay 0,05, warmup 3 epoch + cosine,
40 epoch, patience 8, batch 64, AMP, seed 0–4.

| Kode | Nama | Mekanisme yang diubah |
|---|---|---|
| FT0 | Baseline (kontrol) | — |
| FT1 | Geometri input | Cakupan baris yang terlihat model |
| FT2 | Regularisasi native | `drop_path_rate` + label smoothing |
| FT3 | Freeze parsial + LLRD | Param group optimizer |
| FT4 | Head ArcFace | Fungsi loss & head |
| AUG | Augmentasi kuat | Kekuatan augmentasi |

**FT0 — Baseline.** Pipeline apa adanya. **Tidak perlu dijalankan**: Studi 1 sudah
menghasilkan ConvNeXt × L1 × pretrained × 5 seed dengan konfigurasi persis sama, jadi
barisnya disalin dari `results-pretrained.csv`. Studi 2 karenanya hanya **25 run baru**
(±2,4 jam GPU).

Penggunaan ulang ini sah karena Studi 2 dijalankan di **server 2**, mesin yang sama yang
menghasilkan jalur pretrained di Studi 1 — jadi baseline dan kelima skenario berbagi GPU,
versi library, dan manifest yang sama. Jika karena satu dan lain hal Studi 2 terpaksa
dijalankan di mesin lain, FT0 **harus** dijalankan ulang di mesin itu (5 run, ±28 menit)
dan baris dari Studi 1 tidak boleh dipakai.

**FT1 — Geometri input.** Ganti `Resize(224)` + crop-tengah dengan: tinggi diskalakan ke 224
(aspek dipertahankan, baris jadi ~3284×224), lalu **jendela 224×224 diambil acak sepanjang
baris**. Karena tingginya sudah tepat 224, crop-nya benar-benar bekerja — tersedia ~14,7
posisi jendela per baris, berbeda tiap epoch. Ini memberi dua hal sekaligus: augmentasi
posisional yang selama ini absen, dan lintas 40 epoch model melihat seluruh baris.

Saat evaluasi: **9 jendela merata sepanjang baris**, softmax dirata-ratakan per baris
sebelum agregasi halaman yang ada di `src/cvl/evaluate.py:31`. Vektor fitur untuk retrieval
dirata-ratakan dengan cara yang sama. Waktu latih tidak berubah; evaluasi ~9× lebih lambat
(hitungan detik).

**FT2 — Regularisasi native ConvNeXt.** `drop_path_rate=0.2` (stochastic depth, resep bawaan
ConvNeXt yang belum pernah diaktifkan — `src/cvl/models.py:6` tidak meneruskan parameter ini)
+ `label_smoothing=0.1`. Menyerang overfitting pada 7,4 contoh/kelas dengan 28 juta parameter.

**FT3 — Freeze parsial + LLRD.** Memakai strategi `S3` di `src/cvl/finetune.py` apa adanya:
bekukan `stem` + `stages.0-1`, base_lr 1e-4, decay 0,7 per level. Ini wakil "transfer learning
klasik" sekaligus menuntaskan pekerjaan Indra yang menggantung. `base_lr` S3 kebetulan sudah
1e-4, sama dengan override ConvNeXt di pipeline, jadi tidak ada konflik definisi baseline.

**FT4 — Head ArcFace.** Ganti head Linear dengan classifier kosinus + margin sudut aditif.
**`s=30`, `m=0,3` dipatok di muka** (nilai standar dari papernya) dan tidak diutak-atik setelah
melihat hasil — ini menghilangkan ruang *p-hacking*. Margin di-warmup linear dari 0 ke 0,3
sepanjang 3 epoch warmup yang sudah ada di `src/cvl/train.py:39-46`; ini penawar baku untuk
kegagalan konvergensi ArcFace di awal latihan. Jika sebagian seed kolaps, dilaporkan memakai
aturan kolaps di §6, bukan disembunyikan.

Dipilih di atas alternatif "Mixup + CutMix + EMA" karena alternatif itu menggabungkan tiga
mekanisme dalam satu skenario sehingga hasilnya tidak bisa diatribusikan — bertentangan dengan
prinsip satu-skenario-satu-mekanisme. ArcFace juga sejalan dengan literatur writer
identification dan mempengaruhi metrik retrieval, sehingga memberi dua sinyal.

**AUG — Augmentasi kuat.** Rotasi 6°, translasi 0,05, skala affine 0,9–1,1, shear 5,
ColorJitter 0,4, RandomErasing p=0,25 (ditempatkan **setelah** `ToTensor` karena beroperasi
pada tensor).

Untuk `crop 0,6–1,0`: di bawah geometri baseline parameter ini mati total (§2). Supaya
benar-benar aktif **tanpa** mengubah geometri — yang akan membuatnya rancu dengan FT1 —
urutannya jadi: `Resize(224)` → ambil strip tengah 224×246 (wilayah yang persis sama dengan
yang dilihat baseline) → `RandomResizedCrop(224, scale=(0.6,1.0), ratio=(0.9,1.1))` pada
strip itu, menghasilkan keluaran 224×224. Di dalam strip 224×246 batasan rasio kini dapat
dipenuhi, sehingga crop benar-benar mengacak. Dengan begitu AUG murni menguji kekuatan
augmentasi dan FT1 murni menguji cakupan input — keduanya tidak saling mencemari.

### 4.3 Menjalankan

```
python scripts/run_scenarios.py --date finetune
```

Keluaran: `results/results-finetune.csv` (dengan kolom `scenario`) dan
`results/checkpoints-finetune/`.

## 5. Perubahan kode

Dipecah dua agar Studi 1 yang memakan 41 jam GPU tidak menunggu kode Studi 2 selesai ditulis.

### 5.1 Kelompok 1 — minimal, supaya Studi 1 bisa mulai

1. **`src/cvl/config.py:23`** — `ALL_SEEDS = [0, 1, 2, 3, 4]`.
2. **`scripts/prep_manifests.py`** — tanpa perubahan kode; dijalankan ulang di tiap server
   untuk membangkitkan manifest seed 3 dan 4. Karena `build_manifest` deterministik terhadap
   seed, manifest seed 0–2 tereproduksi identik.
3. **`src/cvl/run_experiments.py:25`** — tambah kolom `gpu_name`
   (`torch.cuda.get_device_name(0)`, atau `"cpu"`), `torch_version`, dan `timm_version` ke
   baris yang ditulis `_append_row`.
4. `pip freeze > requirements.lock.txt` di server pertama.
5. **`README.md` ditulis ulang** — ringkas, berisi rencana kedua studi dan langkah eksekusi
   cloud dua server secara berurutan. Sisa contoh `--date rerun-warmup` dan bagian yang tidak
   lagi dipakai dibuang. Ini prasyarat praktis: langkah-langkahnya dieksekusi manusia di pod
   sewaan, jadi harus benar sebelum 41 jam itu mulai.

Tidak ada perubahan pada logika latih maupun evaluasi, sehingga hasil Studi 1 tetap sebanding
dengan grid lama bila sewaktu-waktu perlu dirujuk dari git history.

### 5.2 Kelompok 2 — mesin skenario

Prinsip: semua yang baru masuk lewat satu pintu, dan pintu itu **default-nya adalah perilaku
sekarang**. `Scenario()` tanpa argumen harus menghasilkan pipeline yang identik dengan Studi 1
— ini yang membuat FT0 sah dipakai sebagai baseline gratis.

**Modul baru `src/cvl/scenarios.py`** — satu registry, satu sumber kebenaran:

```python
@dataclass(frozen=True)
class Scenario:
    geometry: str = "center"            # "center" | "linewindow"
    aug: str = "baseline"               # "baseline" | "strong"
    drop_path: float = 0.0
    label_smoothing: float = 0.0
    freeze_strategy: str | None = None  # delegasi ke finetune.py
    head: str = "linear"                # "linear" | "arcface"
    eval_crops: int = 1

SCENARIOS = {
    "FT0": Scenario(),
    "FT1": Scenario(geometry="linewindow", eval_crops=9),
    "FT2": Scenario(drop_path=0.2, label_smoothing=0.1),
    "FT3": Scenario(freeze_strategy="S3"),
    "FT4": Scenario(head="arcface"),
    "AUG": Scenario(aug="strong"),
}
```

Modul yang tersentuh, masing-masing satu tanggung jawab:

- **`src/cvl/dataset.py`** — `build_transforms(train, geometry, aug)`. Sumbu `geometry`
  mengatur *bagian mana* dari baris yang dilihat; sumbu `aug` mengatur *seberapa keras*
  diacak. Dipisah agar FT1 dan AUG tidak saling mencemari.
- **`src/cvl/models.py`** — teruskan `drop_path_rate` ke `timm.create_model`, dan sediakan
  penggantian head ArcFace.
- **`src/cvl/train.py`** — terima `Scenario`; terapkan `label_smoothing`, param group dari
  `finetune.py`, dan warmup margin ArcFace.
- **`src/cvl/evaluate.py`** — rata-ratakan softmax dan fitur atas `eval_crops` jendela per
  baris sebelum agregasi halaman. Dengan `eval_crops=1` jalurnya persis seperti sekarang.
- **`scripts/run_scenarios.py`** — CLI baru meniru pola `run_grid`; resume-able lewat
  kombinasi kolom `scenario` + `seed` di CSV.

**Tidak disentuh:** `src/cvl/metrics.py`, `src/cvl/data_prep.py`, `src/cvl/report.py`, dan
jalur `run_grid` untuk Studi 1.

## 6. Protokol statistik & aturan pelaporan

**Metrik utama: `top1_page`**, ditetapkan di muka. Pendukung: `top5_page`, `macro_f1_page`,
`map_line`, `top1_retrieval` — dua terakhir khususnya relevan untuk FT4.

**Perbandingan berpasangan.** Semua skenario memakai seed 0–4 dan manifest yang sama,
sehingga tiap skenario dipasangkan seed-per-seed dengan FT0.

**Kepekaan.** Std antar-seed ConvNeXt L1 pretrained = 0,0295. Selisih terkecil yang dapat
dideteksi (uji-t berpasangan, α=0,05):

| Jumlah seed | Selisih minimum terdeteksi |
|---|---|
| 3 | 7,3 poin |
| 5 | 3,7 poin |

Angka 3,7 poin adalah batas atas; karena berpasangan, std selisih biasanya lebih kecil dari
std antar-seed. Konsekuensi yang harus dipatuhi saat menulis: skenario yang menggeser 1–2
poin dilaporkan sebagai **"tidak terdeteksi"**, bukan "sedikit lebih baik".

**Jangan pakai Wilcoxon signed-rank.** Dengan n=5, nilai p terkecil yang mungkin pada uji dua
sisi adalah 0,0625 — secara matematis tidak akan pernah mencapai 0,05 berapa pun besar
efeknya. Gunakan **uji-t berpasangan** dan laporkan **selisih rata-rata beserta interval
kepercayaan 95%**.

**Konfirmatori vs eksploratori.** FT1 dan AUG punya dasar bukti sebelum run (92,5% informasi
terbuang; +6,6 poin pada eksperimen `S8_strongaug`) dan dinyatakan **konfirmatori**, dikoreksi
Holm untuk dua perbandingan. FT2, FT3, FT4 dinyatakan **eksploratori**: p dilaporkan tanpa
koreksi, disertai pernyataan eksplisit bahwa ketiganya bukan uji konfirmatori.

**Aturan kolaps.** Kriteria: `top1_page < 0,05`. Berlaku untuk mode scratch di Studi 1 dan
untuk FT4 di Studi 2. Run yang kolaps **tidak pernah masuk rata-rata**. Format pelaporan:

> ConvNeXt-Tiny scratch L4 — kolaps 2/5; rata-rata run tidak kolaps 0,82 ± 0,03

bukan rata-rata gabungan seperti "0,275", yang mencampur run kolaps dan run sehat sehingga
tidak menggambarkan satu run pun yang benar-benar terjadi.

**Konteks kolaps yang sudah diketahui.** Warmup menyelesaikan kolaps pada jalur **pretrained**
sepenuhnya (0 dari 75 run kolaps; Top-1 terendah 0,6916), tetapi **tidak** pada jalur scratch:
31 dari 75 run scratch tetap kolaps meski warmup aktif (ConvNeXt 13/15, Swin 13/15, ViT 4/15
seluruhnya di L1–L2, ResNet 1/15 di L1, EfficientNet 0/15). Ini temuan, bukan bug. Biayanya
kecil karena run kolaps mati cepat lewat early stopping (rata-rata 14 epoch vs 55 epoch untuk
run sehat; 15% dari jam GPU scratch).

Catatan untuk penulisan ulang bab hasil: klaim "CNN dan ViT tidak pernah kolaps" hanya berlaku
di L4. Di L1, ViT-Small kolaps 3/3 dan ResNet-50 kolaps 1/3. Framing yang bertahan lintas level
adalah "ConvNeXt dan Swin kolaps di hampir semua level, sementara arsitektur lain hanya kolaps
saat data paling sedikit".

## 7. Pengujian

Suite saat ini berjumlah 26 test dan seluruhnya lolos tanpa GPU; pola itu dipertahankan.
Tambahan:

- `Scenario()` default menghasilkan transform yang identik dengan `build_transforms` lama —
  ini penjaga utama keabsahan FT0 sebagai baseline gratis.
- Geometri `linewindow` benar-benar mengembalikan jendela berbeda antar pemanggilan. Wajib
  dites eksplisit karena kegagalan diam-diam persis seperti inilah yang menimpa
  `RandomResizedCrop` selama ini.
- `eval_crops=9` menghasilkan bentuk tensor dan jumlah baris keluaran yang benar, dan
  `eval_crops=1` identik dengan jalur evaluasi lama.
- Forward ArcFace berjalan dan margin naik sesuai jadwal warmup.
- Registry `SCENARIOS` utuh: tiap nama punya definisi lengkap dan dapat dibangun.

## 8. Risiko

| Risiko | Penanganan |
|---|---|
| Dua server memakai GPU berbeda → klaim stabilitas scratch tidak bisa dipertahankan | Sewa dua pod bertipe kartu sama; catat `gpu_name` per run |
| Versi library berbeda antar server | `requirements.lock.txt` dari `pip freeze` |
| FT0 menyimpang dari baseline Studi 1 → baseline gratis tidak sah | `Scenario()` default = perilaku sekarang, dijaga oleh test |
| ArcFace gagal konvergen | `s`/`m` dipatok di muka + warmup margin; kolaps dilaporkan, bukan disembunyikan (biaya hanya ~28 menit GPU) |
| Pod putus di tengah grid | `run_grid` resume-able per file CSV; ulangi perintah `--date` yang sama |
| Efek 1–2 poin ditafsirkan sebagai kemenangan | Ambang 3,7 poin dinyatakan di muka pada §6 |

## 9. Di luar cakupan

- Memperbaiki geometri crop di grid utama (Studi 1) — diuji terkendali sebagai FT1 saja.
- Menjalankan Studi 2 pada Swin-Tiny.
- Level L2–L4 untuk Studi 2.
- Mereproduksi `S5`, `S6`, `S8_strongaug` milik Indra — definisinya tidak ada di
  `src/cvl/finetune.py` (yang hanya memuat S0–S4), jadi angkanya bersifat indikatif dan tidak
  dapat dikutip.
- Perhitungan GFLOPs (sudah tercatat sebagai known-minor sejak grid lama).
