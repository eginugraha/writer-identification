# 04 — Skenario fine-tuning (Studi 2)

## Pertanyaan yang dijawab

Studi 1 menetapkan arsitektur mana yang terbaik. Studi 2 menanyakan hal berbeda:
**dari beberapa mekanisme yang lazim dipakai untuk memperbaiki fine-tuning, mana
yang benar-benar berpengaruh pada tugas ini?**

Dijalankan pada **Swin-Tiny di L1** — arsitektur pemenang di regime data paling
sedikit, tempat perbaikan paling dibutuhkan dan selisih antar-metode paling
mudah terlihat.

## Prinsip: satu skenario, satu mekanisme

`Scenario` (`src/cvl/scenarios.py`) adalah dataclass dengan tujuh medan. Tiap
skenario mengubah **tepat satu** medan dari baseline. Kalau dua diubah
bersamaan, perbandingan terhadap baseline tidak bisa mengaitkan selisihnya ke
salah satunya.

| Skenario | Medan yang diubah | Nilai |
|---|---|---|
| `FT0` | — | baseline apa adanya |
| `FT1` | `geometry`, `eval_crops` | `linewindow`, 9 |
| `FT2` | `drop_path`, `label_smoothing` | 0,2 dan 0,1 |
| `FT3` | `freeze_strategy` | `S3` |
| `FT4` | `head` | `arcface` |
| `AUG` | `aug` | `strong` |

`FT2` mengubah dua medan sekaligus, tapi keduanya adalah satu mekanisme yang
sama — regularisasi bawaan yang lazim dipakai bersama pada ConvNeXt/Swin.

**`FT0` tidak pernah dijalankan ulang.** `Scenario()` tanpa argumen identik
dengan konfigurasi grid utama, jadi barisnya disalin dari
`results-pretrained.csv` dengan pola `swin_tiny_L1_pretrained_s*`. Ini sah
selama berada di mesin, versi library, dan manifest yang sama.

## Rincian tiap skenario

### FT1 — cakupan baris

Mengganti geometri masukan dari `center` ke `linewindow`:

- **latih**: `ResizeHeight(224)` menyetel *tinggi* (bukan sisi pendek), lalu
  `RandomCrop(224)` mengambil jendela acak di sepanjang baris — jadi posisi
  jendela benar-benar bervariasi antar-epoch;
- **uji**: `eval_crops=9`, sembilan jendela tersebar merata sepanjang baris,
  probabilitasnya dirata-ratakan.

Ini satu-satunya skenario yang menyerang keterbatasan 7,5% cakupan baris yang
dijelaskan di [01-dataset.md](01-dataset.md#model-hanya-melihat-75-dari-tiap-baris).

> **Catatan metodologis penting.** FT1 mengubah **dua hal sekaligus**: geometri
> saat latih *dan* protokol evaluasi. Kenaikannya karena itu tidak bisa
> dikaitkan sepenuhnya ke salah satunya tanpa eksperimen tambahan. Ini harus
> disebut eksplisit saat melaporkan hasilnya.

### FT2 — regularisasi

`drop_path=0.2` (stochastic depth) dan `label_smoothing=0.1`. Keduanya resep
standar untuk arsitektur hierarkis modern. Hipotesisnya: dengan hanya ~7 baris
latih per penulis, overfitting adalah penghambat utama.

### FT3 — transfer learning selektif

Strategi `S3` dari `src/cvl/finetune.py`:

- bekukan **stem + dua stage pertama** (`patch_embed`, `layers.0`, `layers.1`
  untuk Swin);
- LR dasar diturunkan ke `1e-4`;
- **LLRD** dengan faktor 0,7 per level: kepala mendapat 1e-4, stage terdalam
  7e-5, berikutnya 4,9e-5, dan seterusnya ke arah lapisan awal.

Logikanya: fitur tingkat rendah dari ImageNet (tepi, tekstur) seharusnya sudah
memadai, jadi membekukannya melindungi dari overfitting sekaligus mempercepat
latih.

> **Strategi ini sempat mati diam-diam.** `STRATEGIES` semula menuliskan nama
> modul ConvNeXt secara harfiah (`stem`, `stages.0`). Pada Swin — yang menamai
> bloknya `patch_embed` dan `layers.N` — tidak ada prefiks yang cocok, sehingga
> nol parameter dibekukan dan LLRD runtuh jadi satu grup seragam. FT3 akan
> berjalan sampai selesai dan menghasilkan angka yang masuk akal, padahal
> isinya "FT0 dengan LR 1e-4". Sekarang strategi dinyatakan sebagai *kedalaman*
> dan namanya diambil dari `LAYER_MAP` per arsitektur; prefiks yang tidak cocok
> **menggagalkan run**, bukan didiamkan. Angka FT3 yang dilaporkan berasal dari
> kode setelah perbaikan ini.

### FT4 — kepala margin sudut

Mengganti kepala linear dengan ArcFace (`s=30.0`, `m=0.3`). L1 adalah 308
identitas dengan ~7 contoh per kelas — persis rezim tempat head margin lazim
dipakai di pengenalan wajah. Margin dinaikkan bertahap sepanjang warmup.

### AUG — augmentasi kuat

`RandomAffine` diperbesar (6°, geser 5%, skala 0,9–1,1, shear 5°),
`RandomResizedCrop` dengan `scale=(0.6,1.0)`, `ColorJitter` 0,4, dan
`RandomErasing(p=0.25)` setelah `ToTensor`.

**AUG sengaja tetap memakai `geometry="center"`.** Jalur `strong`+`center`
menyisipkan `CenterCrop((224, 246))` lebih dulu supaya `RandomResizedCrop`
mengacak di dalam strip yang **sama persis** dengan baseline. Jadi AUG murni
menguji *seberapa keras* citra diacak, bukan *bagian mana* dari baris yang
dilihat — pertanyaan kedua itu jatah FT1. Tanpa penyisipan itu, AUG akan
mencemari perbandingan dengan efek cakupan.

## Eksekusi

```bash
python scripts/run_scenarios.py --arch swin_tiny --date finetune-swin
```

25 run: 5 skenario × 5 seed (FT0 dilewati). Seed di luar, skenario di dalam.
Hasil ke `results/results-finetune-swin.csv`, checkpoint ke
`results/checkpoints-finetune-swin/`, `run_id` berpola `swin_tiny_L1_FT1_s0`.

`--arch` wajib disebut dan dibatasi ke arsitektur yang punya peta lapisan di
`LAYER_MAP`, karena FT3 membutuhkannya.

LR yang dipakai adalah **3e-4** — sama dengan yang dipakai grid utama untuk Swin,
sehingga sebanding dengan baseline FT0. (Kalau Studi 2 dijalankan di ConvNeXt,
`LR_OVERRIDES` otomatis menurunkannya ke 1e-4.) Kecuali FT3, yang memang
menurunkan LR ke 1e-4 sebagai bagian dari mekanismenya.
