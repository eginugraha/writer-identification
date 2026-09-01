# 07 — Hasil eksperimen fine-tuning (Studi 2)

**Status: selesai, 25/25 run** (5 skenario × 5 seed) per 2026-08-09. Sumber:
`results/results-finetune-swin.csv`. Baseline `FT0` diambil dari
`results-pretrained.csv`, pola `swin_tiny_L1_pretrained_s*`.

**Ditambah 5 run FT5** per 2026-09-01 (`results/results-evalonly-swin.csv`).
FT5 tidak melatih apa pun: ia mengevaluasi ulang checkpoint FT0 yang sama dengan
protokol uji FT1 (sembilan jendela dirata-rata), untuk membelah kenaikan FT1
antara sisi latih dan sisi uji. Dijalankan lewat `scripts/eval_only.py`.

Semuanya Swin-Tiny di L1, mode pretrained, LR 3e-4 (kecuali FT3 yang menurunkan
ke 1e-4 sebagai bagian mekanismenya; FT5 tidak punya LR karena tidak melatih).
Desain tiap skenario ada di
[04-skenario-fine-tuning.md](04-skenario-fine-tuning.md).

## Hasil utama

Rata-rata 5 seed ± simpangan baku:

| Skenario | top1_page | top5_page | macro_f1 | map_line | top1_retr |
|---|---|---|---|---|---|
| **FT1** cakupan baris (latih + uji) | **0,9506** ±0,019 | **0,9753** ±0,005 | **0,9353** ±0,025 | **0,8522** ±0,014 | **0,9605** ±0,006 |
| **FT5** 9-crop uji saja, bobot FT0 | 0,9026 ±0,018 | 0,9747 ±0,006 | 0,8741 ±0,020 | 0,8011 ±0,008 | 0,9313 ±0,005 |
| AUG augmentasi kuat | 0,8318 ±0,029 | 0,9474 ±0,020 | 0,7882 ±0,034 | 0,3235 ±0,019 | 0,4489 ±0,031 |
| FT4 ArcFace | 0,8266 ±0,030 | 0,9260 ±0,024 | 0,7817 ±0,037 | 0,3362 ±0,026 | 0,4596 ±0,030 |
| *FT0 baseline* | *0,8084* ±0,005 | *0,9214* ±0,006 | *0,7590* ±0,008 | *0,2828* ±0,010 | *0,4003* ±0,011 |
| FT2 regularisasi | 0,8078 ±0,016 | 0,9416 ±0,012 | 0,7564 ±0,018 | 0,2823 ±0,009 | 0,4042 ±0,011 |
| FT3 freeze + LLRD | 0,7506 ±0,031 | 0,9143 ±0,015 | 0,6933 ±0,036 | 0,2463 ±0,008 | 0,3496 ±0,014 |

**Tidak ada satu pun run yang kolaps** (`top1_page < 0,05`) di seluruh Studi 2,
jadi semua rata-rata di atas sah dibaca apa adanya — berbeda dari mode scratch.

## Signifikansi

Uji-t berpasangan per seed terhadap FT0, df=4, ambang dua sisi 5% |t| > 2,776.
Selisih dalam poin persentase:

| Skenario | Δ top1_page | t | Δ map_line | t |
|---|---|---|---|---|
| FT1 | **+14,22** | +15,17 ✓ | **+56,93** | +210,18 ✓ |
| AUG | +2,34 | +1,71 | +4,06 | +7,96 ✓ |
| FT4 | +1,82 | +1,46 | +5,34 | +4,38 ✓ |
| FT2 | −0,06 | −0,09 | −0,05 | −0,25 |
| FT3 | **−5,78** | −4,61 ✓ | −3,65 | −5,80 ✓ |

FT5 tidak masuk tabel itu karena ia bukan mekanisme latih yang sejajar dengan
yang lain — ia memakai bobot FT0. Pembelahannya ada di tabel berikut.

### Membelah FT1: FT0 → FT5 → FT1

Ketiganya berbagi kelima seed yang sama, jadi seluruh perbandingan di bawah
berpasangan per seed. Selisih dalam poin persentase; `*` = |t| > 2,776.

| Perbandingan | top1_page | top5_page | macro_f1 | map_line | top1_retr |
|---|---|---|---|---|---|
| **FT5 − FT0** (efek protokol uji) | **+9,42** t=11,20 * | +5,32 t=15,93 * | **+11,51** t=11,17 * | +51,83 t=310,32 * | +53,10 t=149,38 * |
| **FT1 − FT5** (efek geometri latih) | **+4,81** t=22,31 * | +0,06 t=1,00 | **+6,12** t=16,16 * | +5,11 t=19,90 * | +2,92 t=21,93 * |
| FT1 − FT0 (total) | +14,22 t=15,17 * | +5,39 t=19,04 * | +17,63 t=13,40 * | +56,93 t=210,18 * | +56,02 t=208,30 * |
| **porsi yang dijelaskan 9-crop saja** | **66,2%** | 98,8% | **65,3%** | 91,0% | 94,8% |

Dihitung ulang kapan saja dengan `python scripts/banding_ft5.py --arch swin_tiny`.

## Bacaan

### FT1 menang telak — tapi dua pertiganya milik protokol uji

+14,2 poin `top1_page`, dan `map_line` naik tiga kali lipat dari 0,283 ke 0,852.
Nilai t-nya besar di semua metrik: efeknya jauh melampaui variasi antar-seed.
Yang tidak bisa dibaca dari angka itu sendirian adalah **dari mana** kenaikannya
datang, karena FT1 mengubah geometri latih dan protokol uji sekaligus. FT5
menjawabnya.

**Dua pertiga kenaikan tidak butuh pelatihan sama sekali.** FT5 memakai
checkpoint FT0 apa adanya — nol parameter berubah — dan hanya merata-ratakan
sembilan jendela saat uji. Hasilnya +9,42 poin `top1_page` (t = 11,20), yaitu
**66,2%** dari total +14,22. Pada `macro_f1` porsinya serupa, 65,3%.

**Sepertiga sisanya nyata dan bukan sisa yang bisa diabaikan.** FT1 − FT5 =
+4,81 poin dengan **t = 22,31** — nilai t yang justru lebih tinggi daripada
bagian ensemble-nya, karena selisih itu sangat konsisten di kelima seed. Sebagai
pembanding, +4,81 poin lebih besar daripada efek AUG (+2,34) maupun FT4 (+1,82),
dua skenario yang tetap dilaporkan sebagai temuan. Jadi sliding-window training
tetap menyumbang, hanya saja bukan pemeran utamanya.

**Pada `macro_f1`, porsi latihnya lebih besar** (+6,12; t = 16,16) daripada di
`top1_page`. Macro-F1 membobot semua penulis sama rata, jadi ini menunjukkan
sliding-window training paling menolong penulis-penulis yang sulit — persis
bagian yang tidak tertolong hanya dengan merata-ratakan sembilan jendela.

**Pada `top5_page`, sisi latih tidak menyumbang apa pun.** FT1 − FT5 = +0,06
poin, t = 1,00, satu-satunya sel yang gagal uji-t di seluruh pembelahan ini:
98,8% kenaikan `top5_page` sudah dijelaskan 9-crop saja. Begitu boleh menebak
lima kali, yang tersisa hanyalah efek perata-rataan.

**Koreksi terhadap edisi sebelumnya.** Sebelum FT5 ada, dokumen ini menyatakan
lompatan `top1_retrieval` 0,400 → 0,961 membuktikan yang membaik adalah
*representasinya*, bukan sekadar batas keputusan classifier. **Itu keliru.** FT5
memakai bobot FT0 yang identik dan sudah mencapai 0,9313 — **94,8%** dari
lompatan itu — sementara `map_line` 91,0%. Representasi per jendela tidak
membaik sedikit pun; yang membaik adalah kestabilan rata-rata sembilan jendela
dibanding satu potongan tengah. Kecurigaan "sebagian kenaikan `map_line`
bersifat mekanis" yang dicatat di edisi lama ternyata bukan sebagian, melainkan
hampir seluruhnya. Metrik retrieval karena itu **tidak boleh** dipakai sebagai
bukti perbaikan representasi selama protokol ujinya ikut berubah.

**Satu efek samping yang layak dicatat.** Simpangan baku antar-seed melonjak
begitu protokol 9-crop dipakai: 0,005 di FT0 menjadi 0,018 di FT5, dan FT1
mewarisinya (0,019). Kenaikan variansi itu datang dari protokol evaluasinya,
bukan dari cara melatihnya.

**Yang masih belum diketahui.** Angka +4,81 adalah efek geometri latih *dengan
syarat* protokol uji 9-crop sudah dipakai. Apakah sliding-window training sendirian
— dilatih `linewindow` tapi diuji satu potongan tengah — juga menolong, belum
diuji; itu sel keempat dari tabel 2×2 di
[04-skenario-fine-tuning.md](04-skenario-fine-tuning.md#ft5--protokol-uji-ft1-tanpa-perubahan-latih),
dan bisa dijalankan tanpa latih ulang dengan
`scripts/eval_only.py --source FT1`.

### FT3 justru merugikan — dan itu informatif

−5,78 poin, signifikan. Membekukan `patch_embed` + dua stage awal ternyata
membuang kapasitas yang dibutuhkan, bukan melindungi dari overfitting.

Ini masuk akal begitu dipikirkan: fitur tingkat rendah ImageNet dilatih pada foto
alami — tepi berwarna, tekstur, gradien pencahayaan. Tulisan tangan adalah goresan
tinta biner pada kertas. Justru **lapisan awal** yang paling perlu beradaptasi,
dan itulah yang dibekukan FT3.

Angka ini baru bisa dipercaya setelah perbaikan peta lapisan yang dijelaskan di
[04-skenario-fine-tuning.md](04-skenario-fine-tuning.md#ft3--transfer-learning-selektif);
sebelum itu FT3 diam-diam identik dengan baseline.

### FT2 tidak berpengaruh sama sekali

−0,06 poin dengan t = −0,09. Ini bukan "efek kecil", melainkan **nol**.
Hipotesis bahwa overfitting adalah penghambat utama di L1 tertolak. Konsisten
dengan FT1: yang kurang bukan regularisasi, melainkan informasi.

### AUG dan FT4 memperbaiki representasi tapi bukan akurasi halaman

Keduanya menunjukkan pola yang sama dan menarik: **`map_line` naik signifikan
(+4,06 dan +5,34) sementara `top1_page` tidak** (+2,34 dan +1,82, keduanya
gagal uji-t).

Penjelasannya masuk akal. Keduanya bekerja pada tingkat baris — augmentasi kuat
dan margin sudut sama-sama memaksa fitur per baris lebih terpisah. Tapi
`top1_page` sudah merata-ratakan ~8 baris per halaman, dan perata-rataan itu
sendiri sudah meredam sebagian besar ketidakpastian tingkat baris. Perbaikan
yang nyata di tingkat baris karena itu sebagian besar terserap sebelum sampai ke
keputusan halaman.

Implikasi praktisnya: **kalau tugasnya identifikasi dari satu baris saja** —
bukan satu halaman penuh — AUG dan FT4 layak dipakai meski di sini terlihat
tidak berpengaruh.

Berbeda dari FT1, kenaikan `map_line` di sini **tidak** bisa dijelaskan efek
perata-rataan: AUG dan FT4 memakai `eval_crops=1` persis seperti FT0, jadi
protokol ujinya identik dan yang berubah hanya bobotnya. Di kedua skenario itu
representasinya memang membaik.

## Biaya

| Skenario | detik/run | epoch |
|---|---|---|
| AUG | 825 | 39,4 |
| FT2 | 786 | 38,0 |
| FT1 | 785 | 40,0 |
| FT4 | 735 | 36,4 |
| FT3 | 735 | 36,2 |
| *FT0* | *711* | *37,0* |
| **FT5** | **0** | **—** |

FT5 adalah baris yang paling penting di tabel ini: **+9,42 poin dengan nol detik
pelatihan.** Yang dibayar hanyalah evaluasi sembilan kali lebih mahal — hitungan
menit untuk 2.490 baris uji, sekali saja. Untuk siapa pun yang sudah punya model
terlatih, ini perbaikan termurah yang tersedia di seluruh Studi 2.

FT1 memberi +14 poin dengan biaya latih hanya 10% di atas baseline — meski
evaluasinya sembilan kali lebih mahal. FT3 memang tercepat bersama FT4, tapi
kecepatan itu didapat dengan mengorbankan akurasi.

FT1 mentok cap 40 epoch di kelima seed, satu-satunya skenario yang begitu. Artinya
ia masih membaik saat anggaran habis — **angka 0,9506 adalah batas bawah**, dan
menaikkan `pretrained_epochs` kemungkinan masih menambah.
