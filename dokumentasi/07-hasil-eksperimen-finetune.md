# 07 — Hasil eksperimen fine-tuning (Studi 2)

**Status: selesai, 25/25 run** (5 skenario × 5 seed) per 2026-08-09. Sumber:
`results/results-finetune-swin.csv`. Baseline `FT0` diambil dari
`results-pretrained.csv`, pola `swin_tiny_L1_pretrained_s*`.

**Ditambah 20 run eval-only** per 2026-09-01 (`results/results-evalonly-swin.csv`),
tanpa satu pun pelatihan — semuanya menilai ulang checkpoint yang sudah ada
dengan protokol uji berbeda, lewat `scripts/eval_only.py`:

- **FT5** — bobot FT0, diuji 9 jendela;
- **FT6** — bobot FT1, diuji satu potongan tengah (sel keempat 2×2);
- **AUG@9** dan **FT4@9** — bobot AUG dan FT4, diuji 9 jendela, supaya
  peringkatnya bisa dibaca di bawah protokol uji yang sama dengan FT1.

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

Tiga kondisi eval-only lainnya (FT6, AUG@9, FT4@9) ada di
[Membelah FT1](#membelah-ft1-tabel-2x2-penuh) — semuanya memakai bobot yang sudah
ada di tabel ini, jadi menaruhnya di sini akan terbaca seolah mekanisme latih
yang berbeda.

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

### Membelah FT1: tabel 2×2 penuh

Keempat sel berbagi kelima seed yang sama, jadi seluruh perbandingan berpasangan
per seed. `top1_page`, rata-rata 5 seed:

| | uji 1 potongan | uji 9 jendela |
|---|---|---|
| **latih `center`** | FT0 0,8084 | FT5 **0,9026** |
| **latih `linewindow`** | FT6 **0,9286** | FT1 0,9506 |

Rata-rata lengkapnya:

| Kondisi | top1_page | top5_page | macro_f1 | map_line | top1_retr |
|---|---|---|---|---|---|
| FT5 (bobot FT0, uji 9) | 0,9026 ±0,018 | 0,9747 ±0,006 | 0,8741 ±0,020 | 0,8011 ±0,008 | 0,9313 ±0,005 |
| FT6 (bobot FT1, uji 1) | 0,9286 ±0,022 | 0,9708 ±0,002 | 0,9071 ±0,029 | 0,4362 ±0,020 | 0,5680 ±0,029 |

Selisih berpasangan, poin persentase; `*` = |t| > 2,776 (df=4):

| Perbandingan | top1_page | top5_page | macro_f1 | map_line | top1_retr |
|---|---|---|---|---|---|
| FT5 − FT0 (uji saja) | +9,42 t=11,20 * | +5,32 t=15,93 * | +11,51 t=11,17 * | +51,83 t=310,32 * | +53,10 t=149,38 * |
| FT6 − FT0 (latih saja) | **+12,01** t=11,82 * | +4,94 t=15,68 * | **+14,81** t=10,65 * | +15,34 t=23,58 * | +16,77 t=16,56 * |
| FT1 − FT6 (tambah uji) | +2,21 t=5,31 * | +0,45 t=2,75 | +2,81 t=5,27 * | +41,60 t=103,62 * | +39,24 t=37,30 * |
| FT1 − FT5 (tambah latih) | +4,81 t=22,31 * | +0,06 t=1,00 | +6,12 t=16,16 * | +5,11 t=19,90 * | +2,92 t=21,93 * |
| FT1 − FT0 (total) | +14,22 t=15,17 * | +5,39 t=19,04 * | +17,63 t=13,40 * | +56,93 t=210,18 * | +56,02 t=208,30 * |
| porsi — 9-crop saja | 66,2% | 98,8% | 65,3% | **91,0%** | **94,8%** |
| porsi — latih linewindow saja | **84,5%** | 91,6% | **84,0%** | 26,9% | 29,9% |
| interaksi | −7,21 | −4,87 | −8,69 | −10,23 | −13,86 |

### Peringkat di bawah protokol uji yang sama

Tabel signifikansi di atas menilai AUG dan FT4 dengan `eval_crops=1`, sementara
protokol uji sendirian bernilai +9,42 poin. Keduanya karena itu dinilai ulang
dengan 9 jendela dan dibandingkan terhadap **FT5** — bukan FT0 — karena hanya
pasangan itu yang protokol ujinya identik dan hanya bobotnya yang berbeda.

| Perbandingan | top1_page | macro_f1 | map_line | top1_retr |
|---|---|---|---|---|
| AUG@9 − FT5 | −1,23 t=−2,22 | −1,35 t=−2,31 | **+3,44** t=25,58 * | +1,47 t=9,49 * |
| FT4@9 − FT5 | +0,45 t=0,31 | +0,54 t=0,30 | **+5,18** t=4,48 * | +1,65 t=3,07 * |
| FT1 − FT5 | **+4,81** t=22,31 * | **+6,12** t=16,16 * | +5,11 t=19,90 * | +2,92 t=21,93 * |

Seluruh tabel di bagian ini dihitung ulang dengan
`python scripts/banding_protokol.py --arch swin_tiny`.

## Bacaan

### FT1 menang telak — dan kedua komponennya saling menggantikan

+14,2 poin `top1_page`, dan `map_line` naik tiga kali lipat dari 0,283 ke 0,852.
Yang tidak bisa dibaca dari angka itu sendirian adalah **dari mana** kenaikannya
datang, karena FT1 mengubah geometri latih dan protokol uji sekaligus. Tabel 2×2
menjawabnya, dan jawabannya bukan pembagian.

**Masing-masing komponen, sendirian, sudah memulihkan sebagian besar
kenaikannya.** 9-crop saja (FT5) memberi +9,42 poin — 66,2% dari total — tanpa
mengubah satu pun parameter. Sliding-window training saja (FT6) memberi +12,01
poin, yakni **84,5%**. Jumlah keduanya 150,7%, yang jelas mustahil: keduanya
**tumpang tindih**, dengan suku interaksi −7,21 poin.

Bacaan yang paling jujur adalah dua arah sekaligus:

- 9-crop menambah **+9,42** pada model yang dilatih `center`, tapi hanya
  **+2,21** pada model yang sudah dilatih `linewindow`;
- sliding-window training menambah **+12,01** kalau diuji satu potongan, tapi
  hanya **+4,81** kalau protokol 9-crop sudah dipakai.

Keduanya menyerang keterbatasan yang sama — 92,5% baris yang tidak terlihat —
dari ujung yang berbeda. Begitu satu dipasang, yang kedua jauh berkurang
gunanya. Karena itu pertanyaan "mana yang jadi kunci" tidak punya jawaban
tunggal: **keduanya kunci, dan satu saja sudah hampir cukup.** Kalau harus
memilih satu, sisi latih sedikit lebih kuat di tingkat halaman (84,5% berbanding
66,2%) — tapi sisi uji tidak menuntut pelatihan sama sekali.

Keempat sisi 2×2-nya tetap signifikan pada `top1_page`, termasuk yang terkecil
(+2,21; t = 5,31). Jadi tidak ada komponen yang bisa dibuang tanpa kerugian yang
terukur; yang berubah hanyalah besarannya.

**Pada metrik tingkat baris, pembagiannya justru terbalik dan ekstrem.** Untuk
`map_line`, 9-crop saja menjelaskan **91,0%** kenaikan sementara sliding-window
training saja hanya **26,9%**; untuk `top1_retrieval` 94,8% berbanding 29,9%.
Metrik retrieval bekerja pada fitur per baris, dan merata-ratakan sembilan
jendela membuat fitur itu jauh lebih stabil terlepas dari bobotnya.

**Koreksi berlapis terhadap edisi sebelumnya.** Edisi awal menyatakan lompatan
`top1_retrieval` 0,400 → 0,961 membuktikan yang membaik adalah *representasinya*.
Itu terlalu kuat: 94,8% lompatan itu didapat FT5 dengan bobot FT0 yang identik.
Tapi edisi berikutnya — yang ditulis sebelum FT6 ada — menyimpulkan sebaliknya
bahwa representasinya "tidak membaik sedikit pun", dan **itu terlalu kuat ke
arah yang lain.** FT6 memakai protokol uji yang sama dengan FT0 dan tetap
menaikkan `map_line` +15,34 poin (t = 23,58): sliding-window training memang
memperbaiki representasi, hanya saja sumbangannya jauh lebih kecil daripada yang
tersirat dari angka mentahnya. Yang tetap berlaku: **metrik retrieval tidak bisa
dipakai sebagai bukti perbaikan representasi kalau protokol ujinya ikut
berubah** — protokolnya harus dipegang tetap dulu, seperti pada perbandingan
FT6 − FT0 dan AUG@9 − FT5.

**Satu efek samping yang layak dicatat.** Simpangan baku antar-seed melonjak
begitu protokol 9-crop dipakai: 0,005 di FT0 menjadi 0,018 di FT5. Tapi FT6 juga
0,022 dengan protokol uji lama, jadi kenaikan variansi itu datang dari **kedua**
perubahan, bukan hanya dari protokol evaluasinya — dugaan yang saya tulis
sebelum FT6 ada.

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

**Dan polanya bertahan setelah keduanya dinilai ulang dengan 9-crop.** Terhadap
FT5 — protokol identik, hanya bobot yang berbeda — AUG memberi `map_line`
**+3,44** (t = 25,58) dan FT4 **+5,18** (t = 4,48), keduanya signifikan, sementara
`top1_page` keduanya gagal uji-t (−1,23 dan +0,45). Jadi kesimpulan "memperbaiki
representasi tingkat baris tanpa memperbaiki keputusan tingkat halaman" sekarang
berlaku di **dua protokol uji yang berbeda**, bukan hanya di satu — jauh lebih
kokoh daripada sebelumnya.

Yang berubah hanyalah arah selisih `top1_page` AUG, dari +2,34 (protokol lama)
menjadi −1,23 (protokol 9-crop). Keduanya gagal uji-t, jadi yang benar untuk
dikatakan bukan "AUG jadi merugikan", melainkan: **di kedua protokol, AUG tidak
bisa dibedakan dari baseline pada akurasi halaman.**

## Biaya

| Skenario | detik/run | epoch |
|---|---|---|
| AUG | 825 | 39,4 |
| FT2 | 786 | 38,0 |
| FT1 | 785 | 40,0 |
| FT4 | 735 | 36,4 |
| FT3 | 735 | 36,2 |
| *FT0* | *711* | *37,0* |
| **FT5 / FT6 / AUG@9 / FT4@9** | **0** | **—** |

Baris terakhir yang paling penting di tabel ini: **+9,42 poin dengan nol detik
pelatihan** (FT5). Yang dibayar hanyalah evaluasi sembilan kali lebih mahal — hitungan
menit untuk 2.490 baris uji, sekali saja. Untuk siapa pun yang sudah punya model
terlatih, ini perbaikan termurah yang tersedia di seluruh Studi 2.

FT1 memberi +14 poin dengan biaya latih hanya 10% di atas baseline — meski
evaluasinya sembilan kali lebih mahal. FT3 memang tercepat bersama FT4, tapi
kecepatan itu didapat dengan mengorbankan akurasi.

FT1 mentok cap 40 epoch di kelima seed, satu-satunya skenario yang begitu. Artinya
ia masih membaik saat anggaran habis — **angka 0,9506 adalah batas bawah**, dan
menaikkan `pretrained_epochs` kemungkinan masih menambah.
