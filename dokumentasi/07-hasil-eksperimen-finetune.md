# 07 — Hasil eksperimen fine-tuning (Studi 2)

**Status: selesai, 25/25 run** (5 skenario × 5 seed) per 2026-08-09. Sumber:
`results/results-finetune-swin.csv`. Baseline `FT0` diambil dari
`results-pretrained.csv`, pola `swin_tiny_L1_pretrained_s*`.

Semuanya Swin-Tiny di L1, mode pretrained, LR 3e-4 (kecuali FT3 yang menurunkan
ke 1e-4 sebagai bagian mekanismenya). Desain tiap skenario ada di
[04-skenario-fine-tuning.md](04-skenario-fine-tuning.md).

## Hasil utama

Rata-rata 5 seed ± simpangan baku:

| Skenario | top1_page | top5_page | macro_f1 | map_line | top1_retr |
|---|---|---|---|---|---|
| **FT1** cakupan baris | **0,9506** ±0,019 | **0,9753** ±0,005 | **0,9353** ±0,025 | **0,8522** ±0,014 | **0,9605** ±0,006 |
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

## Bacaan

### FT1 menang telak, dan bukan karena keberuntungan seed

+14,2 poin `top1_page` dan `map_line` naik tiga kali lipat dari 0,283 ke 0,852.
Nilai t untuk `map_line` mencapai 210 — sebarannya sangat sempit karena efeknya
jauh lebih besar daripada variasi antar-seed. Ini konfirmasi langsung hipotesis
cakupan baris: penghambat utama bukan arsitektur, melainkan 92,5% bagian baris
yang tidak pernah dilihat model.

Lompatan `top1_retrieval` dari 0,400 ke 0,961 sama pentingnya. Metrik ini tidak
memakai kepala klasifikasi sama sekali — ia mengukur apakah dua baris dari
penulis yang sama berdekatan di ruang fitur. Kenaikan sebesar itu berarti yang
membaik adalah **representasinya**, bukan sekadar batas keputusan classifier.

**Dua peringatan yang wajib menyertai angka ini.** Pertama, FT1 mengubah
geometri latih *dan* protokol evaluasi (`eval_crops=9`) sekaligus, jadi
kenaikannya tidak bisa dibagi antara keduanya tanpa eksperimen tambahan. Kedua,
sebagian kenaikan `map_line` bersifat mekanis: retrieval bekerja pada rata-rata
sembilan jendela, bukan satu potongan tengah — rata-rata dari lebih banyak
sampel memang lebih stabil. Untuk memisahkannya, perlu satu run tambahan dengan
`geometry="linewindow"` tapi `eval_crops=1`.

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

## Biaya

| Skenario | detik/run | epoch |
|---|---|---|
| AUG | 825 | 39,4 |
| FT2 | 786 | 38,0 |
| FT1 | 785 | 40,0 |
| FT4 | 735 | 36,4 |
| FT3 | 735 | 36,2 |
| *FT0* | *711* | *37,0* |

FT1 memberi +14 poin dengan biaya latih hanya 10% di atas baseline — meski
evaluasinya sembilan kali lebih mahal. FT3 memang tercepat bersama FT4, tapi
kecepatan itu didapat dengan mengorbankan akurasi.

FT1 mentok cap 40 epoch di kelima seed, satu-satunya skenario yang begitu. Artinya
ia masih membaik saat anggaran habis — **angka 0,9506 adalah batas bawah**, dan
menaikkan `pretrained_epochs` kemungkinan masih menambah.
