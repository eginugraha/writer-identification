# Lampiran D — Laporan revisi: membelah kontribusi FT1

Laporan ini menjawab satu masukan pembimbing dan berdiri sendiri: ia merangkum
apa yang dijalankan, hasilnya, dan klaim mana yang berubah karenanya. Isinya
sama dengan yang tersebar di [04](04-skenario-fine-tuning.md),
[07](07-hasil-eksperimen-finetune.md), dan
[08](08-hipotesis-dan-kesimpulan.md), disusun agar bisa dibaca sekali duduk.

**Ringkas.** FT1 mengubah geometri pelatihan dan protokol pengujian sekaligus,
sehingga kenaikan +14,22 poin tidak bisa dikaitkan ke salah satunya. Empat run
tambahan tanpa pelatihan memisahkannya — dan jawabannya bukan pembagian,
melainkan dua mekanisme yang saling menggantikan.

Swin-Tiny, L1, mode pretrained, LR 3e-4. Kohor 308 penulis, 2.490 baris uji.
Lima seed dengan pembagian data yang sama; seluruh uji berpasangan per seed,
df = 4, ambang dua sisi 5% pada |t| > 2,776. Per 2026-09-01.

## Masukan yang ditindaklanjuti

> Tambahkan satu kondisi eksperimen: terapkan 9-crop averaging saat inferensi
> pada model FT0 (baseline) tanpa mengubah training sama sekali. Ini akan
> menjawab: seberapa besar kontribusi masing-masing komponen FT1. Eksperimen ini
> tidak memerlukan pelatihan ulang model karena menggunakan checkpoint FT0 yang
> sudah ada.
>
> Jika 9-crop saja pada FT0 memberikan sebagian besar peningkatan → ubah klaim
> menjadi "test-time multi-crop ensemble adalah kunci, bukan training strategy".
> Jika 9-crop saja tidak banyak membantu → klaim sliding window training sebagai
> kunci menjadi lebih kuat.

Kondisi itu dijalankan sebagai **FT5**. Sekaligus ditambahkan sel keempatnya
(**FT6**) agar pemisahannya lengkap, dan dua skenario lama dinilai ulang di
protokol yang sama (**AUG@9**, **FT4@9**) karena peringkat Tabel 8 sebelumnya
membandingkan metode di dua protokol pengujian yang berbeda.

## Empat run tanpa satu langkah pelatihan

Keempatnya memuat ulang checkpoint yang sudah ada dan hanya mengganti cara
pengujiannya. Tidak ada bobot yang berubah, tidak ada epoch yang dijalankan;
biayanya hanya evaluasi.

| Kondisi | Bobot dari | Geometri latih | Protokol uji | Menjawab |
|---|---|---|---|---|
| FT5 | FT0 | `center` | 9 jendela | efek protokol uji saja |
| FT6 | FT1 | `linewindow` | 1 potongan | efek geometri latih saja |
| AUG@9 | AUG | `center` | 9 jendela | peringkat di protokol seragam |
| FT4@9 | FT4 | `center` | 9 jendela | peringkat di protokol seragam |

## Tabel 2×2 yang lengkap

`top1_page`, rata-rata 5 seed:

| | uji 1 potongan | uji 9 jendela |
|---|---|---|
| **latih `center`** | FT0 0,8084 | FT5 **0,9026** |
| **latih `linewindow`** | FT6 **0,9286** | FT1 0,9506 |

Dibaca lewat sisinya — dan di sinilah intinya terlihat:

- menambah **9-crop** bernilai **+9,42** poin di atas model yang dilatih
  `center`, tapi hanya **+2,21** poin di atas model yang sudah dilatih
  `linewindow`;
- menambah **sliding-window training** bernilai **+12,01** poin pada pengujian
  satu potongan, tapi hanya **+4,81** poin setelah 9-crop dipakai.

Rata-rata lengkapnya, ± simpangan baku:

| Kondisi | top1_page | top5_page | macro_f1 | map_line | top1_retr |
|---|---|---|---|---|---|
| FT0 baseline | 0,8084 ±0,005 | 0,9214 ±0,006 | 0,7590 ±0,008 | 0,2828 ±0,010 | 0,4003 ±0,011 |
| FT5 uji saja | **0,9026** ±0,018 | 0,9747 ±0,006 | 0,8741 ±0,020 | 0,8011 ±0,008 | 0,9313 ±0,005 |
| FT6 latih saja | **0,9286** ±0,022 | 0,9708 ±0,002 | 0,9071 ±0,029 | 0,4362 ±0,020 | 0,5680 ±0,029 |
| FT1 keduanya | **0,9506** ±0,019 | 0,9753 ±0,005 | 0,9353 ±0,025 | 0,8522 ±0,014 | 0,9605 ±0,006 |

Selisih berpasangan per seed, poin persentase; `*` = |t| > 2,776:

| Perbandingan | top1_page | top5_page | macro_f1 | map_line | top1_retr |
|---|---|---|---|---|---|
| FT5 − FT0 (uji) | +9,42 t=11,20 * | +5,32 t=15,93 * | +11,51 t=11,17 * | +51,83 t=310,32 * | +53,10 t=149,38 * |
| FT6 − FT0 (latih) | **+12,01** t=11,82 * | +4,94 t=15,68 * | **+14,81** t=10,65 * | +15,34 t=23,58 * | +16,77 t=16,56 * |
| FT1 − FT6 (+uji) | +2,21 t=5,31 * | +0,45 t=2,75 | +2,81 t=5,27 * | +41,60 t=103,62 * | +39,24 t=37,30 * |
| FT1 − FT5 (+latih) | +4,81 t=22,31 * | +0,06 t=1,00 | +6,12 t=16,16 * | +5,11 t=19,90 * | +2,92 t=21,93 * |
| FT1 − FT0 (total) | +14,22 t=15,17 * | +5,39 t=19,04 * | +17,63 t=13,40 * | +56,93 t=210,18 * | +56,02 t=208,30 * |

Porsi kenaikan yang dijelaskan tiap komponen sendirian, dan suku interaksinya:

| Komponen | top1_page | top5_page | macro_f1 | map_line | top1_retr |
|---|---|---|---|---|---|
| 9-crop saja (FT5) | 66,2% | 98,8% | 65,3% | **91,0%** | **94,8%** |
| latih linewindow saja (FT6) | **84,5%** | 91,6% | **84,0%** | 26,9% | 29,9% |
| interaksi | −7,21 | −4,87 | −8,69 | −10,23 | −13,86 |

## Apa yang terbaca

### Keduanya kunci, dan satu saja hampir cukup

Masing-masing komponen, berdiri sendiri, sudah memulihkan sebagian besar
kenaikan FT1: 9-crop 66,2% dan sliding-window training 84,5%. Jumlah keduanya
150,7% — mustahil, dan itulah intinya. Keduanya menutup keterbatasan yang sama,
yaitu 92,5% bagian baris yang tidak pernah dilihat model, dari ujung yang
berbeda. Suku interaksinya −7,21 poin.

Keempat sisi tabel tetap signifikan pada `top1_page`, termasuk yang terkecil
(+2,21; t = 5,31). Jadi tidak ada komponen yang bisa dibuang tanpa kerugian
terukur — yang berubah hanya besarannya.

### Di tingkat baris, pembagiannya justru terbalik

Untuk `map_line`, 9-crop menjelaskan 91,0% kenaikan sementara sliding-window
training hanya 26,9%; untuk `top1_retrieval`, 94,8% berbanding 29,9%. Metrik
retrieval bekerja pada fitur per baris, dan merata-ratakan sembilan jendela
membuat fitur itu jauh lebih stabil terlepas dari bobotnya. Di tingkat halaman
urutannya justru sebaliknya — sisi pelatihan yang lebih besar.

### Pada top-5, sisi pelatihan tidak menyumbang apa pun

`top5_page` adalah satu-satunya metrik yang jatuh bersih ke salah satu cabang:
98,8% kenaikannya dijelaskan 9-crop saja, dan sisa pelatihannya (+0,06;
t = 1,00) gagal uji-t. Begitu sistem boleh menebak lima kali, yang tersisa
hanyalah efek perata-rataan.

## Dua klaim yang direvisi

Eksperimen ini menemukan satu klaim yang keliru pada laporan sebelumnya, lalu
koreksi pertamanya sendiri ternyata kebablasan. Keduanya dicatat terbuka agar
jejaknya bisa ditelusuri.

**Ditarik — "yang membaik adalah representasinya".** Laporan lama memakai
lompatan `top1_retrieval` 0,400 → 0,961 sebagai bukti representasi membaik,
dengan alasan metrik itu tidak memakai kepala klasifikasi. FT5 memakai bobot FT0
yang identik — nol parameter berubah — dan sudah mencapai 0,9313, yaitu **94,8%**
dari lompatan itu. Argumennya tidak berlaku.

**Dinyatakan ulang — "representasinya tidak membaik sama sekali".** Koreksi
pertama, yang ditulis sebelum FT6 ada, menyimpulkan sebaliknya bahwa perbaikan
representasi tidak ada. FT6 membantahnya: pada protokol pengujian yang sama
persis dengan FT0, sliding-window training tetap menaikkan `map_line` **+15,34
poin** (t = 23,58). Representasinya memang membaik — jauh lebih kecil daripada
angka mentahnya, tapi nyata.

Yang bertahan adalah rumusan yang lebih sempit dan berlaku umum: **metrik
retrieval hanya menjadi bukti tentang representasi selama protokol pengujiannya
dipegang tetap.** Perbandingan FT6 − FT0 dan AUG@9 − FT5 memenuhi syarat itu;
FT1 − FT0 tidak.

## Peringkat di protokol pengujian yang sama

Tabel signifikansi Studi 2 menilai AUG dan FT4 dengan `eval_crops=1`, sementara
protokol pengujian sendirian bernilai +9,42 poin. Keduanya karena itu dinilai
ulang dengan sembilan jendela dan dibandingkan terhadap **FT5** — bukan FT0 —
karena hanya pasangan itu yang protokol ujinya identik dan hanya bobotnya yang
berbeda.

| Perbandingan | top1_page | macro_f1 | map_line | top1_retr |
|---|---|---|---|---|
| AUG@9 − FT5 | −1,23 t=−2,22 | −1,35 t=−2,31 | **+3,44** t=25,58 * | +1,47 t=9,49 * |
| FT4@9 − FT5 | +0,45 t=0,31 | +0,54 t=0,30 | **+5,18** t=4,48 * | +1,65 t=3,07 * |
| FT1 − FT5 | **+4,81** t=22,31 * | **+6,12** t=16,16 * | +5,11 t=19,90 * | +2,92 t=21,93 * |

**Peringkatnya tidak berbalik** — FT1 tetap menang telak walau semuanya dinilai
dengan protokol yang sama. Dan temuan lama justru menguat: "AUG dan FT4
memperbaiki representasi tingkat baris tanpa memperbaiki keputusan tingkat
halaman" kini terbukti di **dua protokol pengujian yang berbeda**, bukan hanya
satu.

Satu nuansa: selisih `top1_page` AUG berpindah arah dari +2,34 (protokol lama)
menjadi −1,23 (protokol 9-crop). Keduanya gagal uji-t, jadi yang benar dikatakan
bukan "AUG jadi merugikan", melainkan bahwa **di kedua protokol AUG tidak bisa
dibedakan dari baseline** pada akurasi halaman.

## Jawaban atas pertanyaan yang diajukan

Hasilnya tidak jatuh ke salah satu dari dua kemungkinan yang disebutkan dalam
masukan, melainkan ke kemungkinan ketiga yang hanya terlihat setelah sel keempat
dijalankan: **kedua komponen sendirian sudah memulihkan sebagian besar kenaikan,
dan keduanya saling menggantikan.**

Karena itu klaimnya tidak diubah menjadi "test-time multi-crop ensemble adalah
kunci". Rumusan yang dipakai sekarang:

1. Penghambat utamanya tetap **cakupan baris** — 92,5% bagian baris yang tidak
   pernah dilihat model, bukan arsitektur dan bukan overfitting.
2. Penghambat itu punya **dua pintu yang hampir setara**: apa yang dilihat model
   saat dilatih, dan apa yang dilihatnya saat diuji. Menutup salah satu saja
   sudah memulihkan 66–85% kenaikannya.
3. Pintu inferensinya **tidak menuntut pelatihan sama sekali**: +9,42 poin
   dengan nol detik latih, hanya sembilan kali biaya evaluasi. Ini perbaikan
   termurah dalam seluruh studi.
4. Konsekuensi metodologisnya melampaui tugas ini: **membandingkan metode yang
   protokol ujinya berbeda akan mengatributkan ke pelatihan apa yang sebenarnya
   milik inferensi** — dan peringkat Studi 2 sendiri sempat tersusun begitu.

## Yang belum dikerjakan

- **Jumlah jendela belum ditala.** Sembilan dipilih tanpa pencarian. Kurva
  akurasi terhadap `eval_crops` (1, 3, 5, 9, 15) murah didapat karena tidak
  butuh pelatihan.
- **Studi 1 masih memakai protokol lama.** Kalau protokol uji sendirian bernilai
  +9,42 poin, peringkat kelima arsitektur di L1–L4 juga disusun di bawah
  protokol yang merugikan semuanya. Checkpoint-nya ada.
- **FT1 di arsitektur lain dan di L2–L4** belum diuji, jadi kesimpulan tentang
  peringkat arsitektur masih terikat geometri `center` dan level L1.

## Reproduksi

Seluruh angka pada laporan ini dihitung ulang dari CSV dengan satu perintah, dan
run-nya sendiri tidak memerlukan GPU untuk pelatihan — hanya evaluasi atas
checkpoint yang sudah ada.

```bash
# menjalankan kondisi eval-only (tanpa pelatihan)
python scripts/eval_only.py --arch swin_tiny --scenario FT5 \
  --src-ckpt-root results/checkpoints-pretrained   --date evalonly-swin
python scripts/eval_only.py --arch swin_tiny --scenario FT6 --source FT1 \
  --src-ckpt-root results/checkpoints-finetune-swin --date evalonly-swin

# seluruh tabel di laporan ini
python scripts/banding_protokol.py --arch swin_tiny
```

Hasil mentahnya di `results/results-evalonly-swin.csv` (20 baris, masing-masing
mencatat `source_run_id` — checkpoint asal bobotnya). Desain skenario di
[04-skenario-fine-tuning.md](04-skenario-fine-tuning.md), hasil lengkap di
[07-hasil-eksperimen-finetune.md](07-hasil-eksperimen-finetune.md), sintesisnya
di [08-hipotesis-dan-kesimpulan.md](08-hipotesis-dan-kesimpulan.md).
