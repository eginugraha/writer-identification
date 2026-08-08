# 08 — Hipotesis dan kesimpulan

Sintesis dari 225 run: 100 scratch, 100 pretrained, 25 fine-tuning.

## Hipotesis yang diuji

| # | Hipotesis | Hasil |
|---|---|---|
| H1 | Arsitektur transformer hierarkis unggul di data terbatas | **Didukung**, tapi selisihnya menyusut cepat |
| H2 | Pretraining wajib untuk arsitektur modern pada skala ini | **Didukung kuat** |
| H3 | Penghambat utama pada L1 adalah overfitting | **Tertolak** |
| H4 | Penghambat utama pada L1 adalah cakupan baris | **Didukung kuat** |
| H5 | Transfer learning selektif (freeze lapisan awal) membantu | **Tertolak, arahnya terbalik** |

---

## H1 — Swin-Tiny unggul, tapi hanya saat data paling sedikit

`top1_page`, rata-rata 5 seed, mode pretrained:

| arch | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| **swin_tiny** | **0,8084** | **0,9442** | **0,9695** | **0,9799** |
| convnext_tiny | 0,7883 | 0,9279 | 0,9604 | 0,9695 |
| vit_small | 0,7682 | 0,9299 | 0,9591 | 0,9760 |
| efficientnetv2_s | 0,7344 | 0,9000 | 0,9558 | 0,9734 |
| resnet50 | 0,7052 | 0,8734 | 0,9279 | 0,9481 |

Swin menang di keempat level dan ResNet-50 kalah di keempat level — dua-duanya
tanpa pengecualian.

**Temuan yang lebih penting daripada peringkatnya: jarak antar-arsitektur
menyusut dari 10,3 poin di L1 menjadi 3,2 poin di L4.** Pilihan arsitektur
paling menentukan justru ketika data paling sedikit. Di L4 semuanya berkumpul di
0,95–0,98, dan memilih arsitektur hampir tidak berarti lagi.

Uji-t berpasangan memperkuat pembacaan itu: keunggulan Swin signifikan terhadap
**semua** lawan di L1, tapi di L4 hanya tersisa signifikan terhadap ResNet-50.

Perbandingan yang paling informatif adalah **ViT-S vs Swin-T**: keduanya
transformer, berukuran serupa, tapi Swin punya hierarki resolusi dan jendela
lokal. Selisihnya +4,03 poin di L1 (signifikan) tapi menyusut jadi +0,39 poin di
L4. Bias induktif lokal terbayar justru saat data sedikit — persis yang
diprediksi teori, dan besarnya terukur di sini.

> **Kehati-hatian.** Di L4 selisih Swin−ViT sebesar +0,39 poin lolos uji-t hanya
> karena variansi berpasangannya sangat kecil. 0,39 poin setara 1,2 halaman dari
> 308. Signifikan secara statistik, tidak berarti secara praktis. Laporkan ukuran
> efeknya, jangan cuma bintangnya.

## H2 — Pretraining bukan pilihan, melainkan syarat

Dari 100 run scratch, **36 kolaps** (`top1_page < 0,05`, memprediksi ~1 kelas
dari 308). Dari 100 run pretrained, **nol**.

Jumlah seed yang kolaps per arsitektur × level:

| arch | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| swin_tiny | 4/5 | 5/5 | 5/5 | 4/5 |
| convnext_tiny | 5/5 | 5/5 | 3/5 | 1/5 |
| vit_small | 2/5 | 1/5 | 0/5 | 0/5 |
| resnet50 | 1/5 | 0/5 | 0/5 | 0/5 |
| efficientnetv2_s | 0/5 | 0/5 | 0/5 | 0/5 |

Polanya terbalik sempurna dari peringkat pretrained: **arsitektur yang paling
diuntungkan pretraining adalah yang paling tidak bisa dilatih tanpanya.** Swin
juara mode pretrained sekaligus paling rapuh mode scratch (18 dari 20 run
kolaps). EfficientNetV2-S peringkat empat mode pretrained tapi tidak pernah
kolaps sekali pun, dan justru **terbaik** mode scratch (0,895 di L4).

Ini didapat meski scratch diberi anggaran hampir empat kali lipat
(150 vs 40 epoch) dan warmup 3 epoch yang sama. Warmup memperbaiki kolaps mode
pretrained sepenuhnya, tapi tidak menyelesaikan masalah mode scratch.

**Kesimpulan praktis: pada skala 308 kelas dengan ≤4 halaman per penulis,
arsitektur hierarkis modern tidak dapat dilatih dari nol secara andal.** Bukan
"kurang akurat" — melainkan gagal konvergen pada mayoritas seed.

### ResNet-50: tidak kolaps bukan berarti berhasil

ResNet-50 tidak pernah kolaps di L2–L4, tapi sebaran per seed di L4 adalah
0,539 / 0,571 / 0,263 / 0,273 / 0,782. Reratanya 0,486 — **di bawah ConvNeXt-T
yang kolaps 1/5** (0,614). Kriteria kolaps biner tidak menangkap ketidakstabilan
semacam ini; laporkan sebarannya, bukan cuma jumlah kolaps.

## H3 dan H4 — bukan overfitting, melainkan informasi

Dua hipotesis bersaing tentang penghambat utama di L1, diuji langsung oleh Studi 2:

**H3 (overfitting) tertolak.** FT2 menambahkan `drop_path=0.2` dan
`label_smoothing=0.1` — resep regularisasi standar. Hasilnya **−0,06 poin,
t = −0,09**. Bukan efek kecil, melainkan nol.

**H4 (cakupan baris) didukung kuat.** FT1 mengganti geometri masukan sehingga
model melihat seluruh panjang baris, bukan strip tengah 7,5%. Hasilnya
**+14,22 poin `top1_page`** (t = +15,17) dan `map_line` naik dari 0,283 ke 0,852.

Selisihnya tidak sebanding: satu mekanisme memberi nol, satunya memberi 14 poin.
Yang kurang pada L1 bukan pengekangan kapasitas model, melainkan **informasi yang
sampai ke model**.

Kenaikan `top1_retrieval` dari 0,400 ke 0,961 memperkuatnya. Metrik itu tidak
memakai kepala klasifikasi sama sekali, jadi yang membaik adalah representasinya
— bukan sekadar batas keputusan.

> **Batasan yang wajib disebut.** FT1 mengubah geometri latih *dan* protokol
> evaluasi (`eval_crops=9`) sekaligus. Kenaikannya karena itu belum bisa dibagi
> antara "model melihat lebih banyak saat latih" dan "prediksi dirata-ratakan
> atas sembilan jendela saat uji". Eksperimen pemisahnya sederhana —
> `geometry="linewindow"` dengan `eval_crops=1` — dan belum dijalankan.

## H5 — membekukan lapisan awal justru merugikan

FT3 (bekukan stem + 2 stage awal, LR 1e-4, LLRD 0,7) turun **−5,78 poin**,
signifikan.

Arahnya terbalik dari dugaan umum. Penjelasannya: fitur tingkat rendah ImageNet
dilatih pada foto alami — tepi berwarna, tekstur, gradien pencahayaan. Tulisan
tangan adalah goresan tinta biner pada kertas. Justru lapisan paling awal yang
paling perlu beradaptasi ke domain baru, dan itulah yang dibekukan.

Angka ini hanya sahih setelah perbaikan peta lapisan per arsitektur; sebelumnya
FT3 diam-diam identik dengan baseline pada Swin.

## Kesimpulan

1. **Swin-Tiny adalah pilihan terbaik untuk writer-ID pada CVL**, unggul di
   keempat level data. Tapi keunggulannya hanya bermakna praktis di regime data
   sangat terbatas; pada 4 halaman per penulis, kelima arsitektur setara.

2. **Pretraining adalah syarat, bukan optimasi.** Arsitektur hierarkis modern
   (Swin, ConvNeXt) gagal konvergen dari nol pada mayoritas seed meski diberi
   anggaran epoch hampir empat kali lipat. Jika pretraining tidak tersedia,
   EfficientNetV2-S adalah satu-satunya pilihan yang andal di sini.

3. **Penghambat terbesar bukan arsitektur maupun overfitting, melainkan
   pra-pemrosesan.** Memperbaiki cakupan baris memberi +14,2 poin — lebih besar
   daripada seluruh jarak antar-arsitektur di L1 (10,3 poin). Satu keputusan
   pra-pemrosesan mengalahkan seluruh pilihan arsitektur.

4. **Regularisasi tambahan tidak membantu; freeze lapisan awal merugikan.** Dua
   resep yang lazim dianggap "praktik baik" untuk fine-tuning tidak terbukti di
   tugas ini — satu nol, satu negatif.

5. **Augmentasi kuat dan ArcFace memperbaiki representasi tingkat baris**
   (`map_line` +4,06 dan +5,34, keduanya signifikan) **tanpa memperbaiki akurasi
   tingkat halaman.** Perata-rataan ~8 baris per halaman sudah menyerap sebagian
   besar manfaatnya. Untuk tugas yang hanya punya satu baris, keduanya tetap
   layak dipakai.

## Yang belum dikerjakan

- **Memisahkan efek FT1.** Satu run dengan `geometry="linewindow"` dan
  `eval_crops=1` akan membelah +14,2 poin itu menjadi kontribusi latih dan
  kontribusi evaluasi. Ini pekerjaan paling bernilai berikutnya.
- **FT1 pada arsitektur lain.** Kalau cakupan baris memang penghambat dominan,
  peringkat arsitektur bisa berubah setelah semuanya memakai `linewindow`.
  Kesimpulan 1 hanya berlaku di bawah geometri `center`.
- **FT1 di L2–L4.** Seluruh Studi 2 dijalankan di L1. Manfaat cakupan baris
  mungkin menyusut saat data bertambah, seperti halnya selisih antar-arsitektur.
- **Anggaran epoch untuk FT1.** Ia mentok cap 40 epoch di kelima seed, jadi
  0,9506 adalah batas bawah.
- **Kombinasi FT1 + AUG/FT4.** Ketiganya memperbaiki hal yang berbeda dan belum
  pernah diuji bersamaan.
