# Kesimpulan

Penelitian ini membandingkan lima arsitektur *deep learning*—ResNet-50,
EfficientNetV2-S, ViT-S/16, Swin-T, dan ConvNeXt-T—untuk tugas *writer
identification* pada dataset CVL, di bawah dua skenario pelatihan (*pretrained*
dan *scratch*), empat tingkat ukuran data latih (L1–L4), dan tiga *seed*.
Berdasarkan hasil eksperimen, beberapa kesimpulan dapat ditarik.

## 1. Transfer learning efektif untuk semua arsitektur

Dengan bobot pra-latih ImageNet, seluruh arsitektur mencapai akurasi Top-1 yang
tinggi pada data terbanyak (0.942–0.977) dan menunjukkan peningkatan monoton
seiring bertambahnya data latih. Ini menegaskan bahwa *transfer learning*
merupakan pendekatan yang andal untuk writer identification pada dataset
berskala terbatas seperti CVL.

## 2. Arsitektur modern mengungguli baseline, tetapi selisihnya moderat

Diurut berdasarkan rerata Top-1 lintas L1–L4, Swin-T dan ConvNeXt-T menempati
posisi teratas (keduanya 0.904), disusul EfficientNetV2-S (0.893), ViT-S/16
(0.876), dan ResNet-50 sebagai baseline terendah (0.863). Keunggulan arsitektur
modern bersifat konsisten namun moderat (~4 poin Top-1 atas ResNet-50).
Keunggulan ini paling menonjol pada rezim data sangat sedikit (L1), di mana
ConvNeXt-T dan Swin-T memimpin—mengindikasikan manfaat *inductive bias* dan
fitur pra-latih ketika data minim.

## 3. Efisiensi: EfficientNetV2-S paling seimbang

EfficientNetV2-S dan ViT-S/16 menawarkan rasio akurasi-per-parameter terbaik
(~21 juta parameter dengan akurasi kompetitif), sementara Swin-T dan ConvNeXt-T
membutuhkan kapasitas lebih besar (~28 juta) untuk peningkatan marginal. Bila
efisiensi menjadi pertimbangan utama, EfficientNetV2-S adalah pilihan paling
seimbang; bila akurasi absolut diprioritaskan, Swin-T atau ConvNeXt-T lebih
tepat.

## 4. ConvNeXt/Swin tidak dapat dilatih dari nol secara andal

Pada pelatihan dari *scratch* dengan resep identik, muncul disparitas stabilitas
yang tajam. CNN (ResNet-50, EfficientNetV2-S) dan ViT-S/16 tidak pernah kolaps
(0/3 *seed*), sedangkan Swin-T kolaps pada seluruh *seed* (3/3) dan ConvNeXt-T
pada dua dari tiga *seed*. Kegagalan ini bukan artefak *learning rate*—resep yang
sama berhasil menstabilkan jalur *pretrained* keduanya. Dengan demikian,
arsitektur hierarkis modern memiliki lanskap optimisasi yang jauh lebih sulit
saat dilatih dari nol pada dataset writer-ID berskala terbatas, dan secara
praktis **menuntut pra-pelatihan**.

## 5. Kontribusi metodologis: LR warmup untuk fine-tuning

Ditemukan bahwa *fine-tuning* ConvNeXt-T dan Swin-T pada dataset ini menuntut
*learning-rate warmup*. Tanpa *warmup*, kedua arsitektur divergen pada epoch awal
dan kolaps ke prediksi satu kelas di seluruh level dan *seed*; penambahan *warmup*
linear 3 epoch menyelamatkan seluruh *run* tersebut. Arsitektur lain (ResNet-50,
EfficientNetV2-S, ViT-S/16) stabil dengan maupun tanpa *warmup*. Temuan praktis
ini penting untuk direplikasi pada studi sejenis.

## 6. Keterbatasan dan saran penelitian lanjutan

- **Skala dataset.** Kesimpulan mengenai kegagalan pelatihan dari *scratch*
  terikat pada skala CVL; pada dataset yang jauh lebih besar, ConvNeXt/Swin
  kemungkinan dapat dilatih dari nol. Pengujian pada dataset writer-ID yang lebih
  besar (mis. IAM, Firemaker) dapat memperkuat generalisasi temuan.
- **Ragam seed.** Evaluasi menggunakan tiga *seed*; penambahan jumlah *seed* akan
  memperketat estimasi varians, khususnya untuk kasus *seed-lottery* pada
  ConvNeXt-T.
- **Augmentasi dan regularisasi.** Studi lanjutan dapat menguji apakah augmentasi
  agresif atau teknik regularisasi tertentu mampu menstabilkan pelatihan
  ConvNeXt/Swin dari *scratch*.
- **Level ablasi `full`.** Karena ukuran datanya praktis identik dengan L4,
  level `full` dilepas dari pelaporan ablasi; desain penelitian lanjutan dapat
  memilih titik-titik ukuran data yang berjarak lebih merata.
