# Abstrak

*Writer identification* pada tulisan tangan merupakan tugas penting dalam
analisis dokumen, forensik, dan kearsipan. Meskipun arsitektur *deep learning*
modern (Transformer visual dan CNN generasi baru) telah menggeser ResNet sebagai
tulang punggung standar di banyak domain, keunggulannya untuk *writer
identification* pada dataset berskala terbatas belum terpetakan secara sistematis.
Penelitian ini membandingkan lima arsitektur—ResNet-50, EfficientNetV2-S,
ViT-S/16, Swin-T, dan ConvNeXt-T—pada CVL Database (308 penulis, 13.440 citra
baris), di bawah dua skenario pelatihan (*pretrained* dan *scratch*), empat
tingkat ukuran data latih, dan tiga *seed*.

Hasil menunjukkan bahwa dengan *transfer learning*, seluruh arsitektur mencapai
akurasi tinggi (Top-1 0,94–0,98 pada data terbanyak); arsitektur modern
mengungguli baseline ResNet-50 secara konsisten namun moderat (~4 poin Top-1),
dengan Swin-T dan ConvNeXt-T tertinggi (rerata Top-1 0,904) sementara
EfficientNetV2-S menawarkan efisiensi parameter terbaik. Temuan utama kedua:
pada pelatihan dari *scratch* dengan resep identik, Swin-T dan ConvNeXt-T gagal
konvergen (kolaps pada 3/3 dan 2/3 *seed*), sedangkan CNN dan ViT tetap stabil—
menunjukkan bahwa arsitektur hierarkis modern menuntut pra-pelatihan pada skala
dataset ini. Sebagai kontribusi metodologis, ditemukan bahwa *learning-rate
warmup* wajib untuk menstabilkan *fine-tuning* ConvNeXt-T dan Swin-T.

**Kata kunci:** writer identification, CVL Database, perbandingan arsitektur,
transfer learning, Vision Transformer, ConvNeXt, learning-rate warmup.
