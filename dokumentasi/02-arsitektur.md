# 02 — Lima arsitektur yang dibandingkan

## Daftar

Semua model dibuat lewat `timm.create_model` (`src/cvl/models.py`). Nama kunci di
kiri dipakai di seluruh CSV dan konfigurasi; nama timm di kanan adalah bobot yang
sesungguhnya diambil.

| Kunci | Model timm | Parameter | Keluarga | Tahun |
|---|---|---|---|---|
| `resnet50` | `resnet50` | 24.139.124 | CNN residual | 2015 |
| `efficientnetv2_s` | `tf_efficientnetv2_s` | 20.572.036 | CNN ter-skala | 2021 |
| `vit_small` | `vit_small_patch16_224` | 21.784.244 | Transformer isotropik | 2020 |
| `swin_tiny` | `swin_tiny_patch4_window7_224` | 27.756.206 | Transformer hierarkis | 2021 |
| `convnext_tiny` | `convnext_tiny` | 28.056.980 | CNN modern | 2022 |

Jumlah parameter diambil dari kolom `n_params` di CSV hasil (sudah termasuk
kepala klasifikasi 308 kelas), bukan dari angka publikasi.

## Mengapa kelima ini

Pemilihannya bukan daftar "model populer", melainkan **empat sumbu desain** yang
saling melengkapi, dengan ukuran yang sebanding (20–28 juta parameter) supaya
perbandingannya tidak dimenangkan oleh kapasitas semata:

- **ResNet-50** — garis dasar klasik. Hampir semua literatur writer-ID modern
  membandingkan diri terhadapnya, jadi tanpa ResNet hasil ini sulit ditempatkan
  dalam konteks.
- **EfficientNetV2-S** — CNN yang dioptimasi lewat pencarian arsitektur dan
  penskalaan majemuk. Mewakili "CNN yang disetel habis" sebagai lawan CNN yang
  dirancang tangan.
- **ViT-S/16** — transformer isotropik murni: tidak ada hierarki resolusi, tidak
  ada bias induktif lokal. Mewakili ujung ekstrem "semua dipelajari dari data".
- **Swin-T** — transformer dengan jendela bergeser dan hierarki resolusi. Ia
  mengembalikan bias induktif lokal ke dalam transformer, jadi selisihnya
  terhadap ViT-S mengisolasi nilai hierarki tersebut.
- **ConvNeXt-T** — CNN yang dimodernisasi dengan meniru resep transformer
  (kernel besar, LayerNorm, GELU). Ia melengkapi Swin dari arah berlawanan:
  kalau Swin adalah transformer yang meminjam sifat CNN, ConvNeXt adalah CNN
  yang meminjam sifat transformer.

Dua pasangan itu — ViT vs Swin, dan Swin vs ConvNeXt — adalah yang membuat
perbandingan ini bisa menjawab pertanyaan desain, bukan sekadar membuat
peringkat.

## Kepala klasifikasi

Bawaannya `head="linear"`: `timm.create_model(..., num_classes=308)`, yaitu satu
lapis linear di atas fitur ter-pool.

Skenario `FT4` menggantinya dengan **ArcFace** (`src/cvl/arcface.py`): backbone
dibuat dengan `num_classes=0` lalu disambung `ArcFaceHead` dengan `s=30.0` dan
`m=0.3` — nilai standar dari papernya, dipatok di muka dan **tidak disetel
setelah melihat hasil**. Margin dinaikkan linear dari 0 ke 0,3 sepanjang epoch
warmup; tanpa itu ArcFace sering gagal konvergen karena head-nya diinisialisasi
acak sementara margin sudah penuh.

## Dua detail implementasi yang memengaruhi hasil

**`drop_path_rate` hanya diteruskan bila diminta.** Beberapa arsitektur timm
(termasuk `swin_tiny`) punya `drop_path_rate` bukan-nol bawaan di `__init__`
sendiri. Meneruskan `0.0` secara eksplisit akan **mematikan stochastic depth
bawaannya secara diam-diam**, sehingga baseline tidak lagi sama dengan
konfigurasi rujukan timm. Karena itu `build_model` hanya mengirim argumen itu
saat `drop_path` bukan nol.

**Ekstraksi fitur diseragamkan.** `forward_features` mengembalikan vektor
`[B, D]` untuk semua arsitektur: model ArcFace lewat jalurnya sendiri, sisanya
lewat `forward_head(..., pre_logits=True)`. Ini penting karena metrik retrieval
(`map_line`, `top1_retrieval`) menghitung kemiripan kosinus antar fitur — tanpa
penyeragaman, ViT (token) dan CNN (peta spasial) tidak akan sebanding.

## Kecepatan inferensi

`throughput_img_s` diukur saat evaluasi, jadi hanya bergantung pada arsitektur
dan kartu grafis — bukan pada mode latih. Ambil angkanya **hanya dari server 2**;
kolom yang sama di hasil server 1 diabaikan. Pada RTX PRO 4000 Blackwell,
kelimanya berada di kisaran 127–184 citra/detik, dengan selisih antar-arsitektur
yang lebih kecil daripada variasi antar-run pada arsitektur yang sama. Untuk
biaya latih yang bebas hardware, pakai `epochs_ran`.
