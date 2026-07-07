# Analisis Dataset CVL-Database-1-1

> Dokumen ini merangkum hasil eksplorasi dataset CVL Handwriting Database v1.1 yang berada di `cvl-database-1-1/`.
> Dibuat: 2026-07-07

## Apa itu dataset ini?

**CVL Handwriting Database** (Vienna University of Technology, ICDAR 2013) — dataset tulisan tangan *offline* untuk riset **Writer Identification, Writer Retrieval, dan Word Spotting**. Berisi 310 penulis yang menyalin teks cetak dalam bahasa **Inggris (6 teks) + Jerman (1 teks)**.

**Sitasi wajib:** Kleber, Fiel, Diem & Sablatnig, *"CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting"*, ICDAR 2013, pp. 560–564.

**Lisensi:** CC BY-NC 3.0 (non-komersial saja).

## Struktur & Volume

| Level | Trainset | Testset | Total |
|---|---|---|---|
| **Penulis (writers)** | 27 | 283 | **310** |
| **Halaman (pages)** | 189 | 1.415 | **1.604** |
| **Baris (lines)** | 1.624 | 11.849 | **13.473** |
| **Kata (words)** | 12.147 | 87.757 | **99.904** |
| **XML (anotasi)** | 189 | 1.415 | **1.604** |

- **116.590 file** total (~114.981 `.tif`, 1.604 `.xml`, 1 `readme.txt`).
- Setiap split punya 4 folder paralel:
  - `pages/` — gambar halaman penuh
  - `lines/` — potongan per baris (dalam subfolder per penulis)
  - `words/` — potongan per kata (subfolder per penulis)
  - `xml/` — anotasi format PAGE (PRImA schema 2010), encoding UTF-16

## Properti Gambar

- Format **TIFF**, RGB 8-bit.
- Halaman penuh ± **2480×3507 px** (A4 scan; `sips` melaporkan 2499×3520 pada sampel).
- Potongan kata jauh lebih kecil (mis. 202×77 px).

## Konvensi Penamaan (kunci untuk parsing)

```
words:  {writer}-{page}-{line}-{wordIdx}-{transcription}.tif
        0001-1-0-0-Imagine.tif  → penulis 0001, hal 1, baris 0, kata 0, teks "Imagine"
lines:  {writer}-{page}-{line}.tif
        0050-8-4.tif            → penulis 0050, hal 8, baris 4
```

✅ **Kelebihan besar:** ground-truth transkripsi tertanam langsung di nama file kata — **0 file malformed** dari 99.904 kata.

## Temuan Statistik

**Pembagian train/test dirancang untuk Writer Identification (writer-independent):**
- **Writer overlap train↔test = 0** → penulis di train sepenuhnya terpisah dari test (protokol benar).
- Trainset: 27 penulis "power" yang menulis **7 halaman** masing-masing (seragam, ~450 kata/penulis).
- Testset: 283 penulis, mayoritas **5 halaman** (~310 kata/penulis).

**Distribusi & kualitas:**
- Kata/penulis — train: 446–453 (sangat seragam); test: 90–317.
- Panjang kata rata-rata 4,6–4,7 karakter (maks 28).
- Vocabulary kecil & terkontrol: **312 token unik** (train), 302 (test), 224 token dipakai di kedua split.
- Token teratas: `the, of, and, a, on, my, or, in, is, it` + kata Jerman `dann, du, will` (dari teks Goethe/Wikipedia).

## ⚠️ Anomali yang perlu diwaspadai

1. **Writer 0161** (testset) — hanya **1 halaman / 90 kata** (bukan 5). Outlier — pertimbangkan exclude untuk evaluasi seimbang.
2. **Writer 0431** — `readme.txt` menyatakan writer ini "dihapus untuk evaluasi Writer Identification" dan 2 halamannya kosong, **namun file-nya masih ada** di testset (3 halaman, 187 kata). Wajib di-exclude sesuai protokol resmi paper.
3. Encoding `readme.txt` berisi karakter non-ASCII rusak (`Mailüfterl`, `Tragödie`) — masalah kosmetik.
4. XML ber-encoding **UTF-16**, format **PAGE (PRImA 2010)** dengan region berlapis: halaman → paragraf → baris → kata (`AttrRegion` + `minAreaRect`, atribut `fontAngleRad`, `fontSize`, `medianWordHeight`).

## Kesesuaian untuk Thesis (HTR Personalization — TrOCR + QLoRA + HITL)

**Relevan & cocok**, karena:
- **Terorganisir per-penulis** (310 writers) → ideal untuk personalisasi: satu adapter QLoRA per penulis.
- **Punya transkripsi ground-truth** → wajib untuk melatih & mengukur CER/WER.
- **Ada gambar level baris** (13.473 baris) → TrOCR umumnya bekerja pada baris/kata.

**Keterbatasan:**
- **Vocabulary sempit** (~312 kata unik, teks disalin dari sumber tetap) — bagus untuk personalisasi *gaya visual* (konten terkontrol), lemah untuk uji generalisasi bahasa.
- Mayoritas **Inggris + 1 teks Jerman** — cocok dengan TrOCR base (English).
- Untuk skenario **human-in-the-loop inkremental**, aliran koreksi harus *disimulasikan* (CVL tidak menyediakan urutan temporal).

**Rekomendasi:** pertimbangkan **IAM Handwriting Database** sebagai benchmark HTR utama (vocabulary luas, standar di literatur), dengan **CVL** untuk eksperimen personalisasi per-writer.
