# Lampiran C — Kode Program Inti

Lampiran ini memuat kode yang melaksanakan metode penelitian: penyiapan data,
pembangunan model, pelatihan, evaluasi, dan skenario fine-tuning. Setiap berkas
didahului penjelasan singkat mengenai perannya dan keputusan teknis yang
memengaruhi hasil.

Kode pendukung yang tidak memuat metode — orkestrasi grid eksperimen, pembuatan
laporan dan gambar, serta 22 berkas pengujian otomatis — tidak dilampirkan dan
tersedia pada repositori kode penelitian.

## config.py — Konstanta kohor dan katalog arsitektur

Dilampirkan baris 1–24.

Memusatkan konstanta yang tidak berubah antar-run. Aturan kohor ada di sini:
penulis `0431` dan `0161` dikeluarkan, dan penulis dengan kurang dari lima
halaman dibuang — dua aturan inilah yang membentuk kohor 308 penulis.
`ALL_ARCHITECTURES` memetakan nama pendek yang dipakai di seluruh laporan ke nama
model timm yang sebenarnya diunduh. Sisa berkas hanya membaca `.env` untuk
memilih subset grid dan tidak memengaruhi hasil.

```python
import os
from datetime import datetime
from pathlib import Path

EXCLUDE_WRITERS = {"0431", "0161"}
MIN_PAGES = 5
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Katalog lengkap semua arsitektur (jangan diubah — subset diatur lewat .env)
ALL_ARCHITECTURES = {
    "resnet50": "resnet50",
    "convnext_tiny": "convnext_tiny",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "vit_small": "vit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}
# Level `full` (None) di-drop dari grid: ukuran datanya ≈ L4 (9.852 vs 9.455
# baris, beda ~4%) dan akurasinya setara, jadi redundan. Parser `.env` masih
# menerima token "full" bila sewaktu-waktu perlu dijalankan lagi.
ALL_ABLATION_LEVELS = [1, 2, 3, 4]
ALL_SEEDS = [0, 1, 2, 3, 4]
ALL_MODES = ["pretrained", "scratch"]
```

## data_prep.py — Pemindaian dataset dan pembentukan manifest

Mengubah folder mentah CVL menjadi satu tabel manifest yang menentukan citra
mana masuk *train*, *val*, atau *test*. Label penulis dibaca langsung dari pola
nama berkas `<penulis>-<halaman>-<baris>.tif`, tanpa berkas anotasi terpisah.

`build_manifest` membentuk ablasi data: halaman terakhir tiap penulis **selalu**
menjadi data uji, lalu 1–4 halaman diambil acak dari sisanya sebagai data latih
(level L1–L4). Karena set uji sama pada semua level, selisih skor antar-level
murni berasal dari jumlah data latih.

Validasi diambil per baris, bukan per halaman. Pada L1 setiap penulis hanya punya
satu halaman latih, sehingga baris validasi berasal dari halaman yang sama dengan
data latih — karena itu `best_val_acc` hanya dipakai untuk memilih *checkpoint*
dan tidak dilaporkan sebagai hasil.

```python
from pathlib import Path
import re
import pandas as pd
import numpy as np
from .config import EXCLUDE_WRITERS, MIN_PAGES

_LINE_RE = re.compile(r"^(\d+)-(\d+)-(\d+)\.tif$", re.IGNORECASE)

def parse_line_filename(name: str) -> tuple[str, str, int]:
    m = _LINE_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"nama file baris tak valid: {name}")
    writer, page, line = m.group(1), m.group(2), int(m.group(3))
    return writer, page, line

def scan_lines(root: Path) -> pd.DataFrame:
    rows = []
    for tif in Path(root).rglob("*.tif"):
        if "lines" not in {p.name for p in tif.parents}:
            continue  # hanya file di bawah folder lines/
        try:
            writer, page, line = parse_line_filename(tif.name)
        except ValueError:
            continue
        rows.append((writer, page, line, str(tif.resolve())))
    return pd.DataFrame(rows, columns=["writer", "page", "line", "path"])

def filter_cohort(df, min_pages: int = MIN_PAGES, exclude: set = EXCLUDE_WRITERS):
    df = df[~df["writer"].isin(exclude)].copy()
    pages_per_writer = df.groupby("writer")["page"].nunique()
    keep = pages_per_writer[pages_per_writer >= min_pages].index
    dropped = sorted(set(pages_per_writer.index) - set(keep))
    kept = df[df["writer"].isin(keep)].copy()
    info = {
        "n_excluded_rule": len(dropped),
        "n_kept_writers": len(keep),
        "dropped_writers": dropped,
    }
    return kept, info

def build_label_map(df) -> dict:
    return {w: i for i, w in enumerate(sorted(df["writer"].unique()))}

def _page_sort_key(p: str):
    return (int(p) if p.isdigit() else p)

def build_manifest(df, n_train_pages, seed: int, test_pages: int = 1, val_frac: float = 0.1):
    label_map = build_label_map(df)
    rng = np.random.default_rng(seed)
    parts = []
    for writer, g in df.groupby("writer"):
        pages = sorted(g["page"].unique(), key=_page_sort_key)
        test_p = set(pages[-test_pages:])
        pool = [p for p in pages if p not in test_p]
        if n_train_pages is not None:
            chosen = list(rng.permutation(pool))[:n_train_pages]
        else:
            chosen = pool
        g = g.copy()
        g["label"] = label_map[writer]
        g["split"] = "unused"
        g.loc[g["page"].isin(test_p), "split"] = "test"
        train_mask = g["page"].isin(chosen)
        train_lines = g[train_mask].sort_values(["page", "line"])
        n_val = max(1, int(round(len(train_lines) * val_frac))) if len(train_lines) > 1 else 0
        val_idx = set(rng.permutation(train_lines.index)[:n_val].tolist())
        g.loc[train_mask, "split"] = "train"
        g.loc[g.index.isin(val_idx), "split"] = "val"
        parts.append(g[g["split"] != "unused"])
    return pd.concat(parts, ignore_index=True)
```

## dataset.py — Transformasi citra dan pemuat data

Menentukan bagian citra mana yang dilihat model. Dua sumbu sengaja dipisah:
`geometry` mengatur *bagian mana* dari baris yang diambil, `aug` mengatur
*seberapa keras* citra diacak.

Pada geometri `center` (dipakai seluruh Studi 1), `T.Resize(224)` menyetel sisi
**pendek**; karena citra baris CVL berasio sekitar 12:1, hasilnya kira-kira
3.284×224 piksel dan potongan tengah 224×224 hanya mencakup **sekitar 7,5%
panjang baris**. Geometri `linewindow` (skenario FT1) menyetel *tinggi* ke 224
lalu mengambil jendela acak saat latih dan sembilan jendela merata saat uji.
Selisih cakupan inilah yang menjelaskan lompatan akurasi FT1.

```python
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class ResizeHeight:
    """Setel tinggi ke `height`, jaga rasio aspek.

    Berbeda dari `T.Resize(224)` yang menyetel sisi *pendek*: pada citra baris
    CVL (rasio ~12:1) keduanya kebetulan sama, tapi ResizeHeight tetap benar
    untuk baris yang tidak wajar (lebih tinggi daripada lebar) karena hasilnya
    dijamin tidak lebih sempit dari `height`.
    """

    def __init__(self, height: int):
        self.height = height

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w = max(self.height, int(round(w * self.height / h)))
        return img.resize((new_w, self.height), Image.BILINEAR)


def even_windows(img: Image.Image, size: int, k: int) -> list:
    """`k` jendela selebar `size` yang tersebar merata sepanjang citra.

    Dipakai saat evaluasi FT1: satu potongan tengah hanya mewakili 7,5% baris,
    jadi prediksi dirata-ratakan atas beberapa posisi.
    """
    w, h = img.size
    if w <= size:
        return [img] * k
    step = (w - size) / max(1, k - 1) if k > 1 else 0
    return [img.crop((int(round(i * step)), 0, int(round(i * step)) + size, h))
            for i in range(k)]


def _geometry_stage(train: bool, image_size: int, geometry: str):
    """Tahap yang menentukan *bagian mana* dari baris yang dilihat model."""
    if geometry == "center":
        # Perilaku lama. T.Resize(int) menyetel sisi pendek; pada baris 12:1
        # hasilnya ~3284x224 dan crop 224 mengambil bagian tengah saja.
        if train:
            return [T.Resize(image_size)]
        return [T.Resize(image_size), T.CenterCrop(image_size)]
    if geometry == "linewindow":
        # Tinggi dipaskan ke image_size, lalu jendela selebar image_size
        # diambil acak (latih) atau di tengah (uji, saat eval_crops=1).
        if train:
            return [ResizeHeight(image_size), T.RandomCrop(image_size)]
        return [ResizeHeight(image_size), T.CenterCrop(image_size)]
    raise ValueError(f"geometry tidak dikenal: {geometry}")


def _strip_width(image_size: int) -> int:
    """Lebar strip tengah yang dilihat pipeline baseline.

    Torchvision jatuh ke fallback `w = round(h * max(ratio))` ketika batasan
    RandomResizedCrop tidak terpenuhi. Dengan ratio maks 1.1 dan tinggi 224,
    lebarnya 246 — angka ini yang dipakai AUG supaya cakupan barisnya sama
    persis dengan baseline.
    """
    return int(round(image_size * 1.1))


def _aug_stage(geometry: str, image_size: int, aug: str):
    """Tahap augmentasi (hanya dipakai saat train=True)."""
    if aug == "baseline":
        if geometry == "center":
            return [
                T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        # linewindow: jendela sudah 224x224, affine + jitter saja
        return [
            T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    if aug == "strong":
        steps = [
            T.RandomAffine(degrees=6, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        ]
        if geometry == "center":
            # Potong dulu ke strip tengah 224x246 — wilayah yang sama dengan
            # baseline — supaya RandomResizedCrop punya ruang untuk memenuhi
            # batasan rasio dan benar-benar mengacak.
            steps.append(T.CenterCrop((image_size, _strip_width(image_size))))
        steps += [
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
            T.ColorJitter(brightness=0.4, contrast=0.4),
        ]
        return steps
    raise ValueError(f"aug tidak dikenal: {aug}")


def build_transforms(train: bool, image_size: int = IMAGE_SIZE,
                     geometry: str = "center", aug: str = "baseline"):
    """PIL -> Tensor [3, image_size, image_size].

    `geometry` mengatur bagian mana dari baris yang terlihat; `aug` mengatur
    seberapa keras citra diacak. Dua sumbu ini sengaja dipisah agar skenario
    FT1 dan AUG menguji mekanisme yang berbeda tanpa saling mencemari.
    """
    # Validasi tanpa syarat: nilai `aug` salah ketik harus ditolak baik saat
    # train maupun eval, bukan hanya saat _aug_stage benar-benar dipanggil.
    if aug not in ("baseline", "strong"):
        raise ValueError(f"aug tidak dikenal: {aug}")
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    steps = [T.Grayscale(num_output_channels=3)]
    steps += _geometry_stage(train, image_size, geometry)
    if train:
        steps += _aug_stage(geometry, image_size, aug)
    steps += [T.ToTensor()]
    if train and aug == "strong":
        # RandomErasing bekerja pada tensor, bukan PIL, jadi harus setelah
        # ToTensor.
        steps.append(T.RandomErasing(p=0.25))
    steps += [norm]
    return T.Compose(steps)


class LineDataset(Dataset):
    def __init__(self, manifest_subset, train: bool,
                 geometry: str = "center", aug: str = "baseline",
                 eval_crops: int = 1):
        self.rows = manifest_subset.reset_index(drop=True)
        self.train = train
        self.geometry = geometry
        self.eval_crops = eval_crops
        self.tf = build_transforms(train, geometry=geometry, aug=aug)
        # transform untuk satu jendela yang sudah dipotong (dipakai saat eval_crops > 1)
        self.crop_tf = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        label = int(r["label"])
        if self.train or self.eval_crops <= 1:
            return self.tf(img), label
        base = ResizeHeight(IMAGE_SIZE)(img.convert("RGB"))
        wins = even_windows(base, IMAGE_SIZE, self.eval_crops)
        return torch.stack([self.crop_tf(w) for w in wins]), label


def loader_kwargs(hp: dict, device: str) -> dict:
    """Argumen DataLoader yang seragam untuk latih dan evaluasi.

    `persistent_workers` penting di sini: tanpa itu PyTorch menyalakan dan
    mematikan seluruh proses pekerja dua kali tiap epoch (loader latih dan
    validasi). Pada grid lama biaya tetap itu ~3,8 detik per epoch — 38% waktu
    tiap epoch di level terkecil, karena dibagi ke sedikit citra. Semakin banyak
    worker, semakin mahal biaya nyalakan-matikan itu, jadi justru pada pod
    ber-vCPU banyak ia paling terasa.

    `prefetch_factor` dan `pin_memory` hanya sah bila ada worker / ada GPU.
    """
    nw = hp.get("num_workers", 0)
    kw = {"num_workers": nw}
    if nw > 0:
        kw["persistent_workers"] = True
        # Antrean prefetch hidup di shared memory: num_workers x prefetch_factor
        # x 39 MB per loader, dan loader latih + validasi sama-sama persistent.
        # Turunkan ke 2 pada pod ber-RAM kecil atau /dev/shm sempit.
        kw["prefetch_factor"] = hp.get("prefetch_factor", 4)
    if device != "cpu":
        kw["pin_memory"] = True
    return kw
```

## models.py — Pembangun kelima arsitektur

Satu fungsi membangun kelima arsitektur lewat timm, sehingga tidak ada
perbedaan perlakuan yang tidak disengaja antar-arsitektur. Argumen `pretrained`
adalah satu-satunya pembeda mode *scratch* dan *pretrained*.

`drop_path` hanya diteruskan bila bukan nol: sebagian arsitektur timm (misalnya
Swin-Tiny) punya `drop_path_rate` bawaan bukan-nol, dan meneruskan `0.0` secara
eksplisit akan mematikan *stochastic depth* bawaannya secara diam-diam.
`forward_features` menyeragamkan bentuk vektor ciri menjadi `[B, D]` untuk semua
arsitektur, diperlukan sebelum metrik *retrieval* dihitung.

```python
import timm
import torch
from .arcface import ArcFaceHead, ArcFaceModel
from .config import ALL_ARCHITECTURES


def build_model(arch_key: str, num_classes: int, pretrained: bool,
                drop_path: float = 0.0, head: str = "linear"):
    name = ALL_ARCHITECTURES[arch_key]
    # Hanya teruskan drop_path_rate kalau memang diminta. Sebagian arsitektur
    # timm (mis. swin_tiny) punya default drop_path_rate bukan-nol di
    # __init__ sendiri; meneruskan 0.0 secara eksplisit di sini akan
    # mematikan stochastic depth bawaannya secara diam-diam.
    extra = {"drop_path_rate": drop_path} if drop_path else {}
    if head == "linear":
        return timm.create_model(name, pretrained=pretrained,
                                 num_classes=num_classes, **extra)
    if head == "arcface":
        backbone = timm.create_model(name, pretrained=pretrained,
                                     num_classes=0, **extra)
        return ArcFaceModel(backbone, ArcFaceHead(backbone.num_features, num_classes))
    raise ValueError(f"head tidak dikenal: {head}")


def set_arcface_margin(model, m: float) -> None:
    """Setel margin bila model memakai head ArcFace; selain itu tidak apa-apa."""
    if isinstance(model, ArcFaceModel):
        model.head.set_margin(m)


def forward_features(model, x):
    if isinstance(model, ArcFaceModel):
        return model.forward_features(x)
    feats = model.forward_features(x)
    # samakan ke [B, D]: pool spatial/token via head pre_logits jika tersedia
    if hasattr(model, "forward_head"):
        pooled = model.forward_head(feats, pre_logits=True)
    else:
        pooled = feats.mean(dim=tuple(range(2, feats.dim()))) if feats.dim() > 2 else feats
    return pooled


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
```

## arcface.py — Head ArcFace untuk skenario FT4

*Head* bermargin sudut yang menggantikan lapisan klasifikasi linear pada
skenario FT4. Vektor ciri dan bobot kelas sama-sama dinormalisasi sehingga hasil
kalinya adalah kosinus sudut, lalu margin ditambahkan pada sudut kelas yang benar
saja.

Nilai `s = 30.0` dan `m = 0.3` dipatok di muka pada nilai standar makalah
aslinya dan tidak disetel setelah melihat hasil. Perhitungan `clamp` dan `acos`
dipaksa ke fp32 karena di bawah *mixed precision*, fp16 membulatkan `1 − 1e−7`
menjadi tepat `1.0` sehingga `clamp` tidak berfungsi dan gradien meledak.

```python
"""Head ArcFace (additive angular margin) untuk skenario FT4.

Writer identification pada L1 adalah 308 identitas dengan ~7 contoh per kelas
— rezim tempat head margin lazim dipakai. `s` dan `m` dipatok di muka pada
nilai standar papernya dan tidak disetel setelah melihat hasil.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int,
                 s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def set_margin(self, m: float) -> None:
        """Dipakai untuk menaikkan margin bertahap selama epoch warmup."""
        self.m = m

    def forward(self, feats, labels=None):
        cos = F.linear(F.normalize(feats), F.normalize(self.weight))
        if labels is None or self.m == 0.0:
            return self.s * cos
        # Clamp dan acos wajib di fp32: di bawah AMP, F.linear mengeluarkan
        # fp16, dan fp16 tidak bisa merepresentasikan 1.0 - 1e-7 -- ia
        # dibulatkan jadi 1.0 persis, sehingga clamp jadi no-op dan
        # acos(1.0) berada tepat di batas domainnya, meledakkan gradien
        # jadi tak-hingga. Kembalikan ke dtype semula supaya tetap sejalan
        # dengan cabang tanpa-label di atas dan dengan loss yang memakainya.
        orig_dtype = cos.dtype
        cos = cos.float().clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos)
        margin = torch.zeros_like(theta)
        margin.scatter_(1, labels.view(-1, 1), self.m)
        logits = self.s * torch.cos(theta + margin)
        return logits.to(orig_dtype)


class ArcFaceModel(nn.Module):
    """Backbone timm (num_classes=0) + head ArcFace.

    `forward` menerima `labels` opsional supaya margin hanya aktif saat latih;
    tanpa label ia mengembalikan skor kosinus terskala yang aman di-softmax
    saat evaluasi.
    """

    def __init__(self, backbone, head: ArcFaceHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, labels=None):
        return self.head(self.backbone(x), labels)

    def forward_features(self, x):
        return self.backbone(x)
```

## train.py — Loop pelatihan

Menjalankan satu run (satu kombinasi arsitektur, level, mode, dan *seed*).
Fungsi ini sama untuk Studi 1 dan Studi 2; Studi 2 hanya mengoper objek
`Scenario` yang mengubah beberapa perilaku.

Laju pembelajaran dinaikkan linear dari 1% nilai penuh selama tiga epoch
(*warmup*) lalu diturunkan mengikuti kurva kosinus. Tanpa *warmup*, ConvNeXt dan
Swin kerap divergen di epoch awal lalu kolaps ke satu kelas. Margin ArcFace juga
dinaikkan bertahap dari 0 ke 0,3 sepanjang *warmup*, dengan alasan yang sama.

*Checkpoint* yang disimpan adalah bobot dengan akurasi validasi terbaik, bukan
bobot epoch terakhir; *early stopping* memutus setelah delapan epoch berturut-
turut tanpa perbaikan.

```python
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import LineDataset, loader_kwargs
from .models import build_model, set_arcface_margin
from .finetune import freeze_layers, build_param_groups
from .scenarios import Scenario

@dataclass
class RunConfig:
    arch: str
    level: object          # int | None
    mode: str              # "pretrained" | "scratch"
    seed: int
    epochs: int
    lr: float = 3e-4
    batch_size: int = 64
    weight_decay: float = 0.05

def _seed_all(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)

def _num_classes(manifest) -> int:
    return int(manifest["label"].max()) + 1

def arcface_margin_at(epoch: int, warmup_epochs: int, m_target: float) -> float:
    """Margin ArcFace dinaikkan linear 0 -> m_target sepanjang epoch warmup.

    Tanpa ini ArcFace sering gagal konvergen di epoch awal karena head-nya
    diinisialisasi acak sementara margin sudah penuh.
    """
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return m_target
    return m_target * epoch / warmup_epochs

def train_one_run(manifest, rc: RunConfig, out_dir, device, hp: dict,
                  scenario: Scenario | None = None) -> dict:
    sc = scenario or Scenario()
    _seed_all(rc.seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = LineDataset(manifest[manifest.split == "train"], train=True,
                           geometry=sc.geometry, aug=sc.aug)
    val_ds = LineDataset(manifest[manifest.split == "val"], train=False,
                         geometry=sc.geometry, aug=sc.aug)
    tl = DataLoader(train_ds, batch_size=rc.batch_size, shuffle=True,
                    **loader_kwargs(hp, device))
    vl = DataLoader(val_ds, batch_size=rc.batch_size, shuffle=False,
                    **loader_kwargs(hp, device))
    model = build_model(rc.arch, _num_classes(manifest),
                        pretrained=(rc.mode == "pretrained"),
                        drop_path=sc.drop_path, head=sc.head).to(device)
    if sc.freeze_strategy is not None:
        freeze_layers(model, sc.freeze_strategy, arch=rc.arch)
        opt = torch.optim.AdamW(
            build_param_groups(model, sc.freeze_strategy, base_lr=rc.lr,
                               arch=rc.arch),
            weight_decay=rc.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=rc.lr,
                                weight_decay=rc.weight_decay)
    # Warmup LR linear beberapa epoch lalu cosine annealing. Tanpa warmup,
    # ConvNeXt/Swin sering divergen di epoch awal lalu kolaps ke 1 kelas.
    warmup_epochs = min(int(hp.get("warmup_epochs", 3)), max(0, rc.epochs - 1))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, rc.epochs - warmup_epochs))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup, cosine], milestones=[warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, rc.epochs))
    crit = torch.nn.CrossEntropyLoss(label_smoothing=sc.label_smoothing)
    use_amp = hp.get("amp", False) and device != "cpu"
    amp_device = "cuda" if device != "cpu" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)
    best_acc, best_state, patience, bad = -1.0, None, hp.get("early_stop_patience", 8), 0
    t0, epochs_ran = time.time(), 0
    lvl = "full" if rc.level is None else str(rc.level)
    tag = f"{rc.arch}_L{lvl}_{rc.mode}_s{rc.seed}"
    for epoch in range(rc.epochs):
        set_arcface_margin(model, arcface_margin_at(epoch, warmup_epochs, 0.3))
        model.train()
        loss_sum = n_seen = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast(amp_device, enabled=use_amp):
                logits = model(x, y) if sc.head == "arcface" else model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_sum += loss.item() * len(x); n_seen += len(x)
        sched.step(); epochs_ran += 1
        # validasi
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item(); total += len(y)
        acc = correct / max(1, total)
        improved = acc > best_acc
        if improved:
            best_acc, best_state, bad = acc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        train_loss = loss_sum / max(1, n_seen)
        print(f"  [{tag}] epoch {epoch + 1}/{rc.epochs} "
              f"loss={train_loss:.4f} val_acc={acc:.4f} "
              f"best={best_acc:.4f}{' *' if improved else ''} "
              f"patience={bad}/{patience} elapsed={time.time() - t0:.0f}s",
              flush=True)
        if not improved and bad >= patience:
            print(f"  [{tag}] early stop @ epoch {epoch + 1} (best_val_acc={best_acc:.4f})", flush=True)
            break
    torch.save(best_state or model.state_dict(), out_dir / "best.pt")
    return {"best_val_acc": float(best_acc), "train_time_s": time.time() - t0, "epochs_ran": epochs_ran}
```

## evaluate.py — Evaluasi tingkat halaman

Memuat *checkpoint* terbaik dan menghitung seluruh metrik pada data uji.
Prediksi dihitung per baris, lalu baris yang berasal dari halaman yang sama
dikelompokkan dan probabilitasnya dirata-ratakan sebelum diambil kelas
tertingginya — sehingga `top1_page` mengukur 308 keputusan tingkat halaman, bukan
2.490 keputusan tingkat baris.

Pada skenario FT1 satu baris datang sebagai sembilan jendela sekaligus.
Softmax dilakukan lebih dulu, baru dirata-ratakan; urutan sebaliknya menghasilkan
besaran yang berbeda karena ketaksamaan Jensen. *Throughput* dihitung dalam
satuan baris per detik agar sebanding antara skenario satu-jendela dan
sembilan-jendela.

```python
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import LineDataset, loader_kwargs
from .models import build_model, forward_features, count_params
from .metrics import aggregate_by_group, top_k_accuracy, macro_f1, retrieval_map

def _num_classes(manifest) -> int:
    return int(manifest["label"].max()) + 1

def rata_rata_jendela(logits, b: int, k: int):
    """Rata-ratakan prediksi K jendela per baris.

    Urutannya menentukan: softmax dulu, baru dirata-rata. Merata-ratakan logit
    lalu men-softmax menghasilkan besaran yang berbeda (ketaksamaan Jensen) dan
    akan mengubah setiap metrik multi-crop tanpa satu test pun gagal.
    """
    return torch.softmax(logits, dim=1).reshape(b, k, -1).mean(dim=1)

def evaluate_checkpoint(ckpt_path, manifest, arch, device, batch_size: int = 64,
                        scenario=None, num_workers: int = 0) -> dict:
    """Evaluasi checkpoint pada set uji.

    `num_workers` hanya mempercepat pemuatan data; transform evaluasi sepenuhnya
    deterministik (tanpa augmentasi acak), jadi jumlah worker tidak mengubah
    hasil sedikit pun — cuma waktu. Default 0 mempertahankan perilaku lama untuk
    pemanggil yang belum meneruskannya.
    """
    from .scenarios import Scenario
    sc = scenario or Scenario()
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    model = build_model(arch, _num_classes(manifest), pretrained=False,
                        drop_path=sc.drop_path, head=sc.head).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    ds = LineDataset(test, train=False, geometry=sc.geometry, aug=sc.aug,
                     eval_crops=sc.eval_crops)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    **loader_kwargs({"num_workers": num_workers}, device))
    probs, feats = [], []
    t0, n_img = time.time(), 0
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            if x.dim() == 5:
                # [B, K, 3, H, W] -> forward semua jendela, rata-ratakan per baris
                b, k = x.shape[0], x.shape[1]
                flat = x.reshape(b * k, *x.shape[2:])
                # Satuan throughput adalah baris/detik, bukan jendela/detik --
                # jangan diganti ke flat.shape[0], nanti angkanya menipu.
                n_img += x.shape[0]
                p = rata_rata_jendela(model(flat), b, k)
                f = forward_features(model, flat).reshape(b, k, -1).mean(dim=1)
            else:
                n_img += len(x)
                p = torch.softmax(model(x), dim=1)
                f = forward_features(model, x)
            probs.append(p.cpu().numpy())
            feats.append(f.cpu().numpy())
    throughput = n_img / max(1e-6, time.time() - t0)
    probs = np.concatenate(probs); feats = np.concatenate(feats)
    labels = test["label"].to_numpy()
    page_groups = (test["writer"] + "|" + test["page"]).to_numpy()
    gids, page_probs = aggregate_by_group(probs, page_groups)
    page_labels = np.array([test[page_groups == g]["label"].iloc[0] for g in gids])
    map_line, top1_retrieval = retrieval_map(feats, labels)
    return {
        "top1_page": top_k_accuracy(page_probs, page_labels, 1),
        "top5_page": top_k_accuracy(page_probs, page_labels, min(5, page_probs.shape[1])),
        "macro_f1_page": macro_f1(page_probs, page_labels),
        "map_line": map_line,
        "top1_retrieval": top1_retrieval,
        "n_params": count_params(model),
        "throughput_img_s": throughput,
    }
```

## metrics.py — Definisi metrik

Definisi eksplisit setiap angka yang dilaporkan. `top_k_accuracy` menghasilkan
`top1_page` dan `top5_page`; `macro_f1` merata-ratakan F1 per kelas tanpa
pembobotan jumlah contoh; `retrieval_map` mengukur kualitas ruang ciri tanpa
melibatkan lapisan klasifikasi, lewat kemiripan kosinus antar-baris uji, dan
menghasilkan `map_line` serta `top1_retrieval`.

Dua metrik terakhir bekerja pada 2.490 baris uji sedangkan `top1_page` bekerja
pada 308 halaman, sehingga angkanya tidak dapat diperbandingkan satu sama
lain.

```python
import numpy as np
from sklearn.metrics import f1_score

def aggregate_by_group(probs, groups):
    gids = np.unique(groups)
    mp = np.stack([probs[groups == g].mean(axis=0) for g in gids])
    return gids, mp

def top_k_accuracy(probs, labels, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    hits = [labels[i] in topk[i] for i in range(len(labels))]
    return float(np.mean(hits))

def macro_f1(probs, labels) -> float:
    preds = probs.argmax(axis=1)
    return float(f1_score(labels, preds, average="macro"))

def retrieval_map(features, labels):
    f = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sim = f @ f.T
    np.fill_diagonal(sim, -np.inf)  # buang diri sendiri
    n = len(labels)
    aps, top1 = [], []
    for i in range(n):
        order = np.argsort(-sim[i])
        order = order[:-1]  # exclude self (at last position due to -inf)
        rel = (labels[order] == labels[i]).astype(int)
        if len(rel) > 0:
            top1.append(rel[0])
        if rel.sum() == 0:
            continue
        cum = np.cumsum(rel)
        precision_at_hits = cum[rel == 1] / (np.where(rel == 1)[0] + 1)
        aps.append(precision_at_hits.mean())
    return float(np.mean(aps)) if aps else 0.0, float(np.mean(top1)) if top1 else 0.0
```

## scenarios.py — Registry skenario Studi 2

Enam konfigurasi Studi 2 didefinisikan sebagai data, satu baris per skenario,
dan masing-masing mengubah **tepat satu mekanisme** dibanding dasarnya. Karena
itu selisih skornya dapat langsung dibaca sebagai efek mekanisme tersebut.
`Scenario()` tanpa argumen adalah pipeline apa adanya, sehingga FT0 tidak perlu
dijalankan ulang dan barisnya disalin dari hasil Studi 1.

| Skenario | Yang diubah | Mekanisme yang diuji |
|---|---|---|
| FT0 | (tidak ada) | dasar pembanding |
| FT1 | `geometry="linewindow"`, `eval_crops=9` | cakupan baris |
| FT2 | `drop_path=0.2`, `label_smoothing=0.1` | regularisasi bawaan arsitektur |
| FT3 | `freeze_strategy="S3"` | pembekuan lapisan + LLRD |
| FT4 | `head="arcface"` | *head* bermargin sudut |
| AUG | `aug="strong"` | augmentasi kuat |

```python
"""Registry skenario Studi 2 (fine-tuning ConvNeXt-Tiny di L1).

Satu skenario = satu mekanisme yang diubah. `Scenario()` tanpa argumen
adalah pipeline apa adanya, sehingga FT0 tidak perlu dijalankan ulang dan
barisnya bisa disalin dari hasil grid utama.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    geometry: str = "center"            # "center" | "linewindow"
    aug: str = "baseline"               # "baseline" | "strong"
    drop_path: float = 0.0
    label_smoothing: float = 0.0
    freeze_strategy: str | None = None  # kunci di finetune.STRATEGIES
    head: str = "linear"                # "linear" | "arcface"
    eval_crops: int = 1


SCENARIOS: dict[str, Scenario] = {
    # baseline: diambil dari results-pretrained.csv, tidak dijalankan ulang
    "FT0": Scenario(),
    # cakupan baris: jendela acak saat latih, 9 jendela dirata-rata saat uji
    "FT1": Scenario(geometry="linewindow", eval_crops=9),
    # regularisasi bawaan ConvNeXt
    "FT2": Scenario(drop_path=0.2, label_smoothing=0.1),
    # transfer learning klasik (S3 = beku stem+stages.0-1, LLRD 0.7)
    "FT3": Scenario(freeze_strategy="S3"),
    # head margin sudut
    "FT4": Scenario(head="arcface"),
    # augmentasi kuat
    "AUG": Scenario(aug="strong"),
}


def scenario_run_id(name: str, seed: int, arch: str, level) -> str:
    """Identitas satu run Studi 2.

    Mengikuti pola grid utama ({arch}_L{level}_{mode}_s{seed}) dengan nama
    skenario di posisi mode. Arsitektur wajib ikut: tanpa itu dua arsitektur
    yang menulis ke CSV yang sama membuat `already_done` melewati run kedua
    sebagai "sudah selesai", dan folder checkpoint-nya saling menimpa.
    """
    return f"{arch}_L{level}_{name}_s{seed}"
```

## finetune.py — Strategi pembekuan lapisan dan LLRD

Melaksanakan skenario FT3. Mendefinisikan lima strategi *transfer learning* di
sepanjang satu sumbu: berapa lapisan awal yang dibekukan, dan bagaimana laju
pembelajaran dibagi ke lapisan yang masih dilatih.

| Strategi | Dibekukan | Laju pembelajaran |
|---|---|---|
| S0 | tidak ada | seragam (perilaku Studi 1) |
| S1 | seluruh *backbone* | seragam, hanya *head* dilatih |
| S2 | stem + 2 stage pertama | seragam 1e-4 |
| S3 | stem + 2 stage pertama | LLRD: *head* 1e-4, tiap tingkat ke bawah ×0,7 |
| S4 | sama seperti S3 | S3 + *label smoothing* 0,1 |

Strategi dinyatakan sebagai **kedalaman**, bukan nama modul harfiah. timm menamai
pohon modul tiap keluarga arsitektur berbeda (ConvNeXt: `stem`/`stages.N`, Swin:
`patch_embed`/`layers.N`), dan nama yang tidak cocok akan membekukan nol parameter
serta meruntuhkan LLRD menjadi satu grup **tanpa memunculkan galat** — run tetap
selesai dengan strategi yang sebenarnya tidak aktif. Karena itu `LAYER_MAP`
memetakan nama konkret per arsitektur dan `_cek_cocok` menghentikan program bila
ada prefiks yang tidak cocok.

```python
# -*- coding: utf-8 -*-
"""
finetune.py — Strategi fine-tuning (freeze + LLRD) untuk Studi 2.
=================================================================================
Dipakai lewat `Scenario.freeze_strategy` (lihat scenarios.py); train.py yang
memanggil freeze_layers/build_param_groups dan mengoper `rc.arch`. Tidak ada
env-var: strategi ditentukan oleh skenario, arsitektur oleh RunConfig.

Strategi:
  S0  baseline          : fine-tune penuh, LR seragam (perilaku grid utama)
  S1  feature-extract   : bekukan SELURUH backbone, latih head saja
  S2  selective         : bekukan stem + 2 stage pertama, LR 1e-4
  S3  selective + LLRD  : seperti S2 + layer-wise LR decay (head 1e-4, x0.7/stage)
  S4  S3 + smoothing    : S3 + label_smoothing=0.1

Strategi dinyatakan dalam *kedalaman* (berapa stage dibekukan), bukan nama
modul harfiah, supaya berlaku untuk lebih dari satu keluarga arsitektur —
lihat LAYER_MAP di bawah. Menambah arsitektur = menambah satu entri di sana;
prefiks yang tidak cocok dengan model akan menggagalkan run, bukan didiamkan.

Scheduler, early stopping, AMP, dan augmentasi tidak disentuh modul ini.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Peta lapisan per arsitektur
#
# timm menamai pohon modul tiap keluarga arsitektur berbeda-beda: ConvNeXt
# memakai stem/stages.N, Swin memakai patch_embed/layers.N. Prefiks harfiah
# milik ConvNeXt yang dipakai pada Swin tidak cocok dengan satu parameter pun,
# sehingga freeze membekukan nol parameter dan LLRD runtuh jadi satu grup —
# tanpa error, jadi run-nya tetap "berhasil" dengan strategi yang sebenarnya
# tidak aktif. Karena itu strategi didefinisikan atas *kedalaman* (berapa stage
# dibekukan dari lapisan awal), dan nama konkretnya diambil dari peta ini.
# ---------------------------------------------------------------------------
LAYER_MAP = {
    "convnext_tiny": dict(stem="stem",
                          stages=("stages.0", "stages.1", "stages.2", "stages.3")),
    "swin_tiny": dict(stem="patch_embed",
                      stages=("layers.0", "layers.1", "layers.2", "layers.3")),
}
HEAD_PREFIX = "head"

# ---------------------------------------------------------------------------
# Definisi strategi
#   n_freeze  : jumlah stage yang dibekukan dari lapisan awal; stem ikut beku
#               bila > 0. 0 = tidak membekukan apa pun.
#   base_lr   : None = pakai rc.lr apa adanya (S0/S1); angka = override (S2-S4)
#   llrd_decay: faktor pengali LR per level ke arah lapisan awal (None = seragam)
# ---------------------------------------------------------------------------
STRATEGIES = {
    "S0": dict(n_freeze=0, base_lr=None, llrd_decay=None, label_smoothing=0.0),
    "S1": dict(n_freeze=4, base_lr=None, llrd_decay=None, label_smoothing=0.0),
    "S2": dict(n_freeze=2, base_lr=1e-4, llrd_decay=None, label_smoothing=0.0),
    "S3": dict(n_freeze=2, base_lr=1e-4, llrd_decay=0.7,  label_smoothing=0.0),
    "S4": dict(n_freeze=2, base_lr=1e-4, llrd_decay=0.7,  label_smoothing=0.1),
}


def _peta(arch: str) -> dict:
    if arch not in LAYER_MAP:
        raise ValueError(
            f"arsitektur '{arch}' belum ada di LAYER_MAP — tambahkan nama stem "
            f"dan stage-nya dulu. Terdaftar: {sorted(LAYER_MAP)}")
    return LAYER_MAP[arch]


def freeze_prefixes(arch: str, strategy: str = "S0") -> tuple:
    """Prefiks parameter yang dibekukan strategi ini pada arsitektur ini."""
    n = STRATEGIES[strategy]["n_freeze"]
    if n == 0:
        return ()
    peta = _peta(arch)
    return (peta["stem"],) + tuple(peta["stages"][:n])


def llrd_order(arch: str) -> tuple:
    """Level LLRD dari kepala ke lapisan awal.

    head mendapat base_lr, tiap langkah ke bawah dikali llrd_decay.
    """
    peta = _peta(arch)
    return (HEAD_PREFIX,) + tuple(reversed(peta["stages"])) + (peta["stem"],)


def _cek_cocok(model, prefiks, arch: str, konteks: str) -> None:
    """Gagal keras kalau ada prefiks yang tidak cocok dengan parameter mana pun.

    Tanpa ini, arsitektur yang salah peta berakhir sebagai run yang selesai
    normal dengan strategi mati — kegagalan paling mahal karena tidak terlihat.
    """
    nama = [n for n, _ in model.named_parameters()]
    hilang = [p for p in prefiks if not any(n.startswith(p) for n in nama)]
    if hilang:
        raise ValueError(
            f"{konteks}: prefiks {hilang} tidak cocok dengan parameter mana pun "
            f"pada model '{arch}'. Peta lapisan salah — periksa LAYER_MAP.")


def _prefix_of(param_name: str, urutan) -> str:
    """Kembalikan level LLRD untuk sebuah nama parameter."""
    for pfx in urutan:
        if param_name.startswith(pfx):
            return pfx
    # norm / norm_pre / lain-lain: perlakukan seperti head (LR penuh)
    return HEAD_PREFIX


def freeze_layers(model, strategy: str = "S0", arch: str = None) -> None:
    """Set requires_grad sesuai strategi. `arch` wajib — lihat _cek_cocok."""
    prefiks = freeze_prefixes(arch, strategy)
    if prefiks:
        _cek_cocok(model, prefiks, arch, f"freeze_layers({strategy})")
    for name, p in model.named_parameters():
        p.requires_grad = not name.startswith(prefiks) if prefiks else True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[finetune] {strategy}/{arch}: trainable {n_train/1e6:.1f}M / {n_total/1e6:.1f}M param")


def build_param_groups(model, strategy: str = "S0", base_lr: float = 3e-4,
                       arch: str = None):
    """
    Kembalikan param_groups untuk AdamW.
      - S0/S1: satu grup, LR = base_lr (perilaku baseline).
      - S2   : satu grup, LR = 1e-4.
      - S3/S4: grup per level dengan LLRD (head 1e-4; stage terdalam x0.7; dst).
    Hanya parameter requires_grad=True yang dimasukkan.
    """
    cfg = STRATEGIES[strategy]
    lr0 = cfg["base_lr"] if cfg["base_lr"] is not None else base_lr

    if cfg["llrd_decay"] is None:
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "lr": lr0}]

    urutan = llrd_order(arch)
    # Kepala boleh saja bernama lain (fallback _prefix_of menanganinya), tapi
    # stage dan stem wajib cocok — kalau tidak, LLRD runtuh jadi satu grup.
    _cek_cocok(model, urutan[1:], arch, f"build_param_groups({strategy})")

    decay = cfg["llrd_decay"]
    lr_of = {pfx: lr0 * (decay ** i) for i, pfx in enumerate(urutan)}
    buckets = {pfx: [] for pfx in urutan}
    for name, p in model.named_parameters():
        if p.requires_grad:
            buckets[_prefix_of(name, urutan)].append(p)

    groups = [{"params": ps, "lr": lr_of[pfx]} for pfx, ps in buckets.items() if ps]
    print("[finetune] LLRD LR:", {pfx: f"{lr_of[pfx]:.1e}" for pfx, ps in buckets.items() if ps})
    return groups


# ---------------------------------------------------------------------------
# Uji cepat tanpa data:  python -m src.cvl.finetune
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import timm, torch
    from .config import ALL_ARCHITECTURES

    for arch in LAYER_MAP:
        print(f"=== {arch} ({ALL_ARCHITECTURES[arch]}) ===")
        for strat in STRATEGIES:
            model = timm.create_model(ALL_ARCHITECTURES[arch], pretrained=False,
                                      num_classes=308)
            freeze_layers(model, strat, arch=arch)
            groups = build_param_groups(model, strat, base_lr=3e-4, arch=arch)
            opt = torch.optim.AdamW(groups, weight_decay=0.05)
            x = torch.randn(2, 3, 224, 224)
            loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([1, 2]))
            loss.backward(); opt.step()
            print(f"  {strat}: OK ({len(groups)} param group)\n")
    print("Semua strategi lolos smoke-test di semua arsitektur.")
```
