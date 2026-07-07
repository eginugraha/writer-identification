# Writer-ID Architecture Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bangun pipeline eksperimen yang membandingkan 5 arsitektur (ResNet-50, ConvNeXt-Tiny, EfficientNetV2-S, ViT-Small, Swin-Tiny) untuk klasifikasi penulis pada CVL, dengan ablasi data terbatas (halaman latih/penulis = 1,2,3,4,penuh) dan sumbu pretrained vs from-scratch.

**Architecture:** Pipeline modular Python: (1) `data_prep` mem-parse potongan baris CVL & membangun manifest split per-penulis yang reproducible; (2) `dataset`/`models` menyediakan data & backbone `timm`; (3) `train`/`evaluate` menjalankan satu run & menghitung metrik klasifikasi + retrieval; (4) `run_experiments` melooping grid 150 run (resume-able); (5) `report` merangkum `results.csv` jadi tabel & grafik. Logika murni (parsing, split, metrik) di-TDD; loop training di-smoke-test dengan fixture kecil.

**Tech Stack:** Python 3.10+, PyTorch, `timm`, torchvision, numpy, pandas, scikit-learn, Pillow, PyYAML, tqdm, matplotlib, pytest. Package manager: `uv` (fallback: `pip`+venv).

## Global Constraints

- Working dir proyek: `/Users/eginugraha/personal/thesis`. Dataset baris: `cvl-database-1-1/**/lines/` (nama file `{writer}-{page}-{line}.tif`).
- **Exclude writers:** `0431` dan `0161` selalu dibuang.
- **Kohor:** hanya penulis dengan **≥5 halaman**; penulis <5 halaman dibuang dan jumlahnya dicatat.
- **Test split:** tepat **1 halaman/penulis** (indeks halaman terbesar, deterministik), tidak pernah dilatih.
- **Ablasi:** halaman latih/penulis N ∈ `{1,2,3,4,None}` (None = penuh). Test identik antar-level.
- **Input:** grayscale→3ch, resize aspek-terjaga tinggi 224 → pad/crop ke **224×224**, normalisasi ImageNet.
- **Augmentasi:** random resized-crop ringan + rotasi/affine kecil + jitter kontras. **TANPA horizontal flip.**
- **Fairness:** recipe training identik antar-arsitektur dalam tiap mode; params ~21–28M.
- **Metrik:** Top-1, Top-5, macro-F1 (agregasi baris→halaman via rata-rata softmax) + mAP & Top-1 retrieval (fitur pra-head, leave-one-out level-baris) + params/GFLOPs/waktu/throughput.
- **Seed:** semua sampling deterministik; 3 seed `{0,1,2}` per kombinasi.
- **Reproducibility:** manifest split disimpan ke disk; `results.csv` 1 baris/run; runner skip run yang sudah ada.
- Semua path relatif ke root proyek. Semua modul di package `src/cvl/`.

---

### Task 1: Setup proyek & dependensi

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`, `src/cvl/__init__.py`, `tests/__init__.py`, `configs/default.yaml`
- Create: `src/cvl/config.py`

**Interfaces:**
- Produces: konstanta `EXCLUDE_WRITERS: set[str]`, `MIN_PAGES: int`, `IMAGE_SIZE: int`, `IMAGENET_MEAN/STD`, `ARCHITECTURES: dict[str,str]`, `ABLATION_LEVELS: list[int|None]`, `SEEDS: list[int]`; helper `project_root() -> Path`, `lines_root() -> Path`.

- [ ] **Step 1: Inisialisasi git & struktur**

```bash
cd /Users/eginugraha/personal/thesis
git init
mkdir -p src/cvl tests configs results/manifests results/checkpoints results/figures
touch src/cvl/__init__.py tests/__init__.py
```

- [ ] **Step 2: Tulis `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
results/checkpoints/
results/manifests/
results/*.csv
results/figures/
.pytest_cache/
cvl-database-1-1/
```

- [ ] **Step 3: Tulis `requirements.txt`**

```
torch
torchvision
timm>=1.0.0
numpy
pandas
scikit-learn
Pillow
pyyaml
tqdm
matplotlib
pytest
```

- [ ] **Step 4: Tulis `pyproject.toml`**

```toml
[project]
name = "cvl-writer-id"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 5: Buat environment & install**

```bash
cd /Users/eginugraha/personal/thesis
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
# fallback: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

- [ ] **Step 6: Tulis `src/cvl/config.py`**

```python
from pathlib import Path

EXCLUDE_WRITERS = {"0431", "0161"}
MIN_PAGES = 5
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ARCHITECTURES = {
    "resnet50": "resnet50",
    "convnext_tiny": "convnext_tiny",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "vit_small": "vit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}
ABLATION_LEVELS = [1, 2, 3, 4, None]  # None = full
SEEDS = [0, 1, 2]

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def lines_root() -> Path:
    return project_root() / "cvl-database-1-1"
```

- [ ] **Step 7: Tulis `configs/default.yaml`**

```yaml
image_size: 224
batch_size: 64
val_frac: 0.1
pretrained_epochs: 40
scratch_epochs: 150
early_stop_patience: 8
lr: 0.0003
weight_decay: 0.05
num_workers: 8
amp: true
```

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "chore: project scaffold, deps, config constants"
```

---

### Task 2: Parsing nama file baris CVL

**Files:**
- Create: `src/cvl/data_prep.py`
- Test: `tests/test_parse.py`

**Interfaces:**
- Produces: `parse_line_filename(name: str) -> tuple[str, str, int]` → `(writer, page, line_idx)`. `writer`/`page` string asli (zero-padded), `line_idx` int. Raise `ValueError` untuk nama tak valid.

- [ ] **Step 1: Tulis test gagal**

```python
# tests/test_parse.py
import pytest
from src.cvl.data_prep import parse_line_filename

def test_parse_basic():
    assert parse_line_filename("0050-8-4.tif") == ("0050", "8", 4)

def test_parse_writer_padding_preserved():
    assert parse_line_filename("0001-1-0.tif") == ("0001", "1", 0)

def test_parse_invalid_raises():
    with pytest.raises(ValueError):
        parse_line_filename("garbage.tif")
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `pytest tests/test_parse.py -v`
Expected: FAIL (`ImportError`/`function not defined`)

- [ ] **Step 3: Implementasi minimal**

```python
# src/cvl/data_prep.py
from pathlib import Path
import re

_LINE_RE = re.compile(r"^(\d+)-(\d+)-(\d+)\.tif$", re.IGNORECASE)

def parse_line_filename(name: str) -> tuple[str, str, int]:
    m = _LINE_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"nama file baris tak valid: {name}")
    writer, page, line = m.group(1), m.group(2), int(m.group(3))
    return writer, page, line
```

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `pytest tests/test_parse.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cvl/data_prep.py tests/test_parse.py && git commit -m "feat: parse CVL line filenames"
```

---

### Task 3: Scan direktori & inventaris penulis→halaman

**Files:**
- Modify: `src/cvl/data_prep.py`
- Test: `tests/test_scan.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: `parse_line_filename`.
- Produces: `scan_lines(root: Path) -> pandas.DataFrame` kolom `["writer","page","line","path"]` (semua `.tif` di bawah folder `lines/`). `writer`/`page` str, `line` int, `path` str absolut.
- Produces fixture `tiny_lines` (conftest): direktori sementara meniru struktur `lines/` untuk 3 penulis.

- [ ] **Step 1: Tulis fixture generator di `tests/conftest.py`**

```python
# tests/conftest.py
import pytest
from PIL import Image

def _make_line(dirpath, writer, page, line, w=120, h=48):
    dirpath.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (w, h), (200, 200, 200))
    img.save(dirpath / f"{writer}-{page}-{line}.tif")

@pytest.fixture
def tiny_lines(tmp_path):
    """3 penulis: A=6 halaman, B=5 halaman, C=3 halaman (di bawah min)."""
    lines = tmp_path / "lines"
    plan = {"0001": 6, "0002": 5, "0003": 3}
    for writer, npages in plan.items():
        for p in range(1, npages + 1):
            for ln in range(4):  # 4 baris/halaman
                _make_line(lines / writer, writer, p, ln)
    return tmp_path
```

- [ ] **Step 2: Tulis test gagal**

```python
# tests/test_scan.py
from src.cvl.data_prep import scan_lines

def test_scan_counts(tiny_lines):
    df = scan_lines(tiny_lines)
    assert len(df) == (6 + 5 + 3) * 4
    assert set(df["writer"].unique()) == {"0001", "0002", "0003"}
    assert df.loc[df.writer == "0001", "page"].nunique() == 6
```

- [ ] **Step 3: Jalankan, pastikan gagal**

Run: `pytest tests/test_scan.py -v`
Expected: FAIL

- [ ] **Step 4: Implementasi `scan_lines`**

```python
# tambah ke src/cvl/data_prep.py
import pandas as pd

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
```

- [ ] **Step 5: Jalankan, pastikan lolos**

Run: `pytest tests/test_scan.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/cvl/data_prep.py tests/test_scan.py tests/conftest.py && git commit -m "feat: scan CVL lines into dataframe"
```

---

### Task 4: Filter kohor penulis (exclude + min halaman)

**Files:**
- Modify: `src/cvl/data_prep.py`
- Test: `tests/test_filter.py`

**Interfaces:**
- Consumes: output `scan_lines`, `EXCLUDE_WRITERS`, `MIN_PAGES`.
- Produces: `filter_cohort(df, min_pages=MIN_PAGES, exclude=EXCLUDE_WRITERS) -> tuple[pandas.DataFrame, dict]`. DataFrame hanya penulis lolos; dict `{"n_excluded_rule": int, "n_kept_writers": int, "dropped_writers": list[str]}`.

- [ ] **Step 1: Tulis test gagal**

```python
# tests/test_filter.py
from src.cvl.data_prep import scan_lines, filter_cohort

def test_filter_min_pages(tiny_lines):
    df = scan_lines(tiny_lines)
    kept, info = filter_cohort(df, min_pages=5, exclude=set())
    assert set(kept["writer"].unique()) == {"0001", "0002"}  # C punya 3 halaman → dibuang
    assert info["n_kept_writers"] == 2
    assert "0003" in info["dropped_writers"]

def test_filter_exclude(tiny_lines):
    df = scan_lines(tiny_lines)
    kept, info = filter_cohort(df, min_pages=5, exclude={"0001"})
    assert set(kept["writer"].unique()) == {"0002"}
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `pytest tests/test_filter.py -v`
Expected: FAIL

- [ ] **Step 3: Implementasi `filter_cohort`**

```python
# tambah ke src/cvl/data_prep.py
from .config import EXCLUDE_WRITERS, MIN_PAGES

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
```

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `pytest tests/test_filter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cvl/data_prep.py tests/test_filter.py && git commit -m "feat: filter writer cohort by exclusion and min pages"
```

---

### Task 5: Bangun manifest split (test tetap + ablasi + val)

**Files:**
- Modify: `src/cvl/data_prep.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: output `filter_cohort`.
- Produces:
  - `build_label_map(df) -> dict[str,int]` (writer→idx, urut).
  - `build_manifest(df, n_train_pages: int|None, seed: int, test_pages: int = 1, val_frac: float = 0.1) -> pandas.DataFrame` kolom `["writer","page","line","path","label","split"]`, `split ∈ {"train","val","test"}`. Test = `test_pages` halaman ber-indeks terbesar/penulis. Train = N halaman (atau semua jika None) dari sisa, dipilih deterministik oleh `seed`; sebagian baris train→val. Determinisme: seed sama → hasil sama.

- [ ] **Step 1: Tulis test gagal**

```python
# tests/test_manifest.py
from src.cvl.data_prep import scan_lines, filter_cohort, build_label_map, build_manifest

def _prep(tiny_lines):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    return kept

def test_label_map(tiny_lines):
    kept = _prep(tiny_lines)
    lm = build_label_map(kept)
    assert lm == {"0001": 0, "0002": 1}

def test_test_page_fixed_and_excluded_from_train(tiny_lines):
    kept = _prep(tiny_lines)
    m = build_manifest(kept, n_train_pages=2, seed=0)
    for w, g in m.groupby("writer"):
        test_pages = set(g[g.split == "test"]["page"])
        train_pages = set(g[g.split.isin(["train", "val"])]["page"])
        assert len(test_pages) == 1
        assert test_pages.isdisjoint(train_pages)
        assert len(train_pages) == 2  # N=2

def test_full_uses_all_non_test_pages(tiny_lines):
    kept = _prep(tiny_lines)
    m = build_manifest(kept, n_train_pages=None, seed=0)
    tp = m[(m.writer == "0001") & (m.split.isin(["train", "val"]))]["page"].nunique()
    assert tp == 5  # 6 halaman - 1 test

def test_deterministic(tiny_lines):
    kept = _prep(tiny_lines)
    a = build_manifest(kept, n_train_pages=2, seed=0)
    b = build_manifest(kept, n_train_pages=2, seed=0)
    assert a.sort_values("path").reset_index(drop=True).equals(
           b.sort_values("path").reset_index(drop=True))
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `pytest tests/test_manifest.py -v`
Expected: FAIL

- [ ] **Step 3: Implementasi**

```python
# tambah ke src/cvl/data_prep.py
import numpy as np

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

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `pytest tests/test_manifest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cvl/data_prep.py tests/test_manifest.py && git commit -m "feat: build reproducible per-writer split manifests"
```

---

### Task 6: Transform gambar & Dataset

**Files:**
- Create: `src/cvl/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: manifest DataFrame (Task 5), konstanta config.
- Produces:
  - `build_transforms(train: bool, image_size: int = IMAGE_SIZE) -> Callable` (torchvision transform; **tanpa horizontal flip**).
  - `LineDataset(manifest_subset: pandas.DataFrame, train: bool)` — `__len__`, `__getitem__(i) -> (Tensor[3,H,W], int label)`.

- [ ] **Step 1: Tulis test gagal**

```python
# tests/test_dataset.py
import torch
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.dataset import LineDataset, build_transforms

def test_transform_shape():
    from PIL import Image
    t = build_transforms(train=False)
    out = t(Image.new("RGB", (300, 60)))
    assert out.shape == (3, 224, 224)

def test_dataset_item(tiny_lines):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    train = m[m.split == "train"]
    ds = LineDataset(train, train=True)
    x, y = ds[0]
    assert x.shape == (3, 224, 224) and isinstance(y, int)
    assert 0 <= y < 2
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: Implementasi `src/cvl/dataset.py`**

```python
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

def build_transforms(train: bool, image_size: int = IMAGE_SIZE):
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(image_size),
            T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(), norm,
        ])
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(), norm,
    ])

class LineDataset(Dataset):
    def __init__(self, manifest_subset, train: bool):
        self.rows = manifest_subset.reset_index(drop=True)
        self.tf = build_transforms(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), int(r["label"])
```

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cvl/dataset.py tests/test_dataset.py && git commit -m "feat: line dataset and no-flip transforms"
```

---

### Task 7: Factory model + ekstraksi fitur

**Files:**
- Create: `src/cvl/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `ARCHITECTURES`.
- Produces:
  - `build_model(arch_key: str, num_classes: int, pretrained: bool) -> torch.nn.Module`.
  - `forward_features(model, x) -> Tensor[B, D]` (fitur pra-head ter-pool untuk retrieval).
  - `count_params(model) -> int`.

- [ ] **Step 1: Tulis test gagal**

```python
# tests/test_models.py
import torch
from src.cvl.models import build_model, forward_features, count_params

def test_forward_logits_shape():
    m = build_model("resnet50", num_classes=7, pretrained=False).eval()
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 7)

def test_forward_features_shape():
    m = build_model("resnet50", num_classes=7, pretrained=False).eval()
    f = forward_features(m, torch.randn(2, 3, 224, 224))
    assert f.dim() == 2 and f.shape[0] == 2

def test_count_params_positive():
    m = build_model("vit_small", num_classes=5, pretrained=False)
    assert count_params(m) > 1_000_000
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `pytest tests/test_models.py -v`
Expected: FAIL

- [ ] **Step 3: Implementasi `src/cvl/models.py`**

```python
import timm
import torch
from .config import ARCHITECTURES

def build_model(arch_key: str, num_classes: int, pretrained: bool):
    return timm.create_model(
        ARCHITECTURES[arch_key], pretrained=pretrained, num_classes=num_classes
    )

def forward_features(model, x):
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

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `pytest tests/test_models.py -v`
Expected: PASS (butuh unduh cache timm hanya jika pretrained=True; test pakai pretrained=False)

- [ ] **Step 5: Commit**

```bash
git add src/cvl/models.py tests/test_models.py && git commit -m "feat: timm model factory and feature extractor"
```

---

### Task 8: Metrik — agregasi, akurasi, F1, mAP retrieval

**Files:**
- Create: `src/cvl/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `aggregate_by_group(probs: np.ndarray[N,C], groups: np.ndarray[N]) -> tuple[np.ndarray, np.ndarray]` → `(group_ids sorted, mean_probs[G,C])`.
  - `top_k_accuracy(probs: np.ndarray[N,C], labels: np.ndarray[N], k: int) -> float`.
  - `macro_f1(probs, labels) -> float`.
  - `retrieval_map(features: np.ndarray[N,D], labels: np.ndarray[N]) -> tuple[float, float]` → `(mAP, top1_retrieval)` cosine, leave-one-out.

- [ ] **Step 1: Tulis test gagal**

```python
# tests/test_metrics.py
import numpy as np
from src.cvl.metrics import aggregate_by_group, top_k_accuracy, macro_f1, retrieval_map

def test_aggregate_mean():
    probs = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]])
    groups = np.array([0, 0, 1])
    gids, mp = aggregate_by_group(probs, groups)
    assert list(gids) == [0, 1]
    assert np.allclose(mp[0], [0.8, 0.2])

def test_top1_perfect():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 1])
    assert top_k_accuracy(probs, labels, 1) == 1.0

def test_macro_f1_perfect():
    probs = np.eye(3)
    labels = np.array([0, 1, 2])
    assert macro_f1(probs, labels) == 1.0

def test_retrieval_map_separable():
    # dua kelas, fitur terpisah jelas → mAP sempurna
    feats = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    m, t1 = retrieval_map(feats, labels)
    assert m > 0.99 and t1 == 1.0
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Implementasi `src/cvl/metrics.py`**

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
        rel = (labels[order] == labels[i]).astype(int)
        top1.append(rel[0])
        if rel.sum() == 0:
            continue
        cum = np.cumsum(rel)
        precision_at_hits = cum[rel == 1] / (np.where(rel == 1)[0] + 1)
        aps.append(precision_at_hits.mean())
    return float(np.mean(aps)) if aps else 0.0, float(np.mean(top1))
```

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cvl/metrics.py tests/test_metrics.py && git commit -m "feat: classification + retrieval metrics"
```

---

### Task 9: Train satu run (+ smoke test)

**Files:**
- Create: `src/cvl/train.py`
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `LineDataset`, `build_model`, config YAML.
- Produces: `RunConfig` (dataclass: `arch, level, mode("pretrained"|"scratch"), seed, epochs, lr, batch_size, ...`) dan `train_one_run(manifest, run_cfg, out_dir, device, hp) -> dict` → menyimpan `best.pt` ke `out_dir`, kembalikan `{"best_val_acc": float, "train_time_s": float, "epochs_ran": int}`.

- [ ] **Step 1: Tulis smoke test gagal**

```python
# tests/test_train_smoke.py
import torch
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.train import RunConfig, train_one_run

def test_train_smoke(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    rc = RunConfig(arch="resnet50", level=2, mode="scratch", seed=0,
                   epochs=1, lr=1e-3, batch_size=8)
    hp = {"val_frac": 0.1, "num_workers": 0, "amp": False, "early_stop_patience": 1}
    out = train_one_run(m, rc, tmp_path, device="cpu", hp=hp)
    assert (tmp_path / "best.pt").exists()
    assert "best_val_acc" in out and out["epochs_ran"] >= 1
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `pytest tests/test_train_smoke.py -v`
Expected: FAIL

- [ ] **Step 3: Implementasi `src/cvl/train.py`**

```python
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import LineDataset
from .models import build_model

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

def train_one_run(manifest, rc: RunConfig, out_dir, device, hp: dict) -> dict:
    _seed_all(rc.seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = LineDataset(manifest[manifest.split == "train"], train=True)
    val_ds = LineDataset(manifest[manifest.split == "val"], train=False)
    nw = hp.get("num_workers", 0)
    tl = DataLoader(train_ds, batch_size=rc.batch_size, shuffle=True, num_workers=nw)
    vl = DataLoader(val_ds, batch_size=rc.batch_size, shuffle=False, num_workers=nw)
    model = build_model(rc.arch, _num_classes(manifest), pretrained=(rc.mode == "pretrained")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=rc.lr, weight_decay=rc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, rc.epochs))
    crit = torch.nn.CrossEntropyLoss()
    use_amp = hp.get("amp", False) and device != "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_acc, best_state, patience, bad = -1.0, None, hp.get("early_stop_patience", 8), 0
    t0, epochs_ran = time.time(), 0
    for epoch in range(rc.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step(); epochs_ran += 1
        # validasi
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item(); total += len(y)
        acc = correct / max(1, total)
        if acc > best_acc:
            best_acc, best_state, bad = acc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    torch.save(best_state or model.state_dict(), out_dir / "best.pt")
    return {"best_val_acc": float(best_acc), "train_time_s": time.time() - t0, "epochs_ran": epochs_ran}
```

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `pytest tests/test_train_smoke.py -v`
Expected: PASS (lambat di CPU tapi selesai; 1 epoch, batch kecil)

- [ ] **Step 5: Commit**

```bash
git add src/cvl/train.py tests/test_train_smoke.py && git commit -m "feat: single training run with early stopping"
```

---

### Task 10: Evaluate checkpoint (klasifikasi + retrieval + efisiensi)

**Files:**
- Create: `src/cvl/evaluate.py`
- Test: `tests/test_evaluate_smoke.py`

**Interfaces:**
- Consumes: `best.pt`, manifest test split, `build_model`, `forward_features`, metrik (Task 8), `count_params`.
- Produces: `evaluate_checkpoint(ckpt_path, manifest, arch, device, batch_size=64) -> dict` dengan kunci: `top1_page, top5_page, macro_f1_page, map_line, top1_retrieval, n_params, throughput_img_s`. Agregasi klasifikasi ke level **halaman** (grup = `writer|page`).

- [ ] **Step 1: Tulis smoke test gagal**

```python
# tests/test_evaluate_smoke.py
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.train import RunConfig, train_one_run
from src.cvl.evaluate import evaluate_checkpoint

def test_evaluate_smoke(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    rc = RunConfig(arch="resnet50", level=2, mode="scratch", seed=0, epochs=1, batch_size=8)
    train_one_run(m, rc, tmp_path, device="cpu",
                  hp={"num_workers": 0, "amp": False, "early_stop_patience": 1})
    res = evaluate_checkpoint(tmp_path / "best.pt", m, arch="resnet50", device="cpu", batch_size=8)
    for k in ["top1_page", "macro_f1_page", "map_line", "top1_retrieval", "n_params"]:
        assert k in res
    assert 0.0 <= res["top1_page"] <= 1.0
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `pytest tests/test_evaluate_smoke.py -v`
Expected: FAIL

- [ ] **Step 3: Implementasi `src/cvl/evaluate.py`**

```python
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import LineDataset
from .models import build_model, forward_features, count_params
from .metrics import aggregate_by_group, top_k_accuracy, macro_f1, retrieval_map

def _num_classes(manifest) -> int:
    return int(manifest["label"].max()) + 1

def evaluate_checkpoint(ckpt_path, manifest, arch, device, batch_size: int = 64) -> dict:
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    model = build_model(arch, _num_classes(manifest), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    ds = LineDataset(test, train=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, feats = [], []
    t0, n_img = time.time(), 0
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device); n_img += len(x)
            probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
            feats.append(forward_features(model, x).cpu().numpy())
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

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `pytest tests/test_evaluate_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cvl/evaluate.py tests/test_evaluate_smoke.py && git commit -m "feat: evaluate checkpoint with classification + retrieval metrics"
```

---

### Task 11: Runner grid (resume-able) + prep manifest CLI

**Files:**
- Create: `src/cvl/run_experiments.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: semua di atas + `configs/default.yaml`.
- Produces:
  - `run_id(arch, level, mode, seed) -> str` (mis. `resnet50_L2_scratch_s0`; `Lfull` untuk None).
  - `already_done(results_csv, rid) -> bool`.
  - `run_grid(manifest_by_level: dict, archs, levels, modes, seeds, results_csv, ckpt_root, device, hp) -> None` — untuk tiap kombinasi: skip jika sudah ada; train+eval; append 1 baris ke `results_csv`.

- [ ] **Step 1: Tulis test gagal**

```python
# tests/test_runner.py
import pandas as pd
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.run_experiments import run_id, already_done, run_grid

def test_run_id():
    assert run_id("vit_small", 3, "pretrained", 1) == "vit_small_L3_pretrained_s1"
    assert run_id("resnet50", None, "scratch", 0) == "resnet50_Lfull_scratch_s0"

def test_run_grid_and_resume(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    mbl = {2: build_manifest(kept, n_train_pages=2, seed=0)}
    csv = tmp_path / "results.csv"
    hp = {"num_workers": 0, "amp": False, "early_stop_patience": 1,
          "pretrained_epochs": 1, "scratch_epochs": 1, "batch_size": 8, "lr": 1e-3}
    run_grid({0: mbl}, archs=["resnet50"], levels=[2], modes=["scratch"], seeds=[0],
             results_csv=csv, ckpt_root=tmp_path / "ck", device="cpu", hp=hp)
    n1 = len(pd.read_csv(csv))
    assert n1 == 1
    assert already_done(csv, run_id("resnet50", 2, "scratch", 0))
    # jalankan lagi → tidak menambah baris (resume skip)
    run_grid({0: mbl}, archs=["resnet50"], levels=[2], modes=["scratch"], seeds=[0],
             results_csv=csv, ckpt_root=tmp_path / "ck", device="cpu", hp=hp)
    assert len(pd.read_csv(csv)) == 1
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Implementasi `src/cvl/run_experiments.py`**

```python
from pathlib import Path
import pandas as pd
from .train import RunConfig, train_one_run
from .evaluate import evaluate_checkpoint

def run_id(arch, level, mode, seed) -> str:
    lvl = "full" if level is None else str(level)
    return f"{arch}_L{lvl}_{mode}_s{seed}"

def already_done(results_csv, rid: str) -> bool:
    p = Path(results_csv)
    if not p.exists():
        return False
    df = pd.read_csv(p)
    return "run_id" in df.columns and rid in set(df["run_id"].astype(str))

def _append_row(results_csv, row: dict):
    p = Path(results_csv); p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(p, mode="a", header=not p.exists(), index=False)

def run_grid(manifest_by_seed_level, archs, levels, modes, seeds,
             results_csv, ckpt_root, device, hp) -> None:
    ckpt_root = Path(ckpt_root)
    for seed in seeds:
        for level in levels:
            manifest = manifest_by_seed_level[seed][level]
            for arch in archs:
                for mode in modes:
                    rid = run_id(arch, level, mode, seed)
                    if already_done(results_csv, rid):
                        print(f"skip {rid}"); continue
                    epochs = hp["pretrained_epochs"] if mode == "pretrained" else hp["scratch_epochs"]
                    rc = RunConfig(arch=arch, level=level, mode=mode, seed=seed,
                                   epochs=epochs, lr=hp["lr"], batch_size=hp["batch_size"])
                    out_dir = ckpt_root / rid
                    tr = train_one_run(manifest, rc, out_dir, device, hp)
                    ev = evaluate_checkpoint(out_dir / "best.pt", manifest, arch, device,
                                             batch_size=hp["batch_size"])
                    _append_row(results_csv, {"run_id": rid, "arch": arch,
                        "level": ("full" if level is None else level), "mode": mode,
                        "seed": seed, **tr, **ev})
                    print(f"done {rid}: top1={ev['top1_page']:.3f} map={ev['map_line']:.3f}")
```

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `pytest tests/test_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cvl/run_experiments.py tests/test_runner.py && git commit -m "feat: resume-able experiment grid runner"
```

---

### Task 12: Entry-point CLI (prep + run) untuk RunPod

**Files:**
- Create: `scripts/prep_manifests.py`, `scripts/run_all.py`
- Test: manual (dijalankan di RunPod dengan dataset nyata)

**Interfaces:**
- Consumes: semua modul + `configs/default.yaml`.
- Produces: `scripts/prep_manifests.py` menghasilkan `results/manifests/seed{S}_L{level}.parquet` untuk semua seed×level, plus mencetak `n_kept_writers` & `dropped_writers`. `scripts/run_all.py` memuat manifest itu & memanggil `run_grid` untuk grid penuh.

- [ ] **Step 1: Tulis `scripts/prep_manifests.py`**

```python
import yaml
from pathlib import Path
from src.cvl.config import lines_root, ABLATION_LEVELS, SEEDS
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest

def main():
    out = Path("results/manifests"); out.mkdir(parents=True, exist_ok=True)
    df = scan_lines(lines_root())
    kept, info = filter_cohort(df)
    print(f"kept writers={info['n_kept_writers']} dropped(<5 pages)={info['dropped_writers']}")
    for seed in SEEDS:
        for level in ABLATION_LEVELS:
            m = build_manifest(kept, n_train_pages=level, seed=seed)
            tag = "full" if level is None else level
            m.to_parquet(out / f"seed{seed}_L{tag}.parquet")
    print("manifests written")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Tulis `scripts/run_all.py`**

```python
import yaml
from pathlib import Path
import pandas as pd
import torch
from src.cvl.config import ARCHITECTURES, ABLATION_LEVELS, SEEDS
from src.cvl.run_experiments import run_grid

def main():
    hp = yaml.safe_load(open("configs/default.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    man_dir = Path("results/manifests")
    by_seed_level = {}
    for seed in SEEDS:
        by_seed_level[seed] = {}
        for level in ABLATION_LEVELS:
            tag = "full" if level is None else level
            by_seed_level[seed][level] = pd.read_parquet(man_dir / f"seed{seed}_L{tag}.parquet")
    run_grid(by_seed_level, archs=list(ARCHITECTURES.keys()),
             levels=ABLATION_LEVELS, modes=["pretrained", "scratch"], seeds=SEEDS,
             results_csv="results/results.csv", ckpt_root="results/checkpoints",
             device=device, hp=hp)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verifikasi impor (tanpa dataset)**

Run: `python -c "import scripts.prep_manifests, scripts.run_all; print('ok')"`
Expected: `ok` (tidak error impor)

- [ ] **Step 4: (Di RunPod) jalankan prep lalu grid**

```bash
python scripts/prep_manifests.py
python scripts/run_all.py   # resume-able; aman diulang jika sesi putus
```
Expected: `results/results.csv` terisi hingga 150 baris.

- [ ] **Step 5: Commit**

```bash
git add scripts/ && git commit -m "feat: CLI entry-points for manifest prep and full grid"
```

---

### Task 13: Laporan — tabel & grafik dari results.csv

**Files:**
- Create: `src/cvl/report.py`, `scripts/make_report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `results/results.csv`.
- Produces:
  - `summarize(df) -> pandas.DataFrame` — mean±std Top-1/mAP per `(arch, level, mode)`.
  - `pivot_markdown(df, metric, mode) -> str` — tabel markdown arch×level.
  - `plot_accuracy_vs_n(df, mode, out_png) -> None` — kurva Top-1 vs level per arsitektur.
  - `scripts/make_report.py` menulis `dokumentasi/08-hasil-eksperimen.md` + PNG ke `results/figures/`.

- [ ] **Step 1: Tulis test gagal**

```python
# tests/test_report.py
import pandas as pd
from src.cvl.report import summarize, pivot_markdown

def _fake():
    rows = []
    for arch in ["resnet50", "vit_small"]:
        for level in [1, "full"]:
            for seed in [0, 1]:
                rows.append({"arch": arch, "level": level, "mode": "pretrained",
                             "seed": seed, "top1_page": 0.8, "map_line": 0.7})
    return pd.DataFrame(rows)

def test_summarize_shape():
    s = summarize(_fake())
    assert {"arch", "level", "mode", "top1_page_mean", "top1_page_std"} <= set(s.columns)

def test_pivot_markdown_contains_arch():
    md = pivot_markdown(_fake(), metric="top1_page", mode="pretrained")
    assert "resnet50" in md and "|" in md
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `pytest tests/test_report.py -v`
Expected: FAIL

- [ ] **Step 3: Implementasi `src/cvl/report.py`**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_METRICS = ["top1_page", "top5_page", "macro_f1_page", "map_line", "top1_retrieval"]

def summarize(df):
    present = [m for m in _METRICS if m in df.columns]
    g = df.groupby(["arch", "level", "mode"])[present].agg(["mean", "std"]).reset_index()
    g.columns = ["arch", "level", "mode"] + [f"{m}_{s}" for m in present for s in ("mean", "std")]
    return g

def _level_order(v):
    return 999 if v == "full" else int(v)

def pivot_markdown(df, metric: str, mode: str) -> str:
    s = summarize(df)
    s = s[s["mode"] == mode]
    levels = sorted(s["level"].unique(), key=_level_order)
    archs = sorted(s["arch"].unique())
    header = "| arch | " + " | ".join(f"N={l}" for l in levels) + " |"
    sep = "|" + "---|" * (len(levels) + 1)
    lines = [header, sep]
    for a in archs:
        cells = []
        for l in levels:
            row = s[(s.arch == a) & (s.level == l)]
            if len(row):
                cells.append(f"{row[f'{metric}_mean'].iloc[0]:.3f}±{row[f'{metric}_std'].iloc[0]:.3f}")
            else:
                cells.append("-")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

def plot_accuracy_vs_n(df, mode: str, out_png):
    s = summarize(df); s = s[s["mode"] == mode]
    plt.figure(figsize=(7, 5))
    for a in sorted(s["arch"].unique()):
        sub = s[s.arch == a].copy()
        sub["ord"] = sub["level"].map(_level_order)
        sub = sub.sort_values("ord")
        plt.errorbar(range(len(sub)), sub["top1_page_mean"], yerr=sub["top1_page_std"],
                     marker="o", label=a, capsize=3)
        plt.xticks(range(len(sub)), [str(x) for x in sub["level"]])
    plt.xlabel("halaman latih / penulis (N)"); plt.ylabel("Top-1 (halaman)")
    plt.title(f"Akurasi vs data latih ({mode})"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
```

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `pytest tests/test_report.py -v`
Expected: PASS

- [ ] **Step 5: Tulis `scripts/make_report.py`**

```python
from pathlib import Path
import pandas as pd
from src.cvl.report import pivot_markdown, plot_accuracy_vs_n

def main():
    df = pd.read_csv("results/results.csv")
    fig = Path("results/figures"); fig.mkdir(parents=True, exist_ok=True)
    parts = ["# Hasil Eksperimen — Perbandingan Arsitektur Writer-ID CVL\n"]
    for mode in ["pretrained", "scratch"]:
        plot_accuracy_vs_n(df, mode, fig / f"acc_vs_n_{mode}.png")
        parts += [f"\n## Mode: {mode}\n",
                  "\n### Top-1 (halaman)\n", pivot_markdown(df, "top1_page", mode),
                  "\n\n### mAP (retrieval, baris)\n", pivot_markdown(df, "map_line", mode),
                  f"\n\n![acc](../results/figures/acc_vs_n_{mode}.png)\n"]
    Path("dokumentasi/08-hasil-eksperimen.md").write_text("\n".join(parts))
    print("report written to dokumentasi/08-hasil-eksperimen.md")

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/cvl/report.py scripts/make_report.py tests/test_report.py && git commit -m "feat: report tables and accuracy-vs-N figures"
```

---

## Self-Review

**Spec coverage (spec `06`):**
- §2 Dataset/preprocessing → Task 2,3,6 ✅
- §3 Split per-penulis, 1 halaman test → Task 5 ✅
- §4 Ablasi {1,2,3,4,penuh} → Task 5 (`ABLATION_LEVELS`), Task 11/12 ✅
- §5 5 arsitektur + pretrained/scratch → Task 7 (`ARCHITECTURES`), Task 9 (`mode`), Task 11 (`modes`) ✅
- §6 Metrik Top-1/Top-5/F1 + mAP/Top-1 retrieval + efisiensi → Task 8,10 ✅
- §7 Recipe (AdamW, cosine, AMP, early stop, no-flip, 3 seed) → Task 6,9 ✅
- §8 Grid 150 run resume-able + knob → Task 11 (skip-done), knob = ubah `modes`/`SEEDS` di `run_all.py` ✅
- §9 Struktur kode → Task 1–13 ✅
- §10 Deliverable laporan → Task 13 ✅
- §11 Risiko (resume, variansi, distorsi) → Task 11 resume, 3 seed, agregasi halaman ✅

**Placeholder scan:** tak ada TODO/TBD; semua step berisi kode nyata. ✅

**Type consistency:** `build_manifest(..., n_train_pages, seed)` konsisten dipakai di Task 6,9,10,11,12; `run_id`/`already_done` konsisten Task 11–12; kunci metrik (`top1_page`, `map_line`) konsisten Task 10→13. ✅

**Catatan knob efisiensi (spec §8):** untuk memangkas dari 150 run, edit `scripts/run_all.py` → `modes=["pretrained"]` untuk from-scratch cuma sebagian, atau `SEEDS=[0]` untuk from-scratch. Tidak ada perubahan kode inti.

---

## Execution Handoff

Plan complete. Sesudah review, dua opsi eksekusi (Subagent-Driven direkomendasikan, atau Inline).
