# Grid 5 Seed + Studi Fine-Tuning ConvNeXt-Tiny — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menyiapkan grid utama untuk dijalankan pada 5 seed di dua server, lalu membangun mesin skenario yang menjalankan enam varian fine-tuning ConvNeXt-Tiny di level L1.

**Architecture:** Semua perilaku baru masuk lewat satu objek `Scenario` yang default-nya identik dengan pipeline sekarang, sehingga skenario baseline (`FT0`) tidak perlu dijalankan ulang dan bisa disalin dari hasil grid utama. `Scenario` diteruskan ke `dataset` (geometri & augmentasi), `models` (drop_path & head), `train` (loss, param group, warmup margin), dan `evaluate` (jumlah crop). Runner skenario meniru pola `run_grid` yang sudah ada: satu modul berisi logika, satu skrip tipis sebagai CLI.

**Tech Stack:** Python ≥3.10, PyTorch, timm, torchvision, pandas, numpy, pytest.

Spec: `docs/superpowers/specs/2026-08-05-grid-5seed-dan-finetune-convnext-design.md`

## Global Constraints

- **Semua test jalan di CPU tanpa dataset asli.** Suite memakai fixture `tiny_lines` di `tests/conftest.py`. Tidak boleh ada test yang butuh GPU atau folder `cvl-database-1-1/`.
- **26 test yang sudah ada wajib tetap lolos** setelah setiap task. Perintah verifikasi: `.venv/bin/pytest -q`.
- **`Scenario()` tanpa argumen wajib menghasilkan perilaku yang identik dengan pipeline saat ini.** Ini yang membuat `FT0` sah disalin dari `results-pretrained.csv`. Setiap task yang menyentuh `dataset`, `models`, `train`, atau `evaluate` harus mempertahankan jalur default apa adanya.
- **Tidak ada dependensi baru.** Hanya paket yang sudah ada di `requirements.txt`.
- **Tanda tangan fungsi lama tidak boleh berubah secara breaking.** Parameter baru selalu punya nilai default yang mereproduksi perilaku lama, karena `scripts/run_all.py` dan `src/cvl/run_experiments.py` memanggilnya tanpa argumen baru.
- **Bahasa komentar dan pesan cetak: Bahasa Indonesia**, mengikuti kode yang ada.
- **Format pesan commit** mengikuti repo: `feat:` / `fix:` / `docs:` / `test:` / `chore:`.

## Struktur berkas

| Berkas | Tanggung jawab | Status |
|---|---|---|
| `src/cvl/env_info.py` | Kumpulkan metadata lingkungan run (nama GPU, versi torch/timm) | Baru |
| `src/cvl/config.py` | Katalog grid; `ALL_SEEDS` naik jadi 5 | Ubah |
| `src/cvl/run_experiments.py` | Tulis metadata lingkungan ke tiap baris CSV | Ubah |
| `src/cvl/scenarios.py` | Dataclass `Scenario` + registry `SCENARIOS` | Baru |
| `src/cvl/dataset.py` | Sumbu `geometry` dan `aug`; evaluasi multi-crop | Ubah |
| `src/cvl/arcface.py` | Head ArcFace + pembungkus backbone | Baru |
| `src/cvl/models.py` | Teruskan `drop_path`; bangun head sesuai skenario | Ubah |
| `src/cvl/train.py` | Terima `Scenario`: label smoothing, param group, warmup margin | Ubah |
| `src/cvl/evaluate.py` | Rata-ratakan probabilitas & fitur atas K crop | Ubah |
| `src/cvl/run_scenarios.py` | Logika loop skenario × seed, resume-able | Baru |
| `scripts/run_scenarios.py` | CLI tipis | Baru |

---

### Task 1: Studi 1 siap dijalankan — metadata lingkungan + 5 seed

Ini satu-satunya task yang memblokir 41 jam GPU. Tidak menyentuh logika latih/evaluasi sama sekali.

**Files:**
- Create: `src/cvl/env_info.py`
- Create: `tests/test_env_info.py`
- Modify: `src/cvl/config.py:23`
- Modify: `src/cvl/run_experiments.py:52-54`
- Modify: `tests/test_runner.py`

**Interfaces:**
- Consumes: —
- Produces: `env_info.env_metadata(device: str) -> dict` dengan kunci `gpu_name` (str), `torch_version` (str), `timm_version` (str). Dipakai `run_experiments.run_grid` dan nanti `run_scenarios.run_scenario_grid`.

- [ ] **Step 1: Tulis test yang gagal untuk `env_metadata`**

Buat `tests/test_env_info.py`:

```python
import torch
import timm
from src.cvl.env_info import env_metadata


def test_env_metadata_cpu():
    md = env_metadata("cpu")
    assert md["gpu_name"] == "cpu"
    assert md["torch_version"] == torch.__version__
    assert md["timm_version"] == timm.__version__


def test_env_metadata_keys_are_strings():
    md = env_metadata("cpu")
    assert set(md) == {"gpu_name", "torch_version", "timm_version"}
    assert all(isinstance(v, str) for v in md.values())
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/bin/pytest tests/test_env_info.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.cvl.env_info'`

- [ ] **Step 3: Implementasi `src/cvl/env_info.py`**

```python
"""Metadata lingkungan eksekusi, dicatat per run di results.csv.

Grid sebelumnya dijalankan tanpa mencatat model GPU sama sekali, sehingga
angka waktu latih dan throughput tidak bisa diinterpretasikan belakangan.
Kolom-kolom ini juga yang membuktikan dua server memakai kartu dan versi
library yang sama.
"""
import timm
import torch


def env_metadata(device: str) -> dict:
    """Nama GPU + versi library. `device` "cpu" -> gpu_name "cpu"."""
    if device != "cpu" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "cpu"
    return {
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
    }
```

- [ ] **Step 4: Jalankan test, pastikan lolos**

Run: `.venv/bin/pytest tests/test_env_info.py -v`
Expected: PASS — semua test di berkas ini lolos

- [ ] **Step 5: Tulis test yang gagal untuk kolom baru di CSV**

Tambahkan ke `tests/test_runner.py`:

```python
def test_run_grid_records_env_metadata(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    mbl = {2: build_manifest(kept, n_train_pages=2, seed=0)}
    csv = tmp_path / "results.csv"
    hp = {"num_workers": 0, "amp": False, "early_stop_patience": 1,
          "pretrained_epochs": 1, "scratch_epochs": 1, "batch_size": 8, "lr": 1e-3}
    run_grid({0: mbl}, archs=["resnet50"], levels=[2], modes=["scratch"], seeds=[0],
             results_csv=csv, ckpt_root=tmp_path / "ck", device="cpu", hp=hp)
    row = pd.read_csv(csv).iloc[0]
    assert row["gpu_name"] == "cpu"
    assert isinstance(row["torch_version"], str) and row["torch_version"]
    assert isinstance(row["timm_version"], str) and row["timm_version"]
```

- [ ] **Step 6: Jalankan test, pastikan gagal**

Run: `.venv/bin/pytest tests/test_runner.py::test_run_grid_records_env_metadata -v`
Expected: FAIL — `KeyError: 'gpu_name'`

- [ ] **Step 7: Sambungkan metadata ke `run_grid`**

Di `src/cvl/run_experiments.py`, tambahkan import di bagian atas berkas:

```python
from .env_info import env_metadata
```

Lalu ganti pemanggilan `_append_row` (saat ini di baris 52-54) menjadi:

```python
                    _append_row(results_csv, {"run_id": rid, "arch": arch,
                        "level": ("full" if level is None else level), "mode": mode,
                        "seed": seed, **tr, **ev, **env_metadata(device)})
```

- [ ] **Step 8: Jalankan test, pastikan lolos**

Run: `.venv/bin/pytest tests/test_runner.py -v`
Expected: PASS — semua test di berkas ini lolos

- [ ] **Step 9: Naikkan katalog seed jadi 5**

Di `src/cvl/config.py` baris 23, ubah:

```python
ALL_SEEDS = [0, 1, 2]
```

menjadi:

```python
ALL_SEEDS = [0, 1, 2, 3, 4]
```

- [ ] **Step 10: Tulis test bahwa katalog seed adalah 5 seed**

Tambahkan ke `tests/test_env_info.py`:

```python
def test_katalog_seed_lima():
    from src.cvl.config import ALL_SEEDS
    assert ALL_SEEDS == [0, 1, 2, 3, 4]
```

- [ ] **Step 11: Jalankan seluruh suite**

Run: `.venv/bin/pytest -q`
Expected: PASS — tidak ada kegagalan, dan jumlah test bertambah dari 26

- [ ] **Step 12: Commit**

```bash
git add src/cvl/env_info.py src/cvl/config.py src/cvl/run_experiments.py tests/test_env_info.py tests/test_runner.py
git commit -m "feat: record GPU and library versions per run, raise grid to 5 seeds"
```

---

### Task 2: Registry skenario

Data murni, belum mengubah perilaku apa pun. Task berikutnya membaca registry ini.

**Files:**
- Create: `src/cvl/scenarios.py`
- Create: `tests/test_scenarios.py`

**Interfaces:**
- Consumes: —
- Produces: `Scenario` (dataclass beku) dengan field `geometry: str`, `aug: str`, `drop_path: float`, `label_smoothing: float`, `freeze_strategy: str | None`, `head: str`, `eval_crops: int`. Dan `SCENARIOS: dict[str, Scenario]` berisi kunci `"FT0"`, `"FT1"`, `"FT2"`, `"FT3"`, `"FT4"`, `"AUG"`. Dipakai oleh `dataset`, `models`, `train`, `evaluate`, `run_scenarios`.

- [ ] **Step 1: Tulis test yang gagal**

Buat `tests/test_scenarios.py`:

```python
import dataclasses
import pytest
from src.cvl.scenarios import Scenario, SCENARIOS


def test_default_scenario_adalah_perilaku_sekarang():
    s = Scenario()
    assert s.geometry == "center"
    assert s.aug == "baseline"
    assert s.drop_path == 0.0
    assert s.label_smoothing == 0.0
    assert s.freeze_strategy is None
    assert s.head == "linear"
    assert s.eval_crops == 1


def test_scenario_beku():
    s = Scenario()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.aug = "strong"


def test_registry_lengkap():
    assert set(SCENARIOS) == {"FT0", "FT1", "FT2", "FT3", "FT4", "AUG"}
    assert SCENARIOS["FT0"] == Scenario()


def test_tiap_skenario_mengubah_satu_mekanisme():
    """FT1-FT4 dan AUG masing-masing berbeda dari baseline pada satu sumbu.

    FT1 dikecualikan dari hitungan satu-sumbu: geometry dan eval_crops adalah
    satu mekanisme yang sama (cakupan baris), diterapkan di sisi latih dan uji.
    """
    base = dataclasses.asdict(Scenario())
    diff_count = {}
    for name, sc in SCENARIOS.items():
        d = dataclasses.asdict(sc)
        diff_count[name] = {k for k in d if d[k] != base[k]}
    assert diff_count["FT0"] == set()
    assert diff_count["FT1"] == {"geometry", "eval_crops"}
    assert diff_count["FT2"] == {"drop_path", "label_smoothing"}
    assert diff_count["FT3"] == {"freeze_strategy"}
    assert diff_count["FT4"] == {"head"}
    assert diff_count["AUG"] == {"aug"}


def test_nilai_skenario():
    assert SCENARIOS["FT1"].geometry == "linewindow"
    assert SCENARIOS["FT1"].eval_crops == 9
    assert SCENARIOS["FT2"].drop_path == 0.2
    assert SCENARIOS["FT2"].label_smoothing == 0.1
    assert SCENARIOS["FT3"].freeze_strategy == "S3"
    assert SCENARIOS["FT4"].head == "arcface"
    assert SCENARIOS["AUG"].aug == "strong"


def test_freeze_strategy_dikenal_finetune():
    from src.cvl.finetune import STRATEGIES
    for sc in SCENARIOS.values():
        if sc.freeze_strategy is not None:
            assert sc.freeze_strategy in STRATEGIES
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/bin/pytest tests/test_scenarios.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.cvl.scenarios'`

- [ ] **Step 3: Implementasi `src/cvl/scenarios.py`**

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
```

- [ ] **Step 4: Jalankan test, pastikan lolos**

Run: `.venv/bin/pytest tests/test_scenarios.py -v`
Expected: PASS — semua test di berkas ini lolos

- [ ] **Step 5: Commit**

```bash
git add src/cvl/scenarios.py tests/test_scenarios.py
git commit -m "feat: add scenario registry for the ConvNeXt fine-tuning study"
```

---

### Task 3: Sumbu geometri — jendela sepanjang baris

Menyerang temuan §2 spec: `Resize(224)` pada sisi pendek + `RandomResizedCrop` dengan `ratio=(0.9,1.1)` tidak pernah terpenuhi, sehingga jatuh ke center-crop deterministik dan membuang 92,5% tiap baris.

**Files:**
- Modify: `src/cvl/dataset.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_dataset.py`

**Interfaces:**
- Consumes: —
- Produces: `dataset.ResizeHeight(height: int)` — transform PIL→PIL yang menyetel tinggi dan menjaga rasio. `dataset.build_transforms(train: bool, image_size: int = IMAGE_SIZE, geometry: str = "center", aug: str = "baseline")` — mengembalikan callable PIL→Tensor `[3,H,W]`. `dataset.LineDataset(manifest_subset, train, geometry="center", aug="baseline", eval_crops=1)`.

- [ ] **Step 1: Tambah fixture citra lebar**

Tambahkan ke `tests/conftest.py`:

```python
@pytest.fixture
def wide_line_image():
    """Citra baris dengan rasio ~12:1, meniru dimensi asli CVL (1739x137)."""
    return Image.new("RGB", (1740, 140), (200, 200, 200))
```

- [ ] **Step 2: Tulis test yang gagal**

Tambahkan ke `tests/test_dataset.py` (berkas ini belum mengimpor `pytest`; tambahkan `import pytest` di bagian atas karena test di bawah memakai `parametrize`):

```python
import pytest
from src.cvl.dataset import ResizeHeight


def test_resize_height_menjaga_rasio(wide_line_image):
    out = ResizeHeight(224)(wide_line_image)
    assert out.size[1] == 224
    # 1740/140 * 224 = 2784
    assert out.size[0] == 2784


def test_resize_height_tidak_pernah_lebih_sempit_dari_tinggi():
    from PIL import Image
    out = ResizeHeight(224)(Image.new("RGB", (10, 500)))
    assert out.size[0] >= 224 and out.size[1] == 224


def _pipeline_pra_refactor(train: bool, image_size: int = 224):
    """Salinan literal build_transforms sebelum refactor Task 3.

    Sengaja diduplikasi di dalam test: gunanya justru sebagai pembanding
    independen, supaya kesalahan di _geometry_stage/_aug_stage tidak ikut
    tercermin di sisi acuan.
    """
    import torchvision.transforms as T
    from src.cvl.config import IMAGENET_MEAN, IMAGENET_STD
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


@pytest.mark.parametrize("train", [True, False])
def test_geometri_center_identik_dengan_perilaku_lama(wide_line_image, train):
    """Penjaga utama: FT0 harus sama persis dengan pipeline sebelum refactor.

    Dibandingkan terhadap salinan literal di atas, bukan terhadap
    build_transforms itu sendiri — kalau keduanya memanggil implementasi yang
    sama, kesalahan di dalamnya akan tercermin di kedua sisi dan test lolos
    padahal pipeline sudah bergeser. Jalur train=True wajib ikut diuji karena
    di situlah RandomAffine/RandomResizedCrop/ColorJitter berada.
    """
    import torch
    torch.manual_seed(0)
    acuan = _pipeline_pra_refactor(train)(wide_line_image)
    torch.manual_seed(0)
    baru = build_transforms(train=train, geometry="center", aug="baseline")(wide_line_image)
    assert torch.equal(acuan, baru)
    assert baru.shape == (3, 224, 224)


def test_linewindow_menghasilkan_jendela_berbeda(wide_line_image):
    """RandomResizedCrop lama selalu mengembalikan potongan identik; ini
    memastikan geometri baru benar-benar mengacak posisi."""
    import torch
    t = build_transforms(train=True, geometry="linewindow")
    outs = []
    for seed in range(8):
        torch.manual_seed(seed)
        outs.append(t(wide_line_image))
    assert all(o.shape == (3, 224, 224) for o in outs)
    unik = {o.numpy().tobytes() for o in outs}
    assert len(unik) > 1, "jendela linewindow tidak pernah berpindah"


def test_linewindow_eval_bentuk_benar(wide_line_image):
    t = build_transforms(train=False, geometry="linewindow")
    assert t(wide_line_image).shape == (3, 224, 224)


def test_geometri_tidak_dikenal_ditolak(wide_line_image):
    import pytest
    with pytest.raises(ValueError):
        build_transforms(train=True, geometry="entahlah")


def test_aug_tidak_dikenal_ditolak(wide_line_image):
    import pytest
    with pytest.raises(ValueError):
        build_transforms(train=True, aug="entahlah")
```

- [ ] **Step 3: Jalankan test, pastikan gagal**

Run: `.venv/bin/pytest tests/test_dataset.py -v`
Expected: FAIL — `ImportError: cannot import name 'ResizeHeight'`

- [ ] **Step 4: Implementasi sumbu geometri**

Ganti seluruh isi `src/cvl/dataset.py` dengan:

```python
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


def _aug_stage(geometry: str, image_size: int, aug: str):
    """Tahap augmentasi (hanya dipakai saat train=True).

    Nilai "strong" ditambahkan pada task berikutnya; di sini `aug` sudah
    divalidasi supaya nilai salah ketik tidak lolos diam-diam.
    """
    if aug != "baseline":
        raise ValueError(f"aug tidak dikenal: {aug}")
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


def build_transforms(train: bool, image_size: int = IMAGE_SIZE,
                     geometry: str = "center", aug: str = "baseline"):
    """PIL -> Tensor [3, image_size, image_size].

    `geometry` mengatur bagian mana dari baris yang terlihat; `aug` mengatur
    seberapa keras citra diacak. Dua sumbu ini sengaja dipisah agar skenario
    FT1 dan AUG menguji mekanisme yang berbeda tanpa saling mencemari.
    """
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    steps = [T.Grayscale(num_output_channels=3)]
    steps += _geometry_stage(train, image_size, geometry)
    if train:
        steps += _aug_stage(geometry, image_size, aug)
    steps += [T.ToTensor(), norm]
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

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), int(r["label"])
```

Catatan untuk pelaksana: pada `geometry="center"` + `train=True`, urutannya tetap `Resize → RandomAffine → RandomResizedCrop → ColorJitter` persis seperti kode lama, sehingga `test_geometri_center_identik_dengan_perilaku_lama` lolos.

- [ ] **Step 5: Jalankan test, pastikan lolos**

Run: `.venv/bin/pytest tests/test_dataset.py -v`
Expected: PASS — semua test di berkas ini lolos

- [ ] **Step 6: Jalankan seluruh suite**

Run: `.venv/bin/pytest -q`
Expected: PASS — tidak ada kegagalan, jumlah test bertambah

- [ ] **Step 7: Commit**

```bash
git add src/cvl/dataset.py tests/conftest.py tests/test_dataset.py
git commit -m "feat: add linewindow geometry so the model sees the whole line"
```

---

### Task 4: Sumbu augmentasi kuat

**Files:**
- Modify: `src/cvl/dataset.py`
- Modify: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `build_transforms(train, image_size, geometry, aug)` dari Task 3.
- Produces: nilai `aug="strong"` yang valid pada `build_transforms` dan `LineDataset`.

- [ ] **Step 1: Tulis test yang gagal**

Tambahkan ke `tests/test_dataset.py`:

```python
def test_aug_strong_bentuk_benar(wide_line_image):
    import torch
    torch.manual_seed(0)
    out = build_transforms(train=True, aug="strong")(wide_line_image)
    assert out.shape == (3, 224, 224)


def test_aug_strong_benar_benar_mengacak(wide_line_image):
    """Crop pada AUG dipasang setelah strip tengah 224x246 supaya batasan
    rasio bisa dipenuhi — di geometri baseline crop selalu identik."""
    import torch
    t = build_transforms(train=True, aug="strong")
    outs = []
    for seed in range(8):
        torch.manual_seed(seed)
        outs.append(t(wide_line_image))
    assert len({o.numpy().tobytes() for o in outs}) > 1


def test_aug_strong_tidak_mengubah_geometri(wide_line_image):
    """AUG hanya boleh menaikkan kekuatan augmentasi, bukan memperluas
    bagian baris yang terlihat — kalau tidak, ia rancu dengan FT1."""
    from src.cvl.dataset import _strip_width
    assert _strip_width(224) == 246
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/bin/pytest tests/test_dataset.py -v`
Expected: FAIL — `ImportError: cannot import name '_strip_width'`

- [ ] **Step 3: Implementasi augmentasi kuat**

Di `src/cvl/dataset.py`, tambahkan helper sebelum `_aug_stage`:

```python
def _strip_width(image_size: int) -> int:
    """Lebar strip tengah yang dilihat pipeline baseline.

    Torchvision jatuh ke fallback `w = round(h * max(ratio))` ketika batasan
    RandomResizedCrop tidak terpenuhi. Dengan ratio maks 1.1 dan tinggi 224,
    lebarnya 246 — angka ini yang dipakai AUG supaya cakupan barisnya sama
    persis dengan baseline.
    """
    return int(round(image_size * 1.1))
```

Lalu ganti `_aug_stage` menjadi:

```python
def _aug_stage(geometry: str, image_size: int, aug: str):
    """Tahap augmentasi (hanya dipakai saat train=True)."""
    if aug == "baseline":
        if geometry == "center":
            return [
                T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
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
```

Pemanggilnya di `build_transforms` sudah meneruskan `aug` sejak Task 3, jadi tidak perlu diubah.

Sisipkan RandomErasing **setelah** `ToTensor` (ia bekerja pada tensor, bukan PIL):

```python
    steps += [T.ToTensor()]
    if train and aug == "strong":
        steps.append(T.RandomErasing(p=0.25))
    steps += [norm]
```

Catatan: `raise ValueError` untuk `aug` tak dikenal sudah ada sejak Task 3 dan harus tetap dipertahankan di cabang terakhir `_aug_stage`.

- [ ] **Step 4: Jalankan test, pastikan lolos**

Run: `.venv/bin/pytest tests/test_dataset.py -v`
Expected: PASS — semua test di berkas ini lolos

- [ ] **Step 5: Verifikasi ulang bahwa jalur baseline tidak bergeser**

Run: `.venv/bin/pytest tests/test_dataset.py::test_geometri_center_identik_dengan_perilaku_lama -v`
Expected: PASS

- [ ] **Step 6: Jalankan seluruh suite**

Run: `.venv/bin/pytest -q`
Expected: PASS — tidak ada kegagalan, jumlah test bertambah

- [ ] **Step 7: Commit**

```bash
git add src/cvl/dataset.py tests/test_dataset.py
git commit -m "feat: add strong augmentation axis with a crop that actually varies"
```

---

### Task 5: `drop_path` dan head ArcFace di lapisan model

**Files:**
- Create: `src/cvl/arcface.py`
- Create: `tests/test_arcface.py`
- Modify: `src/cvl/models.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: —
- Produces:
  - `arcface.ArcFaceHead(in_features: int, num_classes: int, s: float = 30.0, m: float = 0.3)` dengan `set_margin(m: float)` dan `forward(feats, labels=None) -> Tensor [B, num_classes]`.
  - `arcface.ArcFaceModel(backbone, head)` dengan `forward(x, labels=None)`, `forward_features(x)`.
  - `models.build_model(arch_key, num_classes, pretrained, drop_path: float = 0.0, head: str = "linear")`.
  - `models.set_arcface_margin(model, m: float)` — no-op bila model bukan ArcFace.

- [ ] **Step 1: Tulis test yang gagal untuk ArcFace**

Buat `tests/test_arcface.py`:

```python
import torch
from src.cvl.arcface import ArcFaceHead, ArcFaceModel


def test_head_bentuk_logit():
    h = ArcFaceHead(in_features=16, num_classes=5)
    out = h(torch.randn(4, 16))
    assert out.shape == (4, 5)


def test_head_margin_menurunkan_logit_kelas_benar():
    """Dengan label diberikan, logit kelas target dikurangi margin sudut."""
    torch.manual_seed(0)
    h = ArcFaceHead(in_features=16, num_classes=5, s=30.0, m=0.3)
    feats = torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 3])
    tanpa = h(feats)
    dengan = h(feats, labels)
    idx = torch.arange(4)
    assert (dengan[idx, labels] < tanpa[idx, labels]).all()


def test_set_margin_nol_menyamai_kosinus_polos():
    torch.manual_seed(0)
    h = ArcFaceHead(in_features=16, num_classes=5, s=30.0, m=0.3)
    h.set_margin(0.0)
    feats = torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 3])
    assert torch.allclose(h(feats, labels), h(feats), atol=1e-5)


def test_model_forward_tanpa_label():
    import timm
    backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
    m = ArcFaceModel(backbone, ArcFaceHead(backbone.num_features, 5)).eval()
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 5)


def test_model_forward_features_dua_dimensi():
    import timm
    backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
    m = ArcFaceModel(backbone, ArcFaceHead(backbone.num_features, 5)).eval()
    f = m.forward_features(torch.randn(2, 3, 224, 224))
    assert f.dim() == 2 and f.shape[0] == 2
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/bin/pytest tests/test_arcface.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.cvl.arcface'`

- [ ] **Step 3: Implementasi `src/cvl/arcface.py`**

```python
"""Head ArcFace (additive angular margin) untuk skenario FT4.

Writer identification pada L1 adalah 308 identitas dengan ~7 contoh per kelas
— rezim tempat head margin lazim dipakai. `s` dan `m` dipatok di muka pada
nilai standar papernya dan tidak disetel setelah melihat hasil.
"""
import math
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
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos)
        margin = torch.zeros_like(theta)
        margin.scatter_(1, labels.view(-1, 1), self.m)
        return self.s * torch.cos(theta + margin)


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

- [ ] **Step 4: Jalankan test, pastikan lolos**

Run: `.venv/bin/pytest tests/test_arcface.py -v`
Expected: PASS — semua test di berkas ini lolos

- [ ] **Step 5: Tulis test yang gagal untuk `build_model`**

Tambahkan ke `tests/test_models.py`:

```python
def test_drop_path_diteruskan():
    m = build_model("convnext_tiny", num_classes=7, pretrained=False, drop_path=0.2)
    rates = [mod.drop_prob for mod in m.modules()
             if mod.__class__.__name__ == "DropPath"]
    assert rates and max(rates) > 0.0


def test_drop_path_default_nol():
    m = build_model("convnext_tiny", num_classes=7, pretrained=False)
    rates = [mod.drop_prob for mod in m.modules()
             if mod.__class__.__name__ == "DropPath"]
    assert not rates or max(rates) == 0.0


def test_head_arcface_bentuk_logit():
    m = build_model("resnet50", num_classes=7, pretrained=False, head="arcface").eval()
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 7)


def test_forward_features_bekerja_pada_arcface():
    m = build_model("resnet50", num_classes=7, pretrained=False, head="arcface").eval()
    f = forward_features(m, torch.randn(2, 3, 224, 224))
    assert f.dim() == 2 and f.shape[0] == 2


def test_set_arcface_margin_aman_untuk_head_linear():
    from src.cvl.models import set_arcface_margin
    m = build_model("resnet50", num_classes=7, pretrained=False)
    set_arcface_margin(m, 0.1)  # tidak boleh melempar
```

- [ ] **Step 6: Jalankan test, pastikan gagal**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: FAIL — `TypeError: build_model() got an unexpected keyword argument 'drop_path'`

- [ ] **Step 7: Implementasi perubahan `src/cvl/models.py`**

Ganti seluruh isi `src/cvl/models.py` dengan:

```python
import timm
import torch
from .arcface import ArcFaceHead, ArcFaceModel
from .config import ALL_ARCHITECTURES


def build_model(arch_key: str, num_classes: int, pretrained: bool,
                drop_path: float = 0.0, head: str = "linear"):
    name = ALL_ARCHITECTURES[arch_key]
    if head == "linear":
        return timm.create_model(name, pretrained=pretrained,
                                 num_classes=num_classes,
                                 drop_path_rate=drop_path)
    if head == "arcface":
        backbone = timm.create_model(name, pretrained=pretrained,
                                     num_classes=0, drop_path_rate=drop_path)
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

- [ ] **Step 8: Jalankan test, pastikan lolos**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: PASS — semua test di berkas ini lolos

- [ ] **Step 9: Jalankan seluruh suite**

Run: `.venv/bin/pytest -q`
Expected: PASS — tidak ada kegagalan, jumlah test bertambah

- [ ] **Step 10: Commit**

```bash
git add src/cvl/arcface.py src/cvl/models.py tests/test_arcface.py tests/test_models.py
git commit -m "feat: wire drop_path through timm and add an ArcFace head"
```

---

### Task 6: `train_one_run` menerima `Scenario`

**Files:**
- Modify: `src/cvl/train.py`
- Modify: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `Scenario` (Task 2), `LineDataset(..., geometry, aug)` (Task 3-4), `build_model(..., drop_path, head)` dan `set_arcface_margin` (Task 5), `finetune.freeze_layers` / `finetune.build_param_groups` (sudah ada di `src/cvl/finetune.py`).
- Produces: `train.train_one_run(manifest, rc, out_dir, device, hp, scenario=None)` — `scenario=None` berarti `Scenario()`, yaitu perilaku sekarang.

- [ ] **Step 1: Tulis test yang gagal**

Tambahkan ke `tests/test_train_smoke.py`:

```python
from src.cvl.scenarios import Scenario, SCENARIOS


def _hp():
    return {"val_frac": 0.1, "num_workers": 0, "amp": False, "early_stop_patience": 1}


def _manifest(tiny_lines):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    return build_manifest(kept, n_train_pages=2, seed=0)


def test_train_dengan_scenario_default(tiny_lines, tmp_path):
    m = _manifest(tiny_lines)
    rc = RunConfig(arch="resnet50", level=2, mode="scratch", seed=0,
                   epochs=1, lr=1e-3, batch_size=8)
    out = train_one_run(m, rc, tmp_path, device="cpu", hp=_hp(), scenario=Scenario())
    assert (tmp_path / "best.pt").exists() and out["epochs_ran"] >= 1


def test_train_dengan_label_smoothing_dan_drop_path(tiny_lines, tmp_path):
    m = _manifest(tiny_lines)
    rc = RunConfig(arch="convnext_tiny", level=2, mode="scratch", seed=0,
                   epochs=1, lr=1e-3, batch_size=8)
    out = train_one_run(m, rc, tmp_path, device="cpu", hp=_hp(),
                        scenario=SCENARIOS["FT2"])
    assert out["epochs_ran"] >= 1


def test_train_dengan_freeze_strategy(tiny_lines, tmp_path):
    m = _manifest(tiny_lines)
    rc = RunConfig(arch="convnext_tiny", level=2, mode="pretrained", seed=0,
                   epochs=1, lr=1e-4, batch_size=8)
    out = train_one_run(m, rc, tmp_path, device="cpu", hp=_hp(),
                        scenario=SCENARIOS["FT3"])
    assert out["epochs_ran"] >= 1


def test_train_dengan_arcface(tiny_lines, tmp_path):
    m = _manifest(tiny_lines)
    rc = RunConfig(arch="resnet50", level=2, mode="scratch", seed=0,
                   epochs=1, lr=1e-3, batch_size=8)
    out = train_one_run(m, rc, tmp_path, device="cpu", hp=_hp(),
                        scenario=SCENARIOS["FT4"])
    assert (tmp_path / "best.pt").exists() and out["epochs_ran"] >= 1


def test_jadwal_margin_arcface():
    from src.cvl.train import arcface_margin_at
    # naik linear 0 -> m_target sepanjang epoch warmup, lalu tetap
    assert arcface_margin_at(epoch=0, warmup_epochs=3, m_target=0.3) == 0.0
    assert abs(arcface_margin_at(1, 3, 0.3) - 0.1) < 1e-9
    assert abs(arcface_margin_at(2, 3, 0.3) - 0.2) < 1e-9
    assert abs(arcface_margin_at(3, 3, 0.3) - 0.3) < 1e-9
    assert arcface_margin_at(10, 3, 0.3) == 0.3
    # tanpa warmup langsung penuh
    assert arcface_margin_at(0, 0, 0.3) == 0.3
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/bin/pytest tests/test_train_smoke.py -v`
Expected: FAIL — `TypeError: train_one_run() got an unexpected keyword argument 'scenario'`

- [ ] **Step 3: Tambahkan helper jadwal margin di `src/cvl/train.py`**

Sisipkan setelah fungsi `_num_classes`:

```python
def arcface_margin_at(epoch: int, warmup_epochs: int, m_target: float) -> float:
    """Margin ArcFace dinaikkan linear 0 -> m_target sepanjang epoch warmup.

    Tanpa ini ArcFace sering gagal konvergen di epoch awal karena head-nya
    diinisialisasi acak sementara margin sudah penuh.
    """
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return m_target
    return m_target * epoch / warmup_epochs
```

- [ ] **Step 4: Sambungkan `Scenario` ke `train_one_run`**

Di `src/cvl/train.py`, tambahkan import:

```python
from .models import build_model, set_arcface_margin
from .finetune import freeze_layers, build_param_groups
from .scenarios import Scenario
```

Ubah tanda tangan dan bagian setup (menggantikan baris 27-36 saat ini):

```python
def train_one_run(manifest, rc: RunConfig, out_dir, device, hp: dict,
                  scenario: Scenario | None = None) -> dict:
    sc = scenario or Scenario()
    _seed_all(rc.seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = LineDataset(manifest[manifest.split == "train"], train=True,
                           geometry=sc.geometry, aug=sc.aug)
    val_ds = LineDataset(manifest[manifest.split == "val"], train=False,
                         geometry=sc.geometry, aug=sc.aug)
    nw = hp.get("num_workers", 0)
    tl = DataLoader(train_ds, batch_size=rc.batch_size, shuffle=True, num_workers=nw)
    vl = DataLoader(val_ds, batch_size=rc.batch_size, shuffle=False, num_workers=nw)
    model = build_model(rc.arch, _num_classes(manifest),
                        pretrained=(rc.mode == "pretrained"),
                        drop_path=sc.drop_path, head=sc.head).to(device)
    if sc.freeze_strategy is not None:
        freeze_layers(model, sc.freeze_strategy)
        opt = torch.optim.AdamW(
            build_param_groups(model, sc.freeze_strategy, base_lr=rc.lr),
            weight_decay=rc.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=rc.lr,
                                weight_decay=rc.weight_decay)
```

Ubah pembuatan loss (baris 49 saat ini) menjadi:

```python
    crit = torch.nn.CrossEntropyLoss(label_smoothing=sc.label_smoothing)
```

Di awal tiap epoch (tepat setelah `for epoch in range(rc.epochs):`), setel margin:

```python
        set_arcface_margin(model, arcface_margin_at(epoch, warmup_epochs, 0.3))
```

Ubah forward saat latih agar label ikut diteruskan ketika head ArcFace dipakai (menggantikan baris 63-64 saat ini):

```python
            with torch.amp.autocast(amp_device, enabled=use_amp):
                logits = model(x, y) if sc.head == "arcface" else model(x)
                loss = crit(logits, y)
```

Forward saat validasi tetap tanpa label (`model(x)`), sehingga margin tidak ikut menghukum skor validasi.

- [ ] **Step 5: Jalankan test, pastikan lolos**

Run: `.venv/bin/pytest tests/test_train_smoke.py -v`
Expected: PASS — semua test di berkas ini lolos

- [ ] **Step 6: Jalankan seluruh suite**

Run: `.venv/bin/pytest -q`
Expected: PASS — tidak ada kegagalan, jumlah test bertambah

- [ ] **Step 7: Commit**

```bash
git add src/cvl/train.py tests/test_train_smoke.py
git commit -m "feat: thread Scenario through training with ArcFace margin warmup"
```

---

### Task 7: Evaluasi multi-crop

**Files:**
- Modify: `src/cvl/dataset.py`
- Modify: `src/cvl/evaluate.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_evaluate_smoke.py`

**Interfaces:**
- Consumes: `LineDataset(..., eval_crops)` (Task 3), `Scenario` (Task 2), `build_model(..., head)` (Task 5).
- Produces: `evaluate.evaluate_checkpoint(ckpt_path, manifest, arch, device, batch_size=64, scenario=None)`. `LineDataset.__getitem__` mengembalikan `[K,3,H,W]` saat `eval_crops > 1`.

- [ ] **Step 1: Tulis test yang gagal untuk dataset multi-crop**

Tambahkan ke `tests/test_dataset.py`:

```python
def test_eval_crops_menghasilkan_tumpukan(tiny_lines):
    from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    test = m[m.split == "test"]
    ds = LineDataset(test, train=False, geometry="linewindow", eval_crops=9)
    x, y = ds[0]
    assert x.shape == (9, 3, 224, 224) and isinstance(y, int)


def test_eval_crops_satu_tetap_tiga_dimensi(tiny_lines):
    from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    ds = LineDataset(m[m.split == "test"], train=False, eval_crops=1)
    x, _ = ds[0]
    assert x.shape == (3, 224, 224)


def test_jendela_evaluasi_merata_dan_berbeda():
    """Jendela harus benar-benar berasal dari posisi berbeda dan menutup
    baris dari ujung ke ujung. Diuji dengan citra bergradien — pada citra
    polos semua potongan identik sehingga uji isi tidak membuktikan apa pun."""
    from PIL import Image
    from src.cvl.dataset import even_windows
    grad = Image.new("L", (1740, 140))
    grad.putdata([x % 256 for _ in range(140) for x in range(1740)])
    grad = grad.convert("RGB")

    wins = even_windows(grad, size=224, k=9)
    assert len(wins) == 9
    assert all(w.size == (224, 140) for w in wins)
    # kesembilan jendela berbeda isinya
    assert len({w.tobytes() for w in wins}) == 9
    # jendela pertama mulai di kiri, terakhir berakhir di kanan
    assert wins[0].tobytes() == grad.crop((0, 0, 224, 140)).tobytes()
    assert wins[-1].tobytes() == grad.crop((1740 - 224, 0, 1740, 140)).tobytes()
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/bin/pytest tests/test_dataset.py -v`
Expected: FAIL — `ImportError: cannot import name 'even_windows'`

- [ ] **Step 3: Implementasi multi-crop di `src/cvl/dataset.py`**

Tambahkan helper setelah `ResizeHeight`:

```python
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
```

Ganti `LineDataset.__getitem__` menjadi:

```python
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        label = int(r["label"])
        if self.train or self.eval_crops <= 1:
            return self.tf(img), label
        base = ResizeHeight(IMAGE_SIZE)(img.convert("RGB"))
        wins = even_windows(base, IMAGE_SIZE, self.eval_crops)
        return torch.stack([self.crop_tf(w) for w in wins]), label
```

Tambahkan `import torch` di bagian atas berkas, dan di `__init__` siapkan transform per-jendela:

```python
        # transform untuk satu jendela yang sudah dipotong (dipakai saat eval_crops > 1)
        self.crop_tf = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
```

- [ ] **Step 4: Jalankan test, pastikan lolos**

Run: `.venv/bin/pytest tests/test_dataset.py -v`
Expected: PASS — semua test di berkas ini lolos

- [ ] **Step 5: Tulis test yang gagal untuk evaluasi multi-crop**

Tambahkan ke `tests/test_evaluate_smoke.py`:

```python
from src.cvl.scenarios import Scenario, SCENARIOS


def test_evaluate_multicrop(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    sc = Scenario(geometry="linewindow", eval_crops=4)
    rc = RunConfig(arch="resnet50", level=2, mode="scratch", seed=0,
                   epochs=1, batch_size=8)
    train_one_run(m, rc, tmp_path, device="cpu",
                  hp={"num_workers": 0, "amp": False, "early_stop_patience": 1},
                  scenario=sc)
    res = evaluate_checkpoint(tmp_path / "best.pt", m, arch="resnet50",
                              device="cpu", batch_size=8, scenario=sc)
    assert 0.0 <= res["top1_page"] <= 1.0
    assert "map_line" in res


def test_evaluate_arcface(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    sc = SCENARIOS["FT4"]
    rc = RunConfig(arch="resnet50", level=2, mode="scratch", seed=0,
                   epochs=1, batch_size=8)
    train_one_run(m, rc, tmp_path, device="cpu",
                  hp={"num_workers": 0, "amp": False, "early_stop_patience": 1},
                  scenario=sc)
    res = evaluate_checkpoint(tmp_path / "best.pt", m, arch="resnet50",
                              device="cpu", batch_size=8, scenario=sc)
    assert 0.0 <= res["top1_page"] <= 1.0
```

- [ ] **Step 6: Jalankan test, pastikan gagal**

Run: `.venv/bin/pytest tests/test_evaluate_smoke.py -v`
Expected: FAIL — `TypeError: evaluate_checkpoint() got an unexpected keyword argument 'scenario'`

- [ ] **Step 7: Implementasi perubahan `src/cvl/evaluate.py`**

Ganti tanda tangan dan blok inferensi (baris 13-27 saat ini) menjadi:

```python
def evaluate_checkpoint(ckpt_path, manifest, arch, device, batch_size: int = 64,
                        scenario=None) -> dict:
    from .scenarios import Scenario
    sc = scenario or Scenario()
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    model = build_model(arch, _num_classes(manifest), pretrained=False,
                        drop_path=sc.drop_path, head=sc.head).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    ds = LineDataset(test, train=False, geometry=sc.geometry, aug=sc.aug,
                     eval_crops=sc.eval_crops)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, feats = [], []
    t0, n_img = time.time(), 0
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            if x.dim() == 5:
                # [B, K, 3, H, W] -> forward semua jendela, rata-ratakan per baris
                b, k = x.shape[0], x.shape[1]
                flat = x.reshape(b * k, *x.shape[2:])
                n_img += flat.shape[0]
                p = torch.softmax(model(flat), dim=1).reshape(b, k, -1).mean(dim=1)
                f = forward_features(model, flat).reshape(b, k, -1).mean(dim=1)
            else:
                n_img += len(x)
                p = torch.softmax(model(x), dim=1)
                f = forward_features(model, x)
            probs.append(p.cpu().numpy())
            feats.append(f.cpu().numpy())
    throughput = n_img / max(1e-6, time.time() - t0)
```

Sisa fungsi (mulai dari `probs = np.concatenate(probs)`) tidak berubah.

- [ ] **Step 8: Jalankan test, pastikan lolos**

Run: `.venv/bin/pytest tests/test_evaluate_smoke.py -v`
Expected: PASS — semua test di berkas ini lolos

- [ ] **Step 9: Jalankan seluruh suite**

Run: `.venv/bin/pytest -q`
Expected: PASS — tidak ada kegagalan, jumlah test bertambah

- [ ] **Step 10: Commit**

```bash
git add src/cvl/dataset.py src/cvl/evaluate.py tests/test_dataset.py tests/test_evaluate_smoke.py
git commit -m "feat: average predictions over evenly spaced line windows at eval"
```

---

### Task 8: Runner skenario + CLI

**Files:**
- Create: `src/cvl/run_scenarios.py`
- Create: `scripts/run_scenarios.py`
- Create: `tests/test_run_scenarios.py`

**Interfaces:**
- Consumes: `SCENARIOS` (Task 2), `train_one_run(..., scenario)` (Task 6), `evaluate_checkpoint(..., scenario)` (Task 7), `env_metadata` (Task 1), `run_experiments.already_done` dan `_append_row` (sudah ada).
- Produces: `run_scenarios.scenario_run_id(name: str, seed: int) -> str` dan `run_scenarios.run_scenario_grid(manifest, names, seeds, results_csv, ckpt_root, device, hp, arch="convnext_tiny", level=1) -> None`.

- [ ] **Step 1: Tulis test yang gagal**

Buat `tests/test_run_scenarios.py`:

```python
import pandas as pd
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.run_scenarios import scenario_run_id, run_scenario_grid


def test_scenario_run_id():
    assert scenario_run_id("FT1", 3) == "FT1_s3"
    assert scenario_run_id("AUG", 0) == "AUG_s0"


def test_run_scenario_grid_dan_resume(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    csv = tmp_path / "results-finetune.csv"
    hp = {"num_workers": 0, "amp": False, "early_stop_patience": 1,
          "pretrained_epochs": 1, "scratch_epochs": 1, "batch_size": 8, "lr": 1e-3}
    run_scenario_grid(m, names=["FT2"], seeds=[0], results_csv=csv,
                      ckpt_root=tmp_path / "ck", device="cpu", hp=hp,
                      arch="convnext_tiny", level=2)
    d = pd.read_csv(csv)
    assert len(d) == 1
    assert d.iloc[0]["scenario"] == "FT2"
    assert d.iloc[0]["run_id"] == "FT2_s0"
    assert d.iloc[0]["gpu_name"] == "cpu"
    # jalankan lagi -> resume, tidak menambah baris
    run_scenario_grid(m, names=["FT2"], seeds=[0], results_csv=csv,
                      ckpt_root=tmp_path / "ck", device="cpu", hp=hp,
                      arch="convnext_tiny", level=2)
    assert len(pd.read_csv(csv)) == 1


def test_ft0_dilewati(tiny_lines, tmp_path):
    """FT0 disalin dari results-pretrained.csv, tidak pernah dijalankan."""
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    csv = tmp_path / "results-finetune.csv"
    hp = {"num_workers": 0, "amp": False, "early_stop_patience": 1,
          "pretrained_epochs": 1, "scratch_epochs": 1, "batch_size": 8, "lr": 1e-3}
    run_scenario_grid(m, names=["FT0"], seeds=[0], results_csv=csv,
                      ckpt_root=tmp_path / "ck", device="cpu", hp=hp,
                      arch="convnext_tiny", level=2)
    assert not csv.exists()
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/bin/pytest tests/test_run_scenarios.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.cvl.run_scenarios'`

- [ ] **Step 3: Implementasi `src/cvl/run_scenarios.py`**

```python
"""Runner Studi 2: enam skenario fine-tuning ConvNeXt-Tiny di L1.

Meniru pola `run_grid`: resume-able per berkas CSV, satu baris per
(skenario, seed). FT0 sengaja dilewati — barisnya disalin dari
results-pretrained.csv karena konfigurasinya identik dengan grid utama.
"""
from pathlib import Path
from .env_info import env_metadata
from .evaluate import evaluate_checkpoint
from .run_experiments import already_done, _append_row
from .scenarios import SCENARIOS
from .train import RunConfig, train_one_run


def scenario_run_id(name: str, seed: int) -> str:
    return f"{name}_s{seed}"


def run_scenario_grid(manifest, names, seeds, results_csv, ckpt_root, device, hp,
                      arch: str = "convnext_tiny", level=1) -> None:
    ckpt_root = Path(ckpt_root)
    for name in names:
        if name == "FT0":
            print("skip FT0 (baseline disalin dari results-pretrained.csv)")
            continue
        sc = SCENARIOS[name]
        for seed in seeds:
            rid = scenario_run_id(name, seed)
            if already_done(results_csv, rid):
                print(f"skip {rid}"); continue
            rc = RunConfig(arch=arch, level=level, mode="pretrained", seed=seed,
                           epochs=hp["pretrained_epochs"], lr=hp["lr"],
                           batch_size=hp["batch_size"],
                           weight_decay=hp.get("weight_decay", 0.05))
            out_dir = ckpt_root / rid
            tr = train_one_run(manifest, rc, out_dir, device, hp, scenario=sc)
            ev = evaluate_checkpoint(out_dir / "best.pt", manifest, arch, device,
                                     batch_size=hp["batch_size"], scenario=sc)
            _append_row(results_csv, {"run_id": rid, "scenario": name, "arch": arch,
                "level": level, "mode": "pretrained", "seed": seed,
                **tr, **ev, **env_metadata(device)})
            print(f"done {rid}: top1={ev['top1_page']:.3f} map={ev['map_line']:.3f}")
```

- [ ] **Step 4: Jalankan test, pastikan lolos**

Run: `.venv/bin/pytest tests/test_run_scenarios.py -v`
Expected: PASS — semua test di berkas ini lolos

- [ ] **Step 5: Implementasi CLI `scripts/run_scenarios.py`**

```python
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cvl.config import SEEDS, PRETRAINED_EPOCHS, BATCH_SIZE, add_date_args, date_suffix
from src.cvl.run_experiments import LR_OVERRIDES
from src.cvl.run_scenarios import run_scenario_grid
from src.cvl.scenarios import SCENARIOS

ARCH = "convnext_tiny"
LEVEL = 1


def parse_args():
    p = argparse.ArgumentParser(description="Studi 2: skenario fine-tuning ConvNeXt-Tiny di L1")
    add_date_args(p)
    p.add_argument("--scenarios", default=None, metavar="NAMA",
                   help="daftar skenario dipisah koma (default: semua kecuali FT0)")
    p.add_argument("--results", default=None, metavar="PATH")
    p.add_argument("--ckpt-root", default=None, metavar="DIR")
    return p.parse_args()


def main():
    args = parse_args()
    suffix = date_suffix(args.date)
    results_csv = Path(args.results) if args.results else Path(f"results/results{suffix}.csv")
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else Path(f"results/checkpoints{suffix}")

    names = ([s.strip() for s in args.scenarios.split(",")] if args.scenarios
             else [n for n in SCENARIOS if n != "FT0"])
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"skenario tidak dikenal: {unknown}")

    hp = yaml.safe_load(open("configs/default.yaml"))
    if PRETRAINED_EPOCHS is not None:
        hp["pretrained_epochs"] = PRETRAINED_EPOCHS
    if BATCH_SIZE is not None:
        hp["batch_size"] = BATCH_SIZE
    # samakan dengan grid utama: ConvNeXt pretrained memakai LR 1e-4
    hp["lr"] = LR_OVERRIDES.get((ARCH, "pretrained"), hp["lr"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    man_dir = Path("results/manifests")
    # gagal cepat kalau ada manifest seed yang belum dibangun, sebelum
    # pekerjaan panjang dimulai
    hilang = [s for s in SEEDS if not (man_dir / f"seed{s}_L{LEVEL}.parquet").exists()]
    if hilang:
        raise SystemExit(f"manifest belum ada untuk seed {hilang} — jalankan "
                         f"scripts/prep_manifests.py dulu")
    print(f"skenario: {names} | seeds={SEEDS} | arch={ARCH} L{LEVEL} | device={device}")
    print(f"output: {results_csv} | ckpt: {ckpt_root}")

    for seed in SEEDS:
        m = pd.read_parquet(man_dir / f"seed{seed}_L{LEVEL}.parquet")
        run_scenario_grid(m, names=names, seeds=[seed], results_csv=results_csv,
                          ckpt_root=ckpt_root, device=device, hp=hp,
                          arch=ARCH, level=LEVEL)


if __name__ == "__main__":
    main()
```

Catatan: manifest dibaca ulang per seed karena split L1 berbeda untuk tiap seed — jangan dibaca sekali di luar loop.

- [ ] **Step 6: Verifikasi CLI menolak skenario tak dikenal**

Run: `.venv/bin/python scripts/run_scenarios.py --scenarios TIDAKADA --date uji`
Expected: keluar dengan pesan `skenario tidak dikenal: ['TIDAKADA']`

- [ ] **Step 7: Jalankan seluruh suite**

Run: `.venv/bin/pytest -q`
Expected: PASS — tidak ada kegagalan, jumlah test bertambah

- [ ] **Step 8: Commit**

```bash
git add src/cvl/run_scenarios.py scripts/run_scenarios.py tests/test_run_scenarios.py
git commit -m "feat: add scenario runner and CLI for the fine-tuning study"
```

---

### Task 9: Perbarui status implementasi di README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: seluruh task sebelumnya.
- Produces: —

- [ ] **Step 1: Hapus blok status implementasi**

Di `README.md`, hapus blockquote yang diawali `> **Status implementasi.**` beserta isinya, karena ketiga hal yang disebutkan di sana sudah selesai.

- [ ] **Step 2: Perbarui daftar struktur**

Di blok kode `## Struktur`, ubah baris `scripts/` menjadi:

```
scripts/          # entry-point: prep_manifests.py, run_all.py, run_scenarios.py,
                  #              make_report.py
```

dan baris `src/cvl/` menjadi:

```
src/cvl/          # pipeline: data_prep, dataset, models, arcface, metrics, train,
                  #           evaluate, run_experiments, run_scenarios, scenarios,
                  #           finetune, env_info, report
```

- [ ] **Step 3: Perbarui jumlah test**

Ganti `.venv/bin/pytest -q      # 26 test` menjadi `.venv/bin/pytest -q      # 73 test`, dan baris `tests/` di blok struktur menjadi `tests/            # pytest, 73 test, jalan di CPU tanpa dataset`.

- [ ] **Step 4: Jalankan suite untuk memastikan angkanya benar**

Run: `.venv/bin/pytest -q`
Expected: PASS — tidak ada kegagalan, jumlah test bertambah — bila jumlahnya berbeda, pakai angka sebenarnya di Step 3.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: drop the implementation-status caveat now that both groups are built"
```

---

## Catatan pelaksanaan

**Jangan jalankan Task 2-9 sebelum Studi 1 dimulai.** Task 1 sendiri sudah cukup untuk melepas 41 jam GPU; sisanya dikerjakan sementara grid berjalan. Semakin sedikit kode yang bergerak selama eksperimen panjang, semakin sedikit yang bisa membatalkannya.

**Jumlah test yang tercantum di tiap task bersifat kumulatif dan merupakan perkiraan.** Bila angkanya meleset, yang penting adalah tidak ada test yang gagal dan jumlahnya bertambah, bukan cocok persis.

**Penjaga terpenting di seluruh rencana ini** adalah `test_geometri_center_identik_dengan_perilaku_lama` (Task 3, Step 2). Kalau test itu gagal, `FT0` tidak lagi sah disalin dari `results-pretrained.csv` dan seluruh Studi 2 kehilangan baseline-nya. Jangan longgarkan test itu untuk membuat task lain lolos.

Test itu wajib membandingkan terhadap `_pipeline_pra_refactor` — salinan literal pipeline lama — bukan terhadap `build_transforms` dengan argumen berbeda. Membandingkan implementasi dengan dirinya sendiri hanya menguji bahwa argumen default setara dengan argumen eksplisit; kesalahan di dalam `_geometry_stage` akan tercermin di kedua sisi dan lolos. Jalur `train=True` wajib ikut diuji: di situlah `RandomAffine`, `RandomResizedCrop`, dan `ColorJitter` berada, dan justru itu yang berisiko bergeser.

**§6 spec (protokol statistik) sengaja tidak punya task di sini.** Spec §5.2 menyatakan `src/cvl/report.py` tidak disentuh, jadi aturan kolaps, uji-t berpasangan, dan interval kepercayaan dikerjakan manual saat menulis bab hasil — bukan dibangun sebagai kode. Konsekuensinya `make_report.py` akan tetap menghitung rata-rata gabungan yang mencampur run kolaps dan run sehat. **Angka mode scratch dari laporan otomatis tidak boleh langsung dikutip ke skripsi**; pisahkan kolaps dulu sesuai aturan di §6. Kalau nanti Anda ingin ini otomatis, itu perubahan spec tersendiri dan rencana tersendiri.
