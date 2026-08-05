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
ALL_SEEDS = [0, 1, 2]
ALL_MODES = ["pretrained", "scratch"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def date_suffix(date_arg) -> str:
    """Sufiks penamaan output run/laporan. Dipakai bersama oleh
    scripts/run_all.py dan scripts/make_report.py supaya nama CSV, folder
    checkpoint, laporan, dan figure dari satu run pakai stempel yang sama.

    None -> ''  (nama kanonik, perilaku lama)
    'auto' -> '-2026-08-05'  (tanggal lokal hari ini; hormati env TZ)
    '<tag>' -> '-<tag>'
    """
    if date_arg is None:
        return ""
    tag = datetime.now().strftime("%Y-%m-%d") if date_arg == "auto" else date_arg
    return f"-{tag}"


def add_date_args(parser) -> None:
    """Flag --date/-nya seragam di semua entry-point."""
    parser.add_argument(
        "--date", nargs="?", const="auto", default=None, metavar="TAG",
        help="stempel tanggal pada nama output supaya run lama tidak ketimpa. "
             "Tanpa nilai = YYYY-MM-DD hari ini; boleh diisi tag sendiri, "
             "mis. --date 2026-08-05-rerun-warmup.")


def lines_root() -> Path:
    return project_root() / "cvl-database-1-1"


def _load_dotenv() -> None:
    """Loader .env minimal (tanpa dependency). Var yang sudah ada di
    environment tidak ditimpa, jadi `CVL_ARCHS=... python ...` tetap menang."""
    path = project_root() / ".env"
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def _env_list(name: str, default: list) -> list | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_level(tok: str):
    return None if tok.lower() in ("full", "none") else int(tok)


def _env_int(name: str):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


_load_dotenv()

# --- Seleksi grid (di-override lewat .env / environment) ---
_arch_keys = _env_list("CVL_ARCHS", list(ALL_ARCHITECTURES.keys()))
ARCHITECTURES = {k: ALL_ARCHITECTURES[k] for k in _arch_keys if k in ALL_ARCHITECTURES}

_level_toks = _env_list("CVL_LEVELS", None)
ABLATION_LEVELS = (
    [_parse_level(t) for t in _level_toks] if _level_toks is not None else list(ALL_ABLATION_LEVELS)
)

_seed_toks = _env_list("CVL_SEEDS", None)
SEEDS = [int(s) for s in _seed_toks] if _seed_toks is not None else list(ALL_SEEDS)

MODES = _env_list("CVL_MODES", list(ALL_MODES))

# Batasi jumlah penulis (None = semua). Berguna buat smoke test cepat.
MAX_WRITERS = _env_int("CVL_MAX_WRITERS")

# Override hyperparameter opsional (None = pakai configs/default.yaml)
PRETRAINED_EPOCHS = _env_int("CVL_PRETRAINED_EPOCHS")
SCRATCH_EPOCHS = _env_int("CVL_SCRATCH_EPOCHS")
BATCH_SIZE = _env_int("CVL_BATCH_SIZE")
