"""Studi 2 harus bisa memilih arsitektur, tanpa hasil antar-arsitektur bertabrakan.

`scenario_run_id` lama menghasilkan "FT1_s0" — tanpa arsitektur. Dua arsitektur
yang menulis ke CSV yang sama membuat `already_done` menganggap run arsitektur
kedua sudah selesai, lalu melewatinya tanpa suara; folder checkpoint-nya pun
saling menimpa.

Tes CLI di sini berbasis AST, sama seperti tests/test_cli_entrypoint.py, karena
scripts/run_scenarios.py menarik torch/timm saat diimpor.
"""
import ast
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "run_scenarios.py"


def _pohon():
    return ast.parse(CLI.read_text(), filename=str(CLI))


def _opsi_cli():
    """Kumpulkan nama opsi yang didaftarkan lewat p.add_argument("--x", ...)."""
    opsi = set()
    for n in ast.walk(_pohon()):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"):
            for a in n.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    opsi.add(a.value)
    return opsi


# --------------------------------------------------------------------------
# run_id
# --------------------------------------------------------------------------

def test_run_id_memuat_arsitektur_dan_level():
    from src.cvl.scenarios import scenario_run_id
    assert scenario_run_id("FT1", 0, arch="swin_tiny", level=1) == "swin_tiny_L1_FT1_s0"


def test_run_id_dua_arsitektur_tidak_bertabrakan():
    from src.cvl.scenarios import scenario_run_id
    a = scenario_run_id("FT1", 0, arch="swin_tiny", level=1)
    b = scenario_run_id("FT1", 0, arch="convnext_tiny", level=1)
    assert a != b


def test_run_id_mengikuti_pola_grid_utama():
    """Grid utama memakai {arch}_L{level}_{mode}_s{seed}; Studi 2 menaruh nama
    skenario di posisi mode supaya kedua CSV bisa dibaca dengan pola yang sama."""
    from src.cvl.scenarios import scenario_run_id
    rid = scenario_run_id("AUG", 3, arch="swin_tiny", level=4)
    arch, level, skenario, seed = rid.rsplit("_", 3)
    assert (arch, level, skenario, seed) == ("swin_tiny", "L4", "AUG", "s3")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("opsi", ["--arch", "--level"])
def test_cli_punya_opsi(opsi):
    assert opsi in _opsi_cli(), f"{CLI.name}: opsi {opsi} belum didaftarkan"


def _add_argument(nama_opsi):
    """Node ast.Call untuk p.add_argument("<nama_opsi>", ...)."""
    for n in ast.walk(_pohon()):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument" and n.args
                and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == nama_opsi):
            return n
    return None


def test_arch_wajib_disebut_eksplisit():
    """Studi 2 dijalankan di swin_tiny, tapi katalognya memuat lebih dari satu
    arsitektur. Default apa pun membuat satu flag yang terlupa menghabiskan
    ~5 jam GPU di arsitektur yang salah — dan hasilnya baru ketahuan salah
    setelah CSV-nya dibaca."""
    node = _add_argument("--arch")
    assert node is not None, "opsi --arch belum didaftarkan"
    kw = {k.arg: k.value for k in node.keywords}
    assert "default" not in kw, "--arch punya default; harus disebut eksplisit"
    assert isinstance(kw.get("required"), ast.Constant) and kw["required"].value is True, \
        "--arch harus required=True"


def test_arsitektur_tidak_lagi_dipaku_sebagai_konstanta():
    """ARCH = "convnext_tiny" di level modul berarti flag-nya tidak dipakai."""
    for n in _pohon().body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in {"ARCH", "LEVEL"}:
                    pytest.fail(f"{CLI.name}: {t.id} masih dipaku di level modul")


def test_arch_divalidasi_terhadap_peta_lapisan():
    """Salah ketik --arch harus ditolak sebelum jam-jam latihan dimulai, bukan
    setelah model terlanjur dibuat."""
    sumber = CLI.read_text()
    assert "LAYER_MAP" in sumber, (
        f"{CLI.name}: --arch tidak divalidasi terhadap LAYER_MAP — arsitektur "
        "tanpa peta lapisan akan gagal jauh di dalam run")
