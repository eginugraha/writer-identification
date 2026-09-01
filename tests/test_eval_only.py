"""FT5: 9-crop averaging saat inferensi di atas bobot FT0, tanpa latih ulang.

FT1 mengubah geometri latih *dan* protokol uji sekaligus (lihat
dokumentasi/04-skenario-fine-tuning.md), jadi +14,2 poin top1_page-nya tidak
bisa dibagi antara keduanya. FT5 menjalankan protokol uji FT1 di atas
checkpoint FT0 yang sudah ada: selisih FT5-FT0 adalah efek murni test-time
ensemble, sisanya milik sliding-window training.

Karena tidak ada pelatihan, jalurnya beda dari `run_scenario_grid`: checkpoint
dibaca dari folder grid utama ({arch}_L{level}_pretrained_s{seed}), bukan dari
folder Studi 2, dan baris hasilnya tidak punya kolom latih.
"""
import ast
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.dataset import LineDataset
from src.cvl.run_experiments import _append_row
from src.cvl.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "eval_only.py"
ARCH = "resnet50"
HP = {"num_workers": 0, "batch_size": 8}


def _manifest(tiny_lines):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    return build_manifest(kept, n_train_pages=2, seed=0)


def _ckpt_palsu(m, path: Path, arch: str = ARCH):
    """Bobot acak yang disimpan seperti best.pt — cukup untuk menguji jalur
    eval-only tanpa membayar biaya latih."""
    from src.cvl.models import build_model
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(arch, int(m["label"].max()) + 1, pretrained=False)
    torch.save(model.state_dict(), path)
    return path


# --------------------------------------------------------------------------
# protokol uji
# --------------------------------------------------------------------------

def test_protokol_uji_ft5_identik_dengan_ft1(tiny_lines):
    """Inti klaim FT5: yang berbeda dari FT1 hanya bobotnya, bukan cara ujinya.

    LineDataset mengabaikan `geometry` begitu eval_crops > 1, jadi FT5
    (geometry="center") dan FT1 (geometry="linewindow") menghasilkan sembilan
    jendela yang sama persis. Kalau sifat itu hilang, selisih FT5-FT0 tidak
    lagi sebanding dengan FT1 dan seluruh ablasinya batal.
    """
    m = _manifest(tiny_lines)
    test = m[m.split == "test"]
    ft5 = LineDataset(test, train=False, geometry=SCENARIOS["FT5"].geometry,
                      eval_crops=SCENARIOS["FT5"].eval_crops)
    ft1 = LineDataset(test, train=False, geometry=SCENARIOS["FT1"].geometry,
                      eval_crops=SCENARIOS["FT1"].eval_crops)
    x5, y5 = ft5[0]
    x1, y1 = ft1[0]
    assert x5.shape == (9, 3, 224, 224)
    assert torch.equal(x5, x1) and y5 == y1


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def test_run_eval_only_menulis_baris_dari_checkpoint_grid(tiny_lines, tmp_path):
    from src.cvl.run_scenarios import run_eval_only
    m = _manifest(tiny_lines)
    src = tmp_path / "ck"
    _ckpt_palsu(m, src / f"{ARCH}_L2_pretrained_s0" / "best.pt")
    csv = tmp_path / "results-evalonly.csv"

    run_eval_only(m, name="FT5", seeds=[0], results_csv=csv, src_ckpt_root=src,
                  device="cpu", hp=HP, arch=ARCH, level=2)

    d = pd.read_csv(csv)
    assert len(d) == 1
    row = d.iloc[0]
    assert row["run_id"] == f"{ARCH}_L2_FT5_s0"
    assert row["scenario"] == "FT5"
    assert row["source_run_id"] == f"{ARCH}_L2_pretrained_s0"
    assert row["eval_crops"] == 9
    assert 0.0 <= row["top1_page"] <= 1.0
    assert row["gpu_name"] == "cpu"


def test_run_eval_only_tidak_melatih_apa_pun(tiny_lines, tmp_path):
    """Klaim "tanpa pelatihan ulang" harus terlihat di artefaknya: tidak ada
    kolom latih, dan checkpoint sumbernya tidak tersentuh."""
    from src.cvl.run_scenarios import run_eval_only
    m = _manifest(tiny_lines)
    src = tmp_path / "ck"
    ckpt = _ckpt_palsu(m, src / f"{ARCH}_L2_pretrained_s0" / "best.pt")
    sebelum = ckpt.read_bytes()
    csv = tmp_path / "results-evalonly.csv"

    run_eval_only(m, name="FT5", seeds=[0], results_csv=csv, src_ckpt_root=src,
                  device="cpu", hp=HP, arch=ARCH, level=2)

    kolom = set(pd.read_csv(csv).columns)
    assert not kolom & {"train_time_s", "epochs_ran", "best_val_acc", "lr"}
    assert ckpt.read_bytes() == sebelum
    assert [p.name for p in src.iterdir()] == [f"{ARCH}_L2_pretrained_s0"]


def test_run_eval_only_gagal_kalau_checkpoint_hilang(tiny_lines, tmp_path):
    """Checkpoint grid utama tidak ikut git (results/checkpoints*/ ada di
    .gitignore) dan pod GPU bisa sudah dihapus. Kalau hilang, harus berhenti
    dengan menyebut path-nya — bukan menulis baris kosong."""
    from src.cvl.run_scenarios import run_eval_only
    m = _manifest(tiny_lines)
    csv = tmp_path / "results-evalonly.csv"
    with pytest.raises(SystemExit, match=f"{ARCH}_L2_pretrained_s0"):
        run_eval_only(m, name="FT5", seeds=[0], results_csv=csv,
                      src_ckpt_root=tmp_path / "ck", device="cpu", hp=HP,
                      arch=ARCH, level=2)
    assert not csv.exists()


def test_run_eval_only_resume(tiny_lines, tmp_path):
    from src.cvl.run_scenarios import run_eval_only
    m = _manifest(tiny_lines)
    src = tmp_path / "ck"
    _ckpt_palsu(m, src / f"{ARCH}_L2_pretrained_s0" / "best.pt")
    csv = tmp_path / "results-evalonly.csv"
    for _ in range(2):
        run_eval_only(m, name="FT5", seeds=[0], results_csv=csv,
                      src_ckpt_root=src, device="cpu", hp=HP, arch=ARCH, level=2)
    assert len(pd.read_csv(csv)) == 1


def test_run_eval_only_menolak_csv_berkolom_lain(tiny_lines, tmp_path):
    """_append_row menulis header hanya saat berkas belum ada. Menyalurkan
    baris eval-only ke results-finetune-swin.csv karena itu tidak akan error —
    nilainya akan masuk ke bawah header lain dan diam-diam salah kolom."""
    from src.cvl.run_scenarios import run_eval_only
    m = _manifest(tiny_lines)
    src = tmp_path / "ck"
    _ckpt_palsu(m, src / f"{ARCH}_L2_pretrained_s0" / "best.pt")
    csv = tmp_path / "results-finetune.csv"
    _append_row(csv, {"run_id": f"{ARCH}_L2_FT2_s0", "scenario": "FT2",
                      "lr": 1e-4, "train_time_s": 1.0, "top1_page": 0.5})

    with pytest.raises(SystemExit, match="kolom"):
        run_eval_only(m, name="FT5", seeds=[0], results_csv=csv,
                      src_ckpt_root=src, device="cpu", hp=HP, arch=ARCH, level=2)
    assert len(pd.read_csv(csv)) == 1


# --------------------------------------------------------------------------
# CLI (berbasis AST: scripts/ menarik torch/timm saat diimpor)
# --------------------------------------------------------------------------

def _opsi_cli():
    opsi = set()
    for n in ast.walk(ast.parse(CLI.read_text(), filename=str(CLI))):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"):
            for a in n.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    opsi.add(a.value)
    return opsi


def test_cli_ada():
    assert CLI.exists()


def test_cli_punya_opsi_yang_dibutuhkan():
    assert {"--arch", "--level", "--scenario", "--source", "--src-ckpt-root",
            "--results"} <= _opsi_cli()


# --------------------------------------------------------------------------
# FT5 tidak boleh masuk jalur latih
# --------------------------------------------------------------------------

def test_skenario_latih_tidak_memuat_ft0_dan_ft5():
    """scripts/run_scenarios.py default-nya "semua kecuali FT0". Begitu FT5
    masuk registry, default itu akan melatih FT5 — persis yang tidak boleh
    terjadi, karena seluruh nilainya justru terletak pada tidak adanya latih."""
    from src.cvl.scenarios import skenario_latih
    n = skenario_latih()
    assert "FT0" not in n and "FT5" not in n
    assert set(n) == {"FT1", "FT2", "FT3", "FT4", "AUG"}


def test_run_scenario_grid_melewati_ft5(tiny_lines, tmp_path):
    """Kalau FT5 tetap disebut eksplisit ke runner latih, ia harus dilewati —
    bukan dilatih diam-diam dengan bobot baru yang bukan FT0."""
    from src.cvl.run_scenarios import run_scenario_grid
    m = _manifest(tiny_lines)
    csv = tmp_path / "results-finetune.csv"
    hp = {"num_workers": 0, "amp": False, "early_stop_patience": 1,
          "pretrained_epochs": 1, "scratch_epochs": 1, "batch_size": 8, "lr": 1e-3}
    run_scenario_grid(m, names=["FT5"], seeds=[0], results_csv=csv,
                      ckpt_root=tmp_path / "ck", device="cpu", hp=hp,
                      arch=ARCH, level=2)
    assert not csv.exists()


def test_cli_run_scenarios_memakai_skenario_latih():
    """Default daftar skenario di CLI harus datang dari skenario_latih(),
    bukan dari filter "!= FT0" yang ditulis ulang di scripts/."""
    cli = ROOT / "scripts" / "run_scenarios.py"
    sumber = cli.read_text()
    assert "skenario_latih" in sumber
    assert '!= "FT0"' not in sumber


def test_cli_arch_tidak_ikut_filter_env():
    """`--arch` di sini menamai checkpoint yang sudah ada, bukan grid yang akan
    dijalankan, jadi pilihannya harus datang dari katalog penuh. `ARCHITECTURES`
    disaring `CVL_ARCHS` di .env (src/cvl/config.py:95-96) — memakainya membuat
    `--arch swin_tiny` ditolak di mesin yang .env-nya menyebut arsitektur lain,
    padahal bobotnya ada di disk."""
    sumber = CLI.read_text()
    assert "ALL_ARCHITECTURES" in sumber
    for n in ast.walk(ast.parse(sumber, filename=str(CLI))):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"
                and any(isinstance(a, ast.Constant) and a.value == "--arch" for a in n.args)):
            dipakai = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
            assert "ARCHITECTURES" not in dipakai
            assert "ALL_ARCHITECTURES" in dipakai
            break
    else:
        raise AssertionError("--arch tidak ditemukan")
