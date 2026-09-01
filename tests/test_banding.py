"""Perbandingan berpasangan FT0 → FT5 → FT1.

Ketiga kondisi ada di CSV berbeda (FT0 di results-pretrained.csv, FT5 di
results-evalonly-*.csv, FT1 di results-finetune-*.csv) tapi berbagi pola
run_id yang sama, jadi penyejajarannya lewat seed. Uji-t-nya berpasangan per
seed — sama seperti tabel signifikansi di dokumentasi/07 — karena kelima seed
memakai split yang sama, sehingga variasi antar-seed bisa dibuang.
"""
import numpy as np
import pandas as pd
import pytest

from src.cvl.banding import baca_kondisi, sejajarkan, uji_t, porsi


def _csv(tmp_path, nama, run_ids, top1):
    p = tmp_path / nama
    pd.DataFrame({"run_id": run_ids, "top1_page": top1,
                  "macro_f1_page": [t - 0.05 for t in top1]}).to_csv(p, index=False)
    return p


def test_baca_kondisi_menyaring_arch_level_slot_dan_mengindeks_seed(tmp_path):
    p = _csv(tmp_path, "r.csv",
             ["swin_tiny_L1_pretrained_s0", "swin_tiny_L1_pretrained_s1",
              "swin_tiny_L4_pretrained_s0", "resnet50_L1_pretrained_s0",
              "swin_tiny_L1_FT1_s0"],
             [0.80, 0.81, 0.95, 0.70, 0.95])
    d = baca_kondisi(p, arch="swin_tiny", level=1, slot="pretrained")
    assert list(d.index) == [0, 1]
    assert d.loc[0, "top1_page"] == 0.80


def test_baca_kondisi_gagal_kalau_tidak_ada_yang_cocok(tmp_path):
    p = _csv(tmp_path, "r.csv", ["swin_tiny_L1_pretrained_s0"], [0.80])
    with pytest.raises(SystemExit, match="FT5"):
        baca_kondisi(p, arch="swin_tiny", level=1, slot="FT5")


def test_baca_kondisi_gagal_kalau_seed_ganda(tmp_path):
    """scripts/cek_csv.py sudah mendokumentasikan bahwa run rebutan bisa
    menulis run_id ganda. Kalau itu lolos ke sini, rata-ratanya diam-diam
    membobot satu seed dua kali."""
    p = _csv(tmp_path, "r.csv",
             ["swin_tiny_L1_pretrained_s0", "swin_tiny_L1_pretrained_s0"],
             [0.80, 0.88])
    with pytest.raises(SystemExit, match="ganda"):
        baca_kondisi(p, arch="swin_tiny", level=1, slot="pretrained")


def test_sejajarkan_gagal_kalau_himpunan_seed_beda():
    a = pd.DataFrame({"top1_page": [0.8, 0.8, 0.8]}, index=[0, 1, 2])
    b = pd.DataFrame({"top1_page": [0.9, 0.9]}, index=[0, 1])
    with pytest.raises(SystemExit, match="seed"):
        sejajarkan({"FT0": a, "FT5": b})


def test_sejajarkan_mengembalikan_seed_yang_sama_terurut():
    a = pd.DataFrame({"top1_page": [0.8, 0.8]}, index=[1, 0])
    b = pd.DataFrame({"top1_page": [0.9, 0.9]}, index=[0, 1])
    hasil = sejajarkan({"FT0": a, "FT5": b})
    assert list(hasil["FT0"].index) == [0, 1] == list(hasil["FT5"].index)


def test_uji_t_berpasangan_nilai_diketahui():
    """d = [0,10 0,12 0,08 0,11 0,09] -> mean 0,10; sd(ddof=1) 0,0158114;
    se 0,0070711; t = 14,142136. Dihitung terpisah, bukan dari implementasinya."""
    a = pd.Series([0.80, 0.80, 0.80, 0.80, 0.80], index=range(5))
    b = pd.Series([0.90, 0.92, 0.88, 0.91, 0.89], index=range(5))
    h = uji_t(b, a)
    assert h["n"] == 5
    assert h["delta_pp"] == pytest.approx(10.0)
    assert h["t"] == pytest.approx(14.142136, rel=1e-5)
    assert h["p"] < 0.05


def test_uji_t_selisih_kecil_tidak_signifikan():
    """t = 1,414214 (< ambang 2,776 untuk df=4), jadi p harus di atas 5%."""
    a = pd.Series([0.80] * 5, index=range(5))
    b = pd.Series([0.81, 0.83, 0.79, 0.82, 0.80], index=range(5))
    h = uji_t(b, a)
    assert h["t"] == pytest.approx(1.414214, rel=1e-5)
    assert h["p"] > 0.05


def test_uji_t_arah_selisih():
    a = pd.Series([0.90, 0.91, 0.89, 0.92, 0.88], index=range(5))
    b = pd.Series([0.80] * 5, index=range(5))
    assert uji_t(b, a)["delta_pp"] < 0


def test_uji_t_menolak_seed_tidak_sejajar():
    a = pd.Series([0.80] * 5, index=range(5))
    b = pd.Series([0.90] * 5, index=range(1, 6))
    with pytest.raises(SystemExit, match="seed"):
        uji_t(b, a)


def test_porsi_kenaikan_yang_dijelaskan_tta():
    """Angka inti yang diminta dosen: berapa bagian dari kenaikan FT1 atas FT0
    yang sudah dijelaskan oleh 9-crop saja."""
    assert porsi(ft0=0.80, ft5=0.93, ft1=0.95) == pytest.approx(0.13 / 0.15)


def test_porsi_nan_kalau_ft1_tidak_lebih_baik():
    """Tanpa penyebut yang berarti, "berapa persen dari kenaikan" tidak punya
    arti — lebih baik NaN daripada angka raksasa yang terlihat sah."""
    assert np.isnan(porsi(ft0=0.80, ft5=0.93, ft1=0.80))


# --------------------------------------------------------------------------
# CLI (bisa diuji langsung: modul ini tidak menarik torch/timm)
# --------------------------------------------------------------------------

import subprocess
import sys as _sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "banding_ft5.py"
METRIK = ["top1_page", "top5_page", "macro_f1_page", "map_line", "top1_retrieval"]


def _csv_lengkap(path, slot, top1, arch="swin_tiny", level=1):
    n = len(top1)
    d = {"run_id": [f"{arch}_L{level}_{slot}_s{i}" for i in range(n)]}
    for m in METRIK:
        d[m] = top1
    pd.DataFrame(d).to_csv(path, index=False)
    return path


def test_cli_menghitung_tiga_perbandingan(tmp_path):
    """FT0 0,80 -> FT5 0,93 -> FT1 0,95: 9-crop menjelaskan 86,7% kenaikan."""
    ft0 = _csv_lengkap(tmp_path / "a.csv", "pretrained", [0.80, 0.81, 0.79, 0.80, 0.80])
    ft5 = _csv_lengkap(tmp_path / "b.csv", "FT5", [0.93, 0.94, 0.92, 0.93, 0.93])
    ft1 = _csv_lengkap(tmp_path / "c.csv", "FT1", [0.95, 0.96, 0.94, 0.95, 0.95])

    r = subprocess.run(
        [_sys.executable, str(CLI), "--ft0", str(ft0), "--ft5", str(ft5),
         "--ft1", str(ft1), "--arch", "swin_tiny", "--level", "1"],
        capture_output=True, text=True, cwd=ROOT)

    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "FT5 - FT0" in out and "FT1 - FT5" in out and "FT1 - FT0" in out
    assert "86,7%" in out or "86.7%" in out


def test_cli_gagal_jelas_kalau_csv_ft5_belum_ada(tmp_path):
    ft0 = _csv_lengkap(tmp_path / "a.csv", "pretrained", [0.80] * 5)
    ft1 = _csv_lengkap(tmp_path / "c.csv", "FT1", [0.95] * 5)
    r = subprocess.run(
        [_sys.executable, str(CLI), "--ft0", str(ft0), "--ft5", str(tmp_path / "hilang.csv"),
         "--ft1", str(ft1), "--arch", "swin_tiny", "--level", "1"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "hilang.csv" in (r.stdout + r.stderr)


def test_cli_tidak_menarik_torch():
    """Skrip analisis harus bisa dijalankan di laptop tanpa GPU stack.

    Diperiksa lewat AST, bukan pencarian teks: kata "torch" muncul di
    docstring skripnya, jadi pencarian teks akan gagal untuk alasan yang salah
    dan memaksa dokumentasinya diubah demi menyenangkan test.
    """
    import ast
    modul = set()
    for n in ast.walk(ast.parse(CLI.read_text(), filename=str(CLI))):
        if isinstance(n, ast.Import):
            modul |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            modul.add(n.module.split(".")[0])
    assert not modul & {"torch", "timm", "torchvision"}
