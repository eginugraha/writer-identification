import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cvl.run_experiments import kunci_eksklusif
from scripts.cek_csv import periksa


def test_lock_dilepas_setelah_selesai(tmp_path):
    csv = tmp_path / "r.csv"
    with kunci_eksklusif(csv):
        assert Path(str(csv) + ".lock").exists()
    assert not Path(str(csv) + ".lock").exists()


def test_lock_dilepas_walau_ada_exception(tmp_path):
    csv = tmp_path / "r.csv"
    with pytest.raises(ValueError):
        with kunci_eksklusif(csv):
            raise ValueError("gagal di tengah")
    assert not Path(str(csv) + ".lock").exists(), "lock tertinggal -> run berikutnya terblokir"


def test_menolak_proses_kedua_yang_hidup(tmp_path):
    """Inti fiturnya: proses kedua harus ditolak, bukan ikut menulis."""
    csv = tmp_path / "r.csv"
    Path(str(csv) + ".lock").write_text(str(os.getpid()))   # PID kita sendiri = hidup
    with pytest.raises(SystemExit) as e:
        with kunci_eksklusif(csv):
            pass
    assert str(os.getpid()) in str(e.value)


def test_lock_basi_diabaikan(tmp_path, capsys):
    """PID mati tidak boleh memblokir selamanya — pod yang di-kill meninggalkannya."""
    csv = tmp_path / "r.csv"
    mati = subprocess.Popen([sys.executable, "-c", "pass"]); mati.wait()
    Path(str(csv) + ".lock").write_text(str(mati.pid))
    with kunci_eksklusif(csv):
        pass
    assert "lock basi" in capsys.readouterr().out


def test_cek_csv_menemukan_header_nyasar_dan_ganda():
    df = pd.DataFrame([
        {"run_id": "a_s0", "top1_page": 0.5},
        {"run_id": "run_id", "top1_page": "top1_page"},   # header nyasar
        {"run_id": "b_s0", "top1_page": 0.6},
        {"run_id": "b_s0", "top1_page": 0.61},            # ganda: ditulis 2 proses
    ])
    nyasar, ganda = periksa(df)
    assert len(nyasar) == 1
    assert ganda == ["b_s0"]


def test_cek_csv_bersih_tidak_melaporkan_apa_apa():
    df = pd.DataFrame([{"run_id": "a_s0"}, {"run_id": "b_s0"}])
    nyasar, ganda = periksa(df)
    assert nyasar.empty and ganda == []
