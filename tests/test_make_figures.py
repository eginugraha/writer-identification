"""make_figures.py harus mengikuti data, bukan angka yang dipaku.

Tiga hal yang ketinggalan zaman sejak grid naik ke 5 seed dan hasilnya dipecah
per server:

  * membaca results/results.csv — berkas yang sudah tidak ada sejak pembagian
    dua server, jadi skripnya tidak bisa jalan sama sekali;
  * label "(kolaps n/3)" dan judul "3 seed" dipaku, sama seperti bug kalimat
    penutup di make_report.py — hanya pindah berkas;
  * plt.xlim(0.80, 0.93) dipaku, sehingga arsitektur di luar rentang itu
    hilang dari grafik tanpa peringatan.

Fungsinya mengembalikan apa yang digambar (label, nilai, batas sumbu) supaya
bisa diperiksa tanpa membongkar PNG.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.make_figures as mf

ARCHS = ["resnet50", "convnext_tiny", "efficientnetv2_s", "vit_small", "swin_tiny"]


def _df(mode, seeds=5, kolaps=(), top1=None):
    baris = []
    for arch in ARCHS:
        for level in (1, 2, 3, 4):
            for seed in range(seeds):
                mati = level == 4 and seed < dict(kolaps).get(arch, 0)
                nilai = 0.002 if mati else (top1 or {}).get(arch, 0.5 + 0.05 * level)
                baris.append(dict(run_id=f"{arch}_L{level}_{mode}_s{seed}", arch=arch,
                                  level=level, mode=mode, seed=seed, top1_page=nilai))
    return pd.DataFrame(baris)


# --------------------------------------------------------------------------
# penyebut seed dari data
# --------------------------------------------------------------------------

def test_label_kolaps_memakai_jumlah_seed_sebenarnya(tmp_path):
    hasil = mf.scratch_trainability_bar(
        _df("scratch", seeds=5, kolaps=[("swin_tiny", 4)]), tmp_path / "a.png")
    label = " ".join(hasil["labels"])
    assert "4/5" in label
    assert "/3" not in label


def test_label_kolaps_ikut_saat_seed_lain(tmp_path):
    hasil = mf.scratch_trainability_bar(
        _df("scratch", seeds=3, kolaps=[("swin_tiny", 2)]), tmp_path / "a.png")
    assert "2/3" in " ".join(hasil["labels"])


def test_judul_sumbu_leaderboard_menyebut_jumlah_seed_sebenarnya(tmp_path):
    hasil = mf.leaderboard_pretrained(_df("pretrained", seeds=5), tmp_path / "b.png")
    assert "5 seed" in hasil["xlabel"]
    assert "3 seed" not in hasil["xlabel"]


# --------------------------------------------------------------------------
# batas sumbu dihitung, bukan dipaku
# --------------------------------------------------------------------------

def test_batas_x_leaderboard_memuat_semua_batang(tmp_path):
    """xlim(0.80, 0.93) yang dipaku memotong arsitektur di luar rentang itu."""
    rendah = {"resnet50": 0.62, "swin_tiny": 0.97}
    hasil = mf.leaderboard_pretrained(_df("pretrained", top1=rendah), tmp_path / "b.png")
    lo, hi = hasil["lim"]
    assert lo <= min(hasil["means"]), f"batang terendah {min(hasil['means'])} di luar {lo}"
    assert hi >= max(hasil["means"]), f"batang tertinggi {max(hasil['means'])} di luar {hi}"


def test_batas_y_trainability_memuat_semua_batang(tmp_path):
    hasil = mf.scratch_trainability_bar(
        _df("scratch", top1={"vit_small": 0.99}), tmp_path / "a.png")
    lo, hi = hasil["lim"]
    assert lo <= min(hasil["means"]) and hi >= max(hasil["means"])


# --------------------------------------------------------------------------
# urutan & nama arsitektur dari data
# --------------------------------------------------------------------------

def test_arsitektur_yang_tidak_ada_di_data_tidak_bikin_error(tmp_path):
    df = _df("scratch", kolaps=[("swin_tiny", 2)])
    df = df[df.arch != "resnet50"]
    hasil = mf.scratch_trainability_bar(df, tmp_path / "a.png")
    assert len(hasil["labels"]) == 4
    assert not any("ResNet" in l for l in hasil["labels"])


def test_arsitektur_di_luar_daftar_nama_tampil_apa_adanya(tmp_path):
    df = _df("pretrained")
    df.loc[df.arch == "resnet50", "arch"] = "arsitektur_baru"
    hasil = mf.leaderboard_pretrained(df, tmp_path / "b.png")
    assert "arsitektur_baru" in " ".join(hasil["labels"])


# --------------------------------------------------------------------------
# sumber CSV: berkas terpisah per server
# --------------------------------------------------------------------------

def _jalankan(tmp_path, monkeypatch, berkas):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    argv = ["make_figures.py"]
    for nama, df in berkas.items():
        p = tmp_path / "results" / nama
        df.to_csv(p, index=False)
        argv += ["--results", str(p)]
    monkeypatch.setattr(sys, "argv", argv)
    mf.main()
    return sorted(p.name for p in (tmp_path / "results" / "figures").glob("*.png"))


def test_membaca_dua_csv_terpisah(tmp_path, monkeypatch):
    png = _jalankan(tmp_path, monkeypatch, {
        "results-pretrained.csv": _df("pretrained"),
        "results-scratch.csv": _df("scratch", kolaps=[("swin_tiny", 4)]),
    })
    assert png == ["leaderboard_pretrained.png", "scratch_trainability.png"]


def test_csv_scratch_saja_melewati_leaderboard(tmp_path, monkeypatch):
    png = _jalankan(tmp_path, monkeypatch,
                    {"results-scratch.csv": _df("scratch", kolaps=[("swin_tiny", 4)])})
    assert png == ["scratch_trainability.png"]


def test_csv_pretrained_saja_melewati_trainability(tmp_path, monkeypatch):
    png = _jalankan(tmp_path, monkeypatch, {"results-pretrained.csv": _df("pretrained")})
    assert png == ["leaderboard_pretrained.png"]


def test_csv_hilang_gagal_cepat(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_figures.py", "--results", "results/tiada.csv"])
    with pytest.raises(SystemExit, match="tiada.csv"):
        mf.main()
