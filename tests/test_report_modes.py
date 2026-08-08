"""Laporan harus mengikuti isi CSV-nya, bukan asumsi tetap.

make_report.py ditulis untuk satu CSV yang memuat kedua mode. Pembagian dua
server memecahnya jadi results-scratch.csv dan results-pretrained.csv, dan
skripnya tidak ikut menyesuaikan:

  * bagian pretrained selalu diminta, jadi CSV scratch-only menghasilkan
    empat tabel kosong dan satu figure bersumbu kosong;
  * nama figure memaku kata "pretrained" (acc_vs_n_pretrained-scratch.png);
  * kalimat penutup soal kolaps dipaku dari grid 3-seed yang lama, sehingga
    laporan 5-seed menutup dengan "kolaps 3/3" yang bertentangan dengan
    tabelnya sendiri.

Yang terakhir paling berbahaya: dua yang pertama menghasilkan berkas yang
jelas terlihat rusak, yang ketiga menghasilkan berkas rapi yang angkanya salah.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cvl.report import ringkasan_kolaps
import scripts.make_report as mr

ARCHS = ["resnet50", "convnext_tiny", "efficientnetv2_s", "vit_small", "swin_tiny"]


def _df(mode, seeds=5, kolaps=()):
    """Grid tiruan. `kolaps` = {arch: jumlah seed yang kolaps di L4}."""
    baris = []
    for arch in ARCHS:
        for level in (1, 2, 3, 4):
            for seed in range(seeds):
                mati = level == 4 and seed < dict(kolaps).get(arch, 0)
                baris.append(dict(
                    run_id=f"{arch}_L{level}_{mode}_s{seed}", arch=arch, level=level,
                    mode=mode, seed=seed, top1_page=0.002 if mati else 0.5 + 0.05 * level,
                    top5_page=0.01 if mati else 0.8, macro_f1_page=0.001 if mati else 0.4,
                    map_line=0.2, top1_retrieval=0.3, n_params=2e7,
                    throughput_img_s=160.0, train_time_s=700.0, epochs_ran=40))
    return pd.DataFrame(baris)


# --------------------------------------------------------------------------
# kalimat kolaps dibangkitkan dari data
# --------------------------------------------------------------------------

def test_ringkasan_kolaps_memakai_hitungan_sebenarnya():
    teks = ringkasan_kolaps(_df("scratch", kolaps=[("swin_tiny", 3), ("convnext_tiny", 2)]))
    assert "swin_tiny 3/5" in teks
    assert "convnext_tiny 2/5" in teks


def test_ringkasan_kolaps_menyebut_penyebut_seed_yang_benar():
    """Grid 3-seed lama tidak boleh menghasilkan '/5', dan sebaliknya."""
    teks = ringkasan_kolaps(_df("scratch", seeds=3, kolaps=[("swin_tiny", 3)]))
    assert "swin_tiny 3/3" in teks
    assert "/5" not in teks


def test_ringkasan_kolaps_menyebut_yang_tidak_pernah_kolaps():
    teks = ringkasan_kolaps(_df("scratch", kolaps=[("swin_tiny", 2)]))
    for arch in ("resnet50", "efficientnetv2_s", "vit_small"):
        assert arch in teks


def test_ringkasan_kolaps_tidak_mengarang_saat_semua_sehat():
    teks = ringkasan_kolaps(_df("scratch"))
    assert "0/5" in teks or "tidak ada" in teks.lower()
    assert "3/3" not in teks


# --------------------------------------------------------------------------
# laporan hanya memuat mode yang ada di CSV
# --------------------------------------------------------------------------

def _jalankan(tmp_path, monkeypatch, df, tag):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    csv = tmp_path / "results" / f"results-{tag}.csv"
    df.to_csv(csv, index=False)
    monkeypatch.setattr(sys, "argv",
                        ["make_report.py", "--results", str(csv), "--date", tag])
    mr.main()
    return (tmp_path / "dokumentasi" / f"08-hasil-eksperimen-{tag}.md").read_text()


def test_csv_scratch_tidak_menghasilkan_bagian_pretrained(tmp_path, monkeypatch):
    md = _jalankan(tmp_path, monkeypatch, _df("scratch", kolaps=[("swin_tiny", 3)]), "scratch")
    assert "Mode: pretrained" not in md
    assert "Mode: scratch" in md


def test_csv_pretrained_tidak_menghasilkan_bagian_scratch(tmp_path, monkeypatch):
    md = _jalankan(tmp_path, monkeypatch, _df("pretrained"), "pretrained")
    assert "Mode: scratch" not in md
    assert "Mode: pretrained" in md


def test_csv_gabungan_memuat_keduanya(tmp_path, monkeypatch):
    df = pd.concat([_df("pretrained"), _df("scratch", kolaps=[("swin_tiny", 2)])])
    md = _jalankan(tmp_path, monkeypatch, df, "gabungan")
    assert "Mode: pretrained" in md and "Mode: scratch" in md


def test_tidak_ada_tabel_kosong(tmp_path, monkeypatch):
    """Kerangka tabel tanpa baris ('| arch |  |') berarti bagian itu tidak
    punya data dan seharusnya tidak ditulis sama sekali."""
    md = _jalankan(tmp_path, monkeypatch, _df("scratch", kolaps=[("swin_tiny", 3)]), "scratch")
    assert "| arch |  |" not in md


def test_figure_dinamai_menurut_mode_yang_digambar(tmp_path, monkeypatch):
    _jalankan(tmp_path, monkeypatch, _df("pretrained"), "pretrained")
    fig = tmp_path / "results" / "figures"
    nama = sorted(p.name for p in fig.glob("*.png"))
    assert nama == ["acc_vs_n_pretrained-pretrained.png"], nama


def test_csv_scratch_tidak_menulis_figure_pretrained(tmp_path, monkeypatch):
    _jalankan(tmp_path, monkeypatch, _df("scratch", kolaps=[("swin_tiny", 3)]), "scratch")
    nama = sorted(p.name for p in (tmp_path / "results" / "figures").glob("*.png"))
    assert not any("pretrained" in n for n in nama), nama


def test_kalimat_penutup_konsisten_dengan_tabelnya(tmp_path, monkeypatch):
    """Inti bug-nya: laporan tidak boleh menutup dengan angka yang tidak ada
    di tabel di atasnya."""
    md = _jalankan(tmp_path, monkeypatch,
                   _df("scratch", kolaps=[("swin_tiny", 3), ("convnext_tiny", 2)]), "scratch")
    assert "3/3" not in md, "kalimat 3-seed lama masih terpaku di laporan 5-seed"
    assert "swin_tiny 3/5" in md and "convnext_tiny 2/5" in md


def test_csv_tanpa_mode_dikenal_gagal_cepat(tmp_path, monkeypatch):
    df = _df("scratch")
    df["mode"] = "entah"
    with pytest.raises(SystemExit, match="mode"):
        _jalankan(tmp_path, monkeypatch, df, "aneh")
