import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.preflight as pf


def test_jalan_tanpa_gpu_tanpa_melempar(capsys, monkeypatch, tmp_path):
    """Pra-terbang harus tetap memberi laporan yang berguna di mesin tanpa CUDA,
    bukan melempar — kalau tidak, ia justru menghalangi diagnosis."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    pf.main()
    keluar = capsys.readouterr().out
    assert "== versi ==" in keluar
    assert "== GPU ==" in keluar
    assert "forward+backward" in keluar
    assert "Selesai" in keluar


def test_memperingatkan_num_workers_yang_tak_sesuai_vcpu(capsys, monkeypatch):
    monkeypatch.setattr(pf.os, "cpu_count", lambda: 12)
    pf.cek_cpu_dan_shm(num_workers=8, prefetch=2, batch=64)
    keluar = capsys.readouterr().out
    assert "saran num_workers = 10" in keluar
    assert "!!" in keluar          # 8 != 10 -> harus ditandai


def test_diam_saat_num_workers_sudah_pas(capsys, monkeypatch):
    monkeypatch.setattr(pf.os, "cpu_count", lambda: 12)
    pf.cek_cpu_dan_shm(num_workers=10, prefetch=2, batch=64)
    keluar = capsys.readouterr().out
    assert "saran num_workers = 10" in keluar
    assert "sesuaikan ke" not in keluar


def test_hitungan_shm_ikut_prefetch_dan_worker(capsys, monkeypatch):
    """Angka yang dilaporkan harus benar-benar dihitung, bukan konstanta."""
    monkeypatch.setattr(pf.os, "cpu_count", lambda: 12)
    pf.cek_cpu_dan_shm(num_workers=10, prefetch=2, batch=64)
    kecil = capsys.readouterr().out
    pf.cek_cpu_dan_shm(num_workers=10, prefetch=4, batch=64)
    besar = capsys.readouterr().out
    # 2 loader x 10 worker x 2 x 39MB = 1.5 GB ; prefetch 4 -> 3.1 GB
    assert "1.5 GB di shared memory" in kecil
    assert "3.1 GB di shared memory" in besar


def test_vcpu_pakai_kuota_cgroup_v2(tmp_path, monkeypatch):
    """Kuota kontainer, bukan core host — pod 12 vCPU di host 48 core harus
    melaporkan 12, kalau tidak num_workers akan disetel jauh terlalu tinggi."""
    cg = tmp_path / "cpu.max"
    cg.write_text("1200000 100000\n")          # 12 CPU
    monkeypatch.setattr(pf, "Path", lambda p: cg if p == "/sys/fs/cgroup/cpu.max" else Path(p))
    n, sumber = pf.vcpu_efektif()
    assert n == 12 and "cgroup v2" in sumber


def test_vcpu_abaikan_cgroup_tanpa_kuota(tmp_path, monkeypatch):
    """'max' berarti tanpa batas — jangan diartikan sebagai 1 CPU."""
    cg = tmp_path / "cpu.max"
    cg.write_text("max 100000\n")
    monkeypatch.setattr(pf, "Path", lambda p: cg if p == "/sys/fs/cgroup/cpu.max" else Path(p))
    n, sumber = pf.vcpu_efektif()
    assert n >= 1 and "cgroup v2" not in sumber


def test_vcpu_melaporkan_sumbernya():
    n, sumber = pf.vcpu_efektif()
    assert n >= 1 and isinstance(sumber, str) and sumber
