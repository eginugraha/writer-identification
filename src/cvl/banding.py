"""Perbandingan berpasangan antar kondisi Studi 2 (FT0 / FT5 / FT1).

Ketiga kondisi hidup di CSV berbeda — FT0 di results-pretrained.csv, FT5 di
results-evalonly-*.csv, FT1 di results-finetune-*.csv — tapi run_id-nya
berpola sama (`{arch}_L{level}_{slot}_s{seed}`), jadi penyejajarannya lewat
seed. Uji-t dibuat berpasangan karena kelima seed memakai split yang sama:
selisih per seed membuang variasi antar-split yang tidak menarik.
"""
from pathlib import Path
import pandas as pd
from scipy import stats


def baca_kondisi(csv_path, arch: str, level, slot: str) -> pd.DataFrame:
    """Baris satu kondisi dari sebuah CSV hasil, terindeks seed.

    `slot` adalah token di posisi mode pada run_id: "pretrained" untuk baris
    grid utama, atau nama skenario ("FT1", "FT5") untuk Studi 2. Menyaring
    lewat run_id, bukan kolom `scenario`, karena results-pretrained.csv tidak
    punya kolom itu sama sekali.
    """
    p = Path(csv_path)
    if not p.exists():
        raise SystemExit(f"berkas hasil tidak ada: {p}")
    d = pd.read_csv(p)
    pola = rf"^{arch}_L{level}_{slot}_s(\d+)$"
    cocok = d["run_id"].astype(str).str.extract(pola, expand=False)
    d = d[cocok.notna()].copy()
    if d.empty:
        raise SystemExit(f"tidak ada baris {arch}_L{level}_{slot}_s* di {p}")
    d["seed"] = cocok[cocok.notna()].astype(int).to_numpy()
    ganda = d["seed"][d["seed"].duplicated()].tolist()
    if ganda:
        raise SystemExit(
            f"seed ganda {sorted(set(ganda))} untuk {arch}_L{level}_{slot} di {p} — "
            f"rata-ratanya akan membobot seed itu dua kali; bersihkan dulu "
            f"(lihat scripts/cek_csv.py)")
    return d.set_index("seed").sort_index()


def sejajarkan(kondisi: dict) -> dict:
    """Pastikan semua kondisi punya himpunan seed yang sama persis.

    Uji berpasangan atas seed yang tidak sama bukan cuma kurang tepat — ia
    membandingkan split yang berbeda sambil mengaku membandingkan metode.
    """
    himpunan = {n: set(d.index) for n, d in kondisi.items()}
    bersama = set.intersection(*himpunan.values())
    beda = {n: sorted(s - bersama) for n, s in himpunan.items() if s - bersama}
    if beda:
        raise SystemExit(f"himpunan seed berbeda antar kondisi: {beda}")
    urut = sorted(bersama)
    return {n: d.loc[urut] for n, d in kondisi.items()}


def uji_t(a: pd.Series, b: pd.Series) -> dict:
    """Uji-t berpasangan a - b atas seed yang sama.

    `delta_pp` dalam poin persentase, mengikuti cara tabel signifikansi di
    dokumentasi/07 melaporkannya.
    """
    if list(a.index) != list(b.index):
        raise SystemExit(f"seed tidak sejajar: {list(a.index)} vs {list(b.index)}")
    d = a.to_numpy(dtype=float) - b.to_numpy(dtype=float)
    n = len(d)
    t, p = stats.ttest_rel(a.to_numpy(dtype=float), b.to_numpy(dtype=float))
    return {"n": n, "delta_pp": float(d.mean() * 100), "t": float(t), "p": float(p)}


def porsi(ft0: float, ft5: float, ft1: float) -> float:
    """Bagian dari kenaikan FT1 atas FT0 yang sudah dijelaskan 9-crop saja.

    Ini angka yang diminta dosen. NaN kalau FT1 tidak benar-benar di atas FT0:
    tanpa penyebut yang berarti, "berapa persen dari kenaikan" tidak punya arti,
    dan angka raksasa dari pembagian nyaris-nol akan terlihat sah.
    """
    penyebut = ft1 - ft0
    if abs(penyebut) < 1e-9:
        return float("nan")
    return float((ft5 - ft0) / penyebut)
