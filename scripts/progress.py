"""Ringkas progres grid & estimasi sisa waktu.

Menghormati pembagian grid lewat .env / environment variable, jadi di server
yang hanya mengerjakan satu mode estimasinya tidak ikut menghitung mode lain:

    CVL_MODES=scratch    python scripts/progress.py --date scratch
    CVL_MODES=pretrained python scripts/progress.py --date pretrained

Tanpa --date, dibaca dari results/results.csv seperti sebelumnya.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.cvl.config import (
    ARCHITECTURES, ABLATION_LEVELS, SEEDS, MODES, add_date_args, date_suffix,
)


def run_id(arch, level, mode, seed):
    lvl = "full" if level is None else str(level)
    return f"{arch}_L{lvl}_{mode}_s{seed}"


def build_grid(archs, levels, modes, seeds):
    """Semua (run_id, mode) yang seharusnya dikerjakan oleh server ini."""
    return [(run_id(arch, level, mode, seed), mode)
            for seed in seeds for level in levels
            for arch in archs for mode in modes]


def parse_args():
    p = argparse.ArgumentParser(description="Progres grid & estimasi sisa waktu")
    add_date_args(p)
    p.add_argument("--results", default=None, metavar="PATH",
                   help="path results.csv eksplisit (menimpa penamaan dari --date)")
    return p.parse_args()


def main():
    args = parse_args()
    suffix = date_suffix(args.date)
    csv = Path(args.results) if args.results else Path(f"results/results{suffix}.csv")

    archs = list(ARCHITECTURES.keys())
    grid = build_grid(archs, ABLATION_LEVELS, MODES, SEEDS)
    total = len(grid)
    print(f"cakupan: archs={archs} levels={ABLATION_LEVELS} modes={MODES} "
          f"seeds={SEEDS} -> {total} run")
    print(f"sumber : {csv}\n")

    if not csv.exists():
        print(f"Belum ada {csv} — belum ada run yang selesai.")
        return

    d = pd.read_csv(csv)
    # Hitung hanya run yang memang milik cakupan server ini; baris asing
    # (mis. CSV gabungan dua server) tidak boleh menggelembungkan persentase.
    grid_ids = {rid for rid, _ in grid}
    done_ids = set(d["run_id"].astype(str)) & grid_ids

    print(f"=== progres: {len(done_ids)}/{total} run selesai "
          f"({100 * len(done_ids) / total:.0f}%) ===")
    mine = d[d["run_id"].astype(str).isin(grid_ids)]
    print(mine.groupby("mode")["train_time_s"].agg(["count", "mean"]).round(0))
    print(f"total waktu terpakai: {mine['train_time_s'].sum() / 3600:.1f} jam\n")

    avg = mine.groupby("mode")["train_time_s"].mean().to_dict()
    rem = Counter(m for rid, m in grid if rid not in done_ids)

    print("=== sisa ===")
    eta = 0.0
    for mode, n in sorted(rem.items()):
        per = avg.get(mode)
        if per is None:
            print(f"{mode}: {n} run (belum ada sampel waktu, dilewati)")
            continue
        eta += n * per
        print(f"{mode}: {n} run x {per:.0f}s = {n * per / 3600:.1f} jam")
    print(f"\nEstimasi sisa waktu: {eta / 3600:.1f} jam (~{eta / 86400:.1f} hari)")
    if rem and any(avg.get(m) is None for m in rem):
        print("(catatan: ada mode yang belum punya run selesai -> estimasi masih kasar)")
    print("(estimasi memakai rata-rata run yang sudah selesai; level besar lebih "
          "lambat, jadi angka di awal cenderung terlalu optimis)")


if __name__ == "__main__":
    main()
