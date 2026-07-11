"""Ringkas progres grid & estimasi sisa waktu dari results/results.csv.

Jalankan dari root repo:  python scripts/progress.py
"""
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.cvl.config import ALL_ARCHITECTURES, ALL_ABLATION_LEVELS, ALL_SEEDS, ALL_MODES


def run_id(arch, level, mode, seed):
    lvl = "full" if level is None else str(level)
    return f"{arch}_L{lvl}_{mode}_s{seed}"


def main():
    csv = Path("results/results.csv")
    if not csv.exists():
        print("Belum ada results/results.csv — belum ada run yang selesai.")
        return

    grid = []
    for seed in ALL_SEEDS:
        for level in ALL_ABLATION_LEVELS:
            for arch in ALL_ARCHITECTURES:
                for mode in ALL_MODES:
                    grid.append((run_id(arch, level, mode, seed), mode))
    total = len(grid)

    d = pd.read_csv(csv)
    done_ids = set(d["run_id"].astype(str))

    print(f"=== progres: {len(done_ids)}/{total} run selesai "
          f"({100 * len(done_ids) / total:.0f}%) ===")
    print(d.groupby("mode")["train_time_s"].agg(["count", "mean"]).round(0))
    print(f"total waktu terpakai: {d['train_time_s'].sum() / 3600:.1f} jam\n")

    avg = d.groupby("mode")["train_time_s"].mean().to_dict()
    rem = Counter(m for rid, m in grid if rid not in done_ids)

    print("=== sisa ===")
    eta = 0.0
    for mode, n in rem.items():
        per = avg.get(mode)
        if per is None:
            print(f"{mode}: {n} run (belum ada sampel waktu, dilewati)")
            continue
        eta += n * per
        print(f"{mode}: {n} run x {per:.0f}s = {n * per / 3600:.1f} jam")
    print(f"\nEstimasi sisa waktu: {eta / 3600:.1f} jam (~{eta / 86400:.1f} hari)")
    if rem and any(avg.get(m) is None for m in rem):
        print("(catatan: ada mode yang belum punya run selesai -> estimasi masih kasar)")


if __name__ == "__main__":
    main()
