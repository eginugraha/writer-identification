import argparse
import sys
import yaml
from pathlib import Path
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cvl.config import (
    ARCHITECTURES, ABLATION_LEVELS, SEEDS, MODES,
    PRETRAINED_EPOCHS, SCRATCH_EPOCHS, BATCH_SIZE,
    add_date_args, date_suffix,
)
from src.cvl.run_experiments import kunci_eksklusif, run_grid

def parse_args():
    p = argparse.ArgumentParser(description="Jalankan grid eksperimen writer-ID")
    add_date_args(p)
    p.add_argument("--results", default=None, metavar="PATH",
                   help="path results.csv eksplisit (menimpa penamaan dari --date)")
    p.add_argument("--ckpt-root", default=None, metavar="DIR",
                   help="folder checkpoint eksplisit (menimpa penamaan dari --date)")
    return p.parse_args()

def main():
    args = parse_args()
    suffix = date_suffix(args.date)
    results_csv = Path(args.results) if args.results else Path(f"results/results{suffix}.csv")
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else Path(f"results/checkpoints{suffix}")

    hp = yaml.safe_load(open("configs/default.yaml"))
    # override opsional dari .env
    if PRETRAINED_EPOCHS is not None:
        hp["pretrained_epochs"] = PRETRAINED_EPOCHS
    if SCRATCH_EPOCHS is not None:
        hp["scratch_epochs"] = SCRATCH_EPOCHS
    if BATCH_SIZE is not None:
        hp["batch_size"] = BATCH_SIZE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    man_dir = Path("results/manifests")
    print(f"grid: archs={list(ARCHITECTURES.keys())} levels={ABLATION_LEVELS} "
          f"modes={MODES} seeds={SEEDS} device={device}")
    n_done = len(pd.read_csv(results_csv)) if results_csv.exists() else 0
    print(f"output: {results_csv} ({n_done} run sudah ada -> di-skip) | ckpt: {ckpt_root}")
    by_seed_level = {}
    for seed in SEEDS:
        by_seed_level[seed] = {}
        for level in ABLATION_LEVELS:
            tag = "full" if level is None else level
            by_seed_level[seed][level] = pd.read_parquet(man_dir / f"seed{seed}_L{tag}.parquet")
    with kunci_eksklusif(results_csv):
        run_grid(by_seed_level, archs=list(ARCHITECTURES.keys()),
                 levels=ABLATION_LEVELS, modes=MODES, seeds=SEEDS,
                 results_csv=results_csv, ckpt_root=ckpt_root,
                 device=device, hp=hp)

if __name__ == "__main__":
    main()
