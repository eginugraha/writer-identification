import sys
import yaml
from pathlib import Path
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cvl.config import ARCHITECTURES, ABLATION_LEVELS, SEEDS
from src.cvl.run_experiments import run_grid

def main():
    hp = yaml.safe_load(open("configs/default.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    man_dir = Path("results/manifests")
    by_seed_level = {}
    for seed in SEEDS:
        by_seed_level[seed] = {}
        for level in ABLATION_LEVELS:
            tag = "full" if level is None else level
            by_seed_level[seed][level] = pd.read_parquet(man_dir / f"seed{seed}_L{tag}.parquet")
    run_grid(by_seed_level, archs=list(ARCHITECTURES.keys()),
             levels=ABLATION_LEVELS, modes=["pretrained", "scratch"], seeds=SEEDS,
             results_csv="results/results.csv", ckpt_root="results/checkpoints",
             device=device, hp=hp)

if __name__ == "__main__":
    main()
