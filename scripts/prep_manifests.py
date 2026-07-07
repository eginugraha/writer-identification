import yaml
from pathlib import Path
from src.cvl.config import lines_root, ABLATION_LEVELS, SEEDS
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest

def main():
    out = Path("results/manifests"); out.mkdir(parents=True, exist_ok=True)
    df = scan_lines(lines_root())
    kept, info = filter_cohort(df)
    print(f"kept writers={info['n_kept_writers']} dropped(<5 pages)={info['dropped_writers']}")
    for seed in SEEDS:
        for level in ABLATION_LEVELS:
            m = build_manifest(kept, n_train_pages=level, seed=seed)
            tag = "full" if level is None else level
            m.to_parquet(out / f"seed{seed}_L{tag}.parquet")
    print("manifests written")

if __name__ == "__main__":
    main()
