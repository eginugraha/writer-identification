import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cvl.config import SEEDS, PRETRAINED_EPOCHS, BATCH_SIZE, add_date_args, date_suffix
from src.cvl.finetune import LAYER_MAP
from src.cvl.run_experiments import kunci_eksklusif
from src.cvl.run_scenarios import run_scenario_grid, hp_skenario, cek_manifest
from src.cvl.scenarios import SCENARIOS


def parse_args():
    p = argparse.ArgumentParser(description="Studi 2: skenario fine-tuning di satu arsitektur")
    add_date_args(p)
    # Pilihan dibatasi ke LAYER_MAP, bukan ke seluruh katalog arsitektur:
    # FT3 memakai freeze + LLRD, yang butuh peta nama modul. Arsitektur tanpa
    # peta akan menjalankan FT3 tanpa membekukan apa pun — gagal diam-diam.
    # Wajib, tanpa default: Studi 2 dijalankan di swin_tiny, dan satu flag
    # yang terlupa berarti ~5 jam GPU di arsitektur yang salah.
    p.add_argument("--arch", required=True, choices=sorted(LAYER_MAP),
                   help="arsitektur Studi 2 (wajib)")
    p.add_argument("--level", type=int, default=1, metavar="N",
                   help="level ablasi, halaman latih per penulis (default: 1)")
    p.add_argument("--scenarios", default=None, metavar="NAMA",
                   help="daftar skenario dipisah koma (default: semua kecuali FT0)")
    p.add_argument("--results", default=None, metavar="PATH")
    p.add_argument("--ckpt-root", default=None, metavar="DIR")
    return p.parse_args()


def main():
    args = parse_args()
    suffix = date_suffix(args.date)
    results_csv = Path(args.results) if args.results else Path(f"results/results{suffix}.csv")
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else Path(f"results/checkpoints{suffix}")

    names = ([s.strip() for s in args.scenarios.split(",")] if args.scenarios
             else [n for n in SCENARIOS if n != "FT0"])
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"skenario tidak dikenal: {unknown}")

    hp = yaml.safe_load(open("configs/default.yaml"))
    if PRETRAINED_EPOCHS is not None:
        hp["pretrained_epochs"] = PRETRAINED_EPOCHS
    if BATCH_SIZE is not None:
        hp["batch_size"] = BATCH_SIZE
    # samakan dengan grid utama: LR per (arch, pretrained) diambil dari
    # LR_OVERRIDES, jadi baseline FT0 dan skenario memakai LR yang sama
    hp = hp_skenario(hp, args.arch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    man_dir = Path("results/manifests")
    cek_manifest(man_dir, SEEDS, args.level)
    print(f"skenario: {names} | seeds={SEEDS} | arch={args.arch} L{args.level} "
          f"| lr={hp['lr']:g} | device={device}")
    print(f"output: {results_csv} | ckpt: {ckpt_root}")
    print(f"baseline FT0: ambil dari results-pretrained.csv, pola "
          f"{args.arch}_L{args.level}_pretrained_s*")

    with kunci_eksklusif(results_csv):
        for seed in SEEDS:
            m = pd.read_parquet(man_dir / f"seed{seed}_L{args.level}.parquet")
            run_scenario_grid(m, names=names, seeds=[seed], results_csv=results_csv,
                              ckpt_root=ckpt_root, device=device, hp=hp,
                              arch=args.arch, level=args.level)


if __name__ == "__main__":
    main()
