"""Evaluasi ulang checkpoint yang sudah ada dengan protokol uji lain.

Dipakai untuk FT5: 9-crop averaging saat inferensi di atas bobot FT0, tanpa
melatih apa pun. Contoh:

    python scripts/eval_only.py --arch swin_tiny \
        --src-ckpt-root results/checkpoints-pretrained --date evalonly-swin

Checkpoint sumbernya berasal dari grid utama, yang punya --date sendiri, jadi
--src-ckpt-root wajib disebut: menebaknya dari --date akan diam-diam menunjuk
folder yang salah (atau kosong).
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cvl.config import ALL_ARCHITECTURES, SEEDS, BATCH_SIZE, add_date_args, date_suffix
from src.cvl.run_experiments import kunci_eksklusif
from src.cvl.run_scenarios import run_eval_only, cek_manifest
from src.cvl.scenarios import SCENARIOS
import yaml


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluasi ulang checkpoint dengan protokol uji lain (tanpa latih)")
    add_date_args(p)
    # Katalog penuh, bukan ARCHITECTURES: yang disebut di sini adalah
    # checkpoint yang sudah ada di disk, bukan grid yang akan dijalankan, jadi
    # filter CVL_ARCHS dari .env tidak relevan dan hanya akan menolaknya.
    p.add_argument("--arch", required=True, choices=sorted(ALL_ARCHITECTURES),
                   help="arsitektur checkpoint sumber (wajib)")
    p.add_argument("--level", type=int, default=1, metavar="N",
                   help="level ablasi (default: 1)")
    p.add_argument("--scenario", default="FT5", metavar="NAMA",
                   help="skenario yang menentukan protokol uji (default: FT5)")
    p.add_argument("--source", default="pretrained", metavar="NAMA",
                   help="posisi mode pada run_id sumber: 'pretrained' untuk "
                        "checkpoint grid utama, atau nama skenario Studi 2 "
                        "(default: pretrained)")
    p.add_argument("--src-ckpt-root", required=True, metavar="DIR",
                   help="folder checkpoint run sumber (wajib)")
    p.add_argument("--results", default=None, metavar="PATH",
                   help="path results.csv eksplisit (menimpa penamaan dari --date)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.scenario not in SCENARIOS:
        raise SystemExit(f"skenario tidak dikenal: {args.scenario}")
    sc = SCENARIOS[args.scenario]
    if sc.eval_crops <= 1:
        print(f"catatan: {args.scenario} memakai eval_crops={sc.eval_crops} — "
              f"protokol ujinya sama dengan baseline")

    suffix = date_suffix(args.date)
    results_csv = Path(args.results) if args.results else Path(f"results/results{suffix}.csv")
    src_ckpt_root = Path(args.src_ckpt_root)

    hp = yaml.safe_load(open("configs/default.yaml"))
    if BATCH_SIZE is not None:
        hp["batch_size"] = BATCH_SIZE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    man_dir = Path("results/manifests")
    cek_manifest(man_dir, SEEDS, args.level)
    print(f"eval-only: {args.scenario} (eval_crops={sc.eval_crops}) atas "
          f"checkpoint {args.arch}_L{args.level}_{args.source}_s* "
          f"| seeds={SEEDS} | device={device}")
    print(f"sumber: {src_ckpt_root} | output: {results_csv}")

    with kunci_eksklusif(results_csv):
        for seed in SEEDS:
            m = pd.read_parquet(man_dir / f"seed{seed}_L{args.level}.parquet")
            run_eval_only(m, name=args.scenario, seeds=[seed],
                          results_csv=results_csv, src_ckpt_root=src_ckpt_root,
                          device=device, hp=hp, arch=args.arch, level=args.level,
                          source=args.source)


if __name__ == "__main__":
    main()
