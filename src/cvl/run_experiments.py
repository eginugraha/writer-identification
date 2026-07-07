from pathlib import Path
import pandas as pd
from .train import RunConfig, train_one_run
from .evaluate import evaluate_checkpoint

def run_id(arch, level, mode, seed) -> str:
    lvl = "full" if level is None else str(level)
    return f"{arch}_L{lvl}_{mode}_s{seed}"

def already_done(results_csv, rid: str) -> bool:
    p = Path(results_csv)
    if not p.exists():
        return False
    df = pd.read_csv(p)
    return "run_id" in df.columns and rid in set(df["run_id"].astype(str))

def _append_row(results_csv, row: dict):
    p = Path(results_csv); p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(p, mode="a", header=not p.exists(), index=False)

def run_grid(manifest_by_seed_level, archs, levels, modes, seeds,
             results_csv, ckpt_root, device, hp) -> None:
    ckpt_root = Path(ckpt_root)
    for seed in seeds:
        for level in levels:
            manifest = manifest_by_seed_level[seed][level]
            for arch in archs:
                for mode in modes:
                    rid = run_id(arch, level, mode, seed)
                    if already_done(results_csv, rid):
                        print(f"skip {rid}"); continue
                    epochs = hp["pretrained_epochs"] if mode == "pretrained" else hp["scratch_epochs"]
                    rc = RunConfig(arch=arch, level=level, mode=mode, seed=seed,
                                   epochs=epochs, lr=hp["lr"], batch_size=hp["batch_size"],
                                   weight_decay=hp.get("weight_decay", 0.05))
                    out_dir = ckpt_root / rid
                    tr = train_one_run(manifest, rc, out_dir, device, hp)
                    ev = evaluate_checkpoint(out_dir / "best.pt", manifest, arch, device,
                                             batch_size=hp["batch_size"])
                    _append_row(results_csv, {"run_id": rid, "arch": arch,
                        "level": ("full" if level is None else level), "mode": mode,
                        "seed": seed, **tr, **ev})
                    print(f"done {rid}: top1={ev['top1_page']:.3f} map={ev['map_line']:.3f}")
