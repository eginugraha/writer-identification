"""Runner Studi 2: enam skenario fine-tuning ConvNeXt-Tiny di L1.

Meniru pola `run_grid`: resume-able per berkas CSV, satu baris per
(skenario, seed). FT0 sengaja dilewati — barisnya disalin dari
results-pretrained.csv karena konfigurasinya identik dengan grid utama.
"""
from pathlib import Path
from .env_info import env_metadata
from .evaluate import evaluate_checkpoint
from .run_experiments import already_done, _append_row, LR_OVERRIDES
from .scenarios import SCENARIOS
from .train import RunConfig, train_one_run


def scenario_run_id(name: str, seed: int) -> str:
    return f"{name}_s{seed}"


def hp_skenario(hp: dict, arch: str = "convnext_tiny") -> dict:
    """Terapkan override LR yang sama dengan grid utama.

    Grid utama melatih ConvNeXt pretrained pada 1e-4, bukan nilai default
    configs/default.yaml, karena divergen di LR yang lebih tinggi. Baseline
    FT0 disalin dari grid itu, jadi skenario wajib memakai LR yang sama —
    kalau tidak, seluruh perbandingan Studi 2 batal.
    """
    hp = dict(hp)
    hp["lr"] = LR_OVERRIDES.get((arch, "pretrained"), hp["lr"])
    return hp


def cek_manifest(man_dir, seeds, level: int = 1) -> None:
    """Gagal cepat kalau ada manifest seed yang belum dibangun, sebelum
    pekerjaan latih yang lama dimulai."""
    man_dir = Path(man_dir)
    hilang = [s for s in seeds if not (man_dir / f"seed{s}_L{level}.parquet").exists()]
    if hilang:
        raise SystemExit(f"manifest belum ada untuk seed {hilang} — jalankan "
                         f"scripts/prep_manifests.py dulu")


def run_scenario_grid(manifest, names, seeds, results_csv, ckpt_root, device, hp,
                      arch: str = "convnext_tiny", level=1) -> None:
    ckpt_root = Path(ckpt_root)
    for name in names:
        if name == "FT0":
            print("skip FT0 (baseline disalin dari results-pretrained.csv)")
            continue
        sc = SCENARIOS[name]
        for seed in seeds:
            rid = scenario_run_id(name, seed)
            if already_done(results_csv, rid):
                print(f"skip {rid}"); continue
            rc = RunConfig(arch=arch, level=level, mode="pretrained", seed=seed,
                           epochs=hp["pretrained_epochs"], lr=hp["lr"],
                           batch_size=hp["batch_size"],
                           weight_decay=hp.get("weight_decay", 0.05))
            out_dir = ckpt_root / rid
            tr = train_one_run(manifest, rc, out_dir, device, hp, scenario=sc)
            ev = evaluate_checkpoint(out_dir / "best.pt", manifest, arch, device,
                                     batch_size=hp["batch_size"], scenario=sc,
                                     num_workers=hp.get("num_workers", 0))
            _append_row(results_csv, {"run_id": rid, "scenario": name, "arch": arch,
                "level": level, "mode": "pretrained", "seed": seed, "lr": rc.lr,
                **tr, **ev, **env_metadata(device)})
            print(f"done {rid}: top1={ev['top1_page']:.3f} map={ev['map_line']:.3f}")
