"""Runner Studi 2: enam skenario fine-tuning ConvNeXt-Tiny di L1.

Meniru pola `run_grid`: resume-able per berkas CSV, satu baris per
(skenario, seed). FT0 sengaja dilewati — barisnya disalin dari
results-pretrained.csv karena konfigurasinya identik dengan grid utama.
"""
from pathlib import Path
import pandas as pd
from .env_info import env_metadata
from .evaluate import evaluate_checkpoint
from .run_experiments import already_done, _append_row, run_id, LR_OVERRIDES
from .scenarios import SCENARIOS, EVAL_ONLY, scenario_run_id
from .train import RunConfig, train_one_run


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
        if name in EVAL_ONLY:
            print(f"skip {name} (eval-only — jalankan scripts/eval_only.py)")
            continue
        sc = SCENARIOS[name]
        for seed in seeds:
            rid = scenario_run_id(name, seed, arch=arch, level=level)
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


def _cek_kolom(results_csv, row: dict) -> None:
    """Tolak menulis ke CSV yang headernya beda.

    `_append_row` menulis header hanya kalau berkasnya belum ada. Menyalurkan
    baris eval-only (tanpa kolom latih) ke results-finetune-*.csv karena itu
    tidak akan error — nilainya masuk ke bawah header lain, dan hasilnya CSV
    yang kolomnya bergeser tanpa satu pun peringatan.
    """
    p = Path(results_csv)
    if not p.exists():
        return
    ada = set(pd.read_csv(p, nrows=0).columns)
    if ada != set(row):
        raise SystemExit(
            f"kolom {p} tidak cocok dengan baris eval-only "
            f"(hanya di CSV: {sorted(ada - set(row))}; hanya di baris: "
            f"{sorted(set(row) - ada)}) — pakai --results terpisah")


def run_eval_only(manifest, name, seeds, results_csv, src_ckpt_root, device, hp,
                  arch: str, level=1, source: str = "pretrained") -> None:
    """Evaluasi ulang checkpoint yang sudah ada dengan protokol uji lain.

    Dipakai FT5: protokol uji FT1 (9 jendela dirata-rata) di atas bobot FT0.
    Karena `LineDataset` mengabaikan `geometry` begitu `eval_crops > 1`, cara
    ujinya identik dengan FT1 — yang berbeda hanya bobotnya. Selisih FT5-FT0
    karena itu adalah efek murni test-time ensemble, dan FT1-FT5 sisanya milik
    sliding-window training. Di FT1 keduanya tercampur jadi satu angka.

    Tidak ada pelatihan sama sekali: barisnya tidak punya kolom latih
    (`train_time_s`, `epochs_ran`, `best_val_acc`, `lr`) karena angka-angka itu
    milik run sumber, bukan run ini — `source_run_id` yang menunjuk ke sana.

    `source` mengisi posisi mode pada nama run sumber, jadi "pretrained"
    menunjuk checkpoint grid utama ({arch}_L{level}_pretrained_s{seed}) dan
    nama skenario menunjuk checkpoint Studi 2 (mis. "FT1", untuk mengukur
    kebalikannya: latih linewindow tapi uji satu potongan).
    """
    sc = SCENARIOS[name]
    src_ckpt_root = Path(src_ckpt_root)
    for seed in seeds:
        rid = scenario_run_id(name, seed, arch=arch, level=level)
        if already_done(results_csv, rid):
            print(f"skip {rid}"); continue
        src_rid = run_id(arch, level, source, seed)
        ckpt = src_ckpt_root / src_rid / "best.pt"
        if not ckpt.exists():
            raise SystemExit(
                f"checkpoint {src_rid} tidak ada di {ckpt} — eval-only butuh "
                f"bobot run sumber, dan results/checkpoints*/ tidak ikut git")
        ev = evaluate_checkpoint(ckpt, manifest, arch, device,
                                 batch_size=hp["batch_size"], scenario=sc,
                                 num_workers=hp.get("num_workers", 0))
        row = {"run_id": rid, "scenario": name, "source_run_id": src_rid,
               "arch": arch, "level": level, "mode": "pretrained", "seed": seed,
               "eval_crops": sc.eval_crops, **ev, **env_metadata(device)}
        _cek_kolom(results_csv, row)
        _append_row(results_csv, row)
        print(f"done {rid} (dari {src_rid}): top1={ev['top1_page']:.3f} "
              f"map={ev['map_line']:.3f}")
