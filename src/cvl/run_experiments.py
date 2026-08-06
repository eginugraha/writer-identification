import os
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
from .train import RunConfig, train_one_run
from .evaluate import evaluate_checkpoint
from .env_info import env_metadata

# Override LR per (arch, mode). ConvNeXt-Tiny divergen saat fine-tune pada LR
# bersama 3e-4 (kolaps ke prediksi 1 kelas, top1 ~0.003); arsitektur lain stabil.
# Ini sensitivitas fine-tuning yang memang dikenal pada ConvNeXt, jadi khusus
# jalur pretrained-nya LR diturunkan ke 1e-4. Dicatat di metodologi skripsi.
LR_OVERRIDES = {
    ("convnext_tiny", "pretrained"): 1e-4,
}

def _proses_hidup(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def kunci_eksklusif(results_csv):
    """Cegah dua proses menulis ke CSV hasil yang sama.

    Menjalankan perintah run dua kali (mudah terjadi: satu baris `nohup ... &`
    yang tidak sengaja diulang) membuat dua proses menambahkan baris ke CSV
    dan menulis ke folder checkpoint yang sama tanpa saling tahu. Gejalanya
    baru terlihat belakangan: header nyasar di tengah berkas, run_id ganda,
    dan checkpoint yang metriknya berasal dari bobot yang ditulis berebut.
    """
    lock = Path(str(results_csv) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        isi = lock.read_text().strip()
        if isi.isdigit() and _proses_hidup(int(isi)):
            raise SystemExit(
                f"Sudah ada proses (PID {isi}) yang menulis ke {results_csv}.\n"
                f"Menjalankan dua proses pada tag --date yang sama merusak CSV "
                f"dan checkpoint.\n"
                f"Pantau yang sedang jalan, atau hentikan dengan: kill {isi}\n"
                f"Kalau yakin proses itu sudah mati, hapus: rm {lock}")
        print(f"lock basi dari PID {isi} diabaikan (prosesnya sudah tidak ada)")
    lock.write_text(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


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
                    lr = LR_OVERRIDES.get((arch, mode), hp["lr"])
                    rc = RunConfig(arch=arch, level=level, mode=mode, seed=seed,
                                   epochs=epochs, lr=lr, batch_size=hp["batch_size"],
                                   weight_decay=hp.get("weight_decay", 0.05))
                    if lr != hp["lr"]:
                        print(f"  [{rid}] LR override -> {lr:g} (default {hp['lr']:g})")
                    out_dir = ckpt_root / rid
                    tr = train_one_run(manifest, rc, out_dir, device, hp)
                    ev = evaluate_checkpoint(out_dir / "best.pt", manifest, arch, device,
                                             batch_size=hp["batch_size"],
                                             num_workers=hp.get("num_workers", 0))
                    _append_row(results_csv, {"run_id": rid, "arch": arch,
                        "level": ("full" if level is None else level), "mode": mode,
                        "seed": seed, "lr": lr, **tr, **ev, **env_metadata(device)})
                    print(f"done {rid}: top1={ev['top1_page']:.3f} map={ev['map_line']:.3f}")
