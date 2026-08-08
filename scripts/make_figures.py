"""Grafik tambahan untuk skripsi: bar chart leaderboard (pretrained) dan
bar chart trainability scratch @ L4. Output ke results/figures/.

Sumbernya satu atau beberapa CSV — pembagian dua server memecah hasil jadi
results-pretrained.csv dan results-scratch.csv, jadi keduanya digabung dulu
lalu tiap grafik mengambil mode yang dibutuhkannya. Mode yang tidak ada di
CSV manapun dilewati, bukan digambar kosong.

Jumlah seed dan batas sumbu dihitung dari data. Versi sebelumnya memaku
"kolaps n/3" dan xlim(0.80, 0.93); yang pertama membuat grafik 5-seed
berbohong, yang kedua diam-diam memotong arsitektur di luar rentang itu.
"""
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DISP = {"resnet50": "ResNet-50", "efficientnetv2_s": "EfficientNetV2-S",
        "vit_small": "ViT-S/16", "swin_tiny": "Swin-T", "convnext_tiny": "ConvNeXt-T"}
LV = ["1", "2", "3", "4"]


def nama_tampil(arch: str) -> str:
    """Nama enak dibaca; arsitektur baru tampil apa adanya, bukan KeyError."""
    return DISP.get(arch, arch)


def batas_sumbu(nilai, pad=0.03):
    """Rentang yang memuat semua batang plus sedikit ruang untuk label angka.

    Dipaksa memuat data, bukan sebaliknya: batas tetap membuat arsitektur di
    luar rentang hilang dari grafik tanpa peringatan apa pun.
    """
    if not len(nilai):
        return (0.0, 1.0)
    lo, hi = min(nilai), max(nilai)
    span = hi - lo
    return (max(0.0, lo - (span * 0.15 + pad)), min(1.0, hi + (span * 0.15 + pad)))


def leaderboard_pretrained(df, out_png):
    s = df[(df["mode"] == "pretrained") & (df["level"].astype(str).isin(LV))]
    n_seed = s["seed"].nunique()
    g = s.groupby("arch")["top1_page"].mean().reset_index(name="mean")
    g = g.sort_values("mean", ascending=True)  # terkecil di bawah utk barh
    labels = [nama_tampil(a) for a in g["arch"]]
    means = list(g["mean"])
    lim = batas_sumbu(means)
    xlabel = f"Rerata Top-1 (halaman), lintas L1–L4 & {n_seed} seed"

    plt.figure(figsize=(7, 4))
    bars = plt.barh(labels, means, color="#4C72B0", edgecolor="black", linewidth=0.5)
    for b, m in zip(bars, means):
        plt.text(m + (lim[1] - lim[0]) * 0.01, b.get_y() + b.get_height() / 2,
                 f"{m:.3f}", va="center", fontsize=9)
    plt.xlim(*lim)
    plt.xlabel(xlabel)
    plt.title("Leaderboard arsitektur — mode pretrained")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    return {"labels": labels, "means": means, "lim": lim, "xlabel": xlabel}


def scratch_trainability_bar(df, out_png, collapse_thresh=0.05):
    s = df[(df["mode"] == "scratch") & (df["level"].astype(str) == "4")]
    n_seed = s["seed"].nunique()
    # urutan dari data (terbaik ke terburuk), bukan daftar tetap: arsitektur
    # yang tidak ada di CSV dulu membuat skrip ini KeyError.
    g = s.groupby("arch")["top1_page"].mean().sort_values(ascending=False)
    labels, means, colors = [], [], []
    for arch, m in g.items():
        vals = s[s.arch == arch]["top1_page"]
        n_coll = int((vals < collapse_thresh).sum())
        labels.append(f"{nama_tampil(arch)}\n(kolaps {n_coll}/{n_seed})")
        means.append(float(m))
        colors.append("#C44E52" if n_coll > 0 else "#55A868")
    lim = batas_sumbu(means + [0.0])

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, means, color=colors, edgecolor="black", linewidth=0.5)
    for b, m in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2, m + (lim[1] - lim[0]) * 0.02,
                 f"{m:.3f}", ha="center", fontsize=9)
    plt.ylim(*lim)
    plt.ylabel(f"Rerata Top-1 (halaman) @ L4, {n_seed} seed")
    plt.title("Trainability dari scratch — hijau: tak kolaps, merah: ada kolaps")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    return {"labels": labels, "means": means, "lim": lim}


GRAFIK = {
    "pretrained": ("leaderboard_pretrained.png", leaderboard_pretrained),
    "scratch": ("scratch_trainability.png", scratch_trainability_bar),
}
BAWAAN = ["results/results-pretrained.csv", "results/results-scratch.csv"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", action="append", default=None, metavar="PATH",
                   help="CSV sumber; boleh diulang (default: hasil kedua server)")
    return p.parse_args()


def baca(paths):
    ada = [Path(p) for p in paths]
    hilang = [p for p in ada if not p.exists()]
    if hilang:
        raise SystemExit(f"CSV tidak ditemukan: {', '.join(str(p) for p in hilang)}")
    return pd.concat([pd.read_csv(p) for p in ada], ignore_index=True)


def main():
    args = parse_args()
    paths = args.results or [p for p in BAWAAN if Path(p).exists()]
    if not paths:
        raise SystemExit(f"tidak ada CSV sumber; dicari: {', '.join(BAWAAN)}")

    df = baca(paths)
    modes = set(df["mode"].astype(str))
    fig = Path("results/figures"); fig.mkdir(parents=True, exist_ok=True)

    ditulis = []
    for mode, (nama, fn) in GRAFIK.items():
        if mode not in modes:
            print(f"(mode {mode} tidak ada di CSV sumber -> {nama} dilewati)")
            continue
        fn(df, fig / nama)
        ditulis.append(nama)
    if ditulis:
        print(f"figures written to results/figures/: {', '.join(ditulis)}")


if __name__ == "__main__":
    main()
