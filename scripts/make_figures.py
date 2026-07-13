"""Grafik tambahan untuk skripsi: bar chart leaderboard (pretrained) dan
bar chart trainability scratch @ data penuh. Output ke results/figures/."""
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


def leaderboard_pretrained(df, out_png):
    s = df[(df["mode"] == "pretrained") & (df["level"].astype(str).isin(LV))]
    g = s.groupby("arch")["top1_page"].mean().reset_index(name="mean")
    g = g.sort_values("mean", ascending=True)  # terkecil di bawah utk barh
    labels = [DISP[a] for a in g["arch"]]
    plt.figure(figsize=(7, 4))
    bars = plt.barh(labels, g["mean"], color="#4C72B0", edgecolor="black", linewidth=0.5)
    for b, m in zip(bars, g["mean"]):
        plt.text(m + 0.002, b.get_y() + b.get_height() / 2, f"{m:.3f}",
                 va="center", fontsize=9)
    plt.xlim(0.80, 0.93)
    plt.xlabel("Rerata Top-1 (halaman), lintas L1–L4 & 3 seed")
    plt.title("Leaderboard arsitektur — mode pretrained")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()


def scratch_trainability_bar(df, out_png, collapse_thresh=0.05):
    s = df[(df["mode"] == "scratch") & (df["level"].astype(str) == "full")]
    order = ["efficientnetv2_s", "vit_small", "resnet50", "convnext_tiny", "swin_tiny"]
    means, labels, colors = [], [], []
    for a in order:
        vals = s[s.arch == a]["top1_page"].tolist()
        m = sum(vals) / len(vals)
        n_coll = sum(1 for v in vals if v < collapse_thresh)
        means.append(m)
        labels.append(f"{DISP[a]}\n(kolaps {n_coll}/3)")
        colors.append("#C44E52" if n_coll > 0 else "#55A868")
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, means, color=colors, edgecolor="black", linewidth=0.5)
    for b, m in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2, m + 0.01, f"{m:.3f}",
                 ha="center", fontsize=9)
    plt.ylim(0, 1.0)
    plt.ylabel("Rerata Top-1 (halaman) @ data penuh")
    plt.title("Trainability dari scratch — hijau: tak kolaps, merah: ada kolaps")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()


def main():
    df = pd.read_csv("results/results.csv")
    fig = Path("results/figures"); fig.mkdir(parents=True, exist_ok=True)
    leaderboard_pretrained(df, fig / "leaderboard_pretrained.png")
    scratch_trainability_bar(df, fig / "scratch_trainability.png")
    print("figures written to results/figures/{leaderboard_pretrained,scratch_trainability}.png")


if __name__ == "__main__":
    main()
