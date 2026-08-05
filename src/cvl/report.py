import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_METRICS = ["top1_page", "top5_page", "macro_f1_page", "map_line", "top1_retrieval"]

def summarize(df):
    present = [m for m in _METRICS if m in df.columns]
    g = df.groupby(["arch", "level", "mode"])[present].agg(["mean", "std"]).reset_index()
    g.columns = ["arch", "level", "mode"] + [f"{m}_{s}" for m in present for s in ("mean", "std")]
    return g

def _level_order(v):
    return 999 if v == "full" else int(v)

def pivot_markdown(df, metric: str, mode: str, exclude_levels=()) -> str:
    s = summarize(df)
    s = s[s["mode"] == mode]
    exclude = {str(x) for x in exclude_levels}
    levels = [l for l in sorted(s["level"].unique(), key=_level_order) if str(l) not in exclude]
    archs = sorted(s["arch"].unique())
    header = "| arch | " + " | ".join(f"N={l}" for l in levels) + " |"
    sep = "|" + "---|" * (len(levels) + 1)
    lines = [header, sep]
    for a in archs:
        cells = []
        for l in levels:
            row = s[(s.arch == a) & (s.level == l)]
            if len(row):
                cells.append(f"{row[f'{metric}_mean'].iloc[0]:.3f}±{row[f'{metric}_std'].iloc[0]:.3f}")
            else:
                cells.append("-")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

def scratch_trainability_markdown(df, level=4, metric="top1_page", collapse_thresh=0.05) -> str:
    """Tabel per-seed di satu level (default L4 = data latih terbanyak) untuk
    mode scratch.
    Memisahkan stabilitas (jumlah seed yang KOLAPS, metric < collapse_thresh =
    prediksi ~1 kelas) dari akurasi (rerata)."""
    s = df[(df["mode"] == "scratch") & (df["level"].astype(str) == str(level))]
    archs = sorted(s["arch"].unique())
    seeds = sorted(s["seed"].unique())
    header = "| arch | " + " | ".join(f"seed {sd}" for sd in seeds) + " | rerata | kolaps |"
    sep = "|" + "---|" * (len(seeds) + 3)
    lines = [header, sep]
    for a in archs:
        sub = s[s.arch == a]
        vals = []
        for sd in seeds:
            row = sub[sub.seed == sd]
            vals.append(float(row[metric].iloc[0]) if len(row) else float("nan"))
        mean = sum(vals) / len(vals) if vals else float("nan")
        n_coll = sum(1 for v in vals if v < collapse_thresh)
        cells = " | ".join(f"{v:.3f}" for v in vals)
        lines.append(f"| {a} | {cells} | {mean:.3f} | {n_coll}/{len(seeds)} |")
    return "\n".join(lines)


def efficiency_markdown(df, mode: str) -> str:
    s = df[df["mode"] == mode]
    cols = [c for c in ("n_params", "throughput_img_s", "train_time_s") if c in s.columns]
    g = s.groupby("arch")[cols].mean().reset_index()
    header = "| arch | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for _, row in g.sort_values("arch").iterrows():
        cells = [f"{row[c]:.3f}" for c in cols]
        lines.append(f"| {row['arch']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

def plot_accuracy_vs_n(df, mode: str, out_png, exclude_levels=()):
    s = summarize(df); s = s[s["mode"] == mode]
    exclude = {str(x) for x in exclude_levels}
    s = s[~s["level"].astype(str).isin(exclude)]
    plt.figure(figsize=(7, 5))
    for a in sorted(s["arch"].unique()):
        sub = s[s.arch == a].copy()
        sub["ord"] = sub["level"].map(_level_order)
        sub = sub.sort_values("ord")
        plt.errorbar(range(len(sub)), sub["top1_page_mean"], yerr=sub["top1_page_std"],
                     marker="o", label=a, capsize=3)
        plt.xticks(range(len(sub)), [str(x) for x in sub["level"]])
    plt.xlabel("halaman latih / penulis (N)"); plt.ylabel("Top-1 (halaman)")
    plt.title(f"Akurasi vs data latih ({mode})"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
