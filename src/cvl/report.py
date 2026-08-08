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


def hitung_kolaps(df, level=4, metric="top1_page", collapse_thresh=0.05):
    """(daftar (arch, n_kolaps, n_seed), n_seed) untuk mode scratch di satu level."""
    s = df[(df["mode"] == "scratch") & (df["level"].astype(str) == str(level))]
    seeds = sorted(s["seed"].unique())
    hasil = []
    for a in sorted(s["arch"].unique()):
        vals = s[s.arch == a][metric].astype(float)
        hasil.append((a, int((vals < collapse_thresh).sum()), len(seeds)))
    return hasil, len(seeds)


def ringkasan_kolaps(df, level=4, metric="top1_page", collapse_thresh=0.05) -> str:
    """Kalimat penutup bagian scratch, dibangkitkan dari data.

    Versi sebelumnya memaku teks hasil grid 3-seed ("kolaps 3/3"), sehingga
    laporan 5-seed menutup dengan angka yang bertentangan dengan tabel tepat
    di atasnya. Kalimat yang tampak rapi tapi salah lebih berbahaya daripada
    tabel yang jelas kosong, karena tidak ada yang memeriksanya ulang.
    """
    hasil, n_seed = hitung_kolaps(df, level, metric, collapse_thresh)
    if not hasil:
        return (f"> Tidak ada run scratch di L{level} pada CSV ini, jadi "
                "stabilitas tidak bisa dinilai.")
    kolaps = [(a, n, t) for a, n, t in hasil if n > 0]
    aman = [a for a, n, _ in hasil if n == 0]
    if not kolaps:
        return (f"> Tidak ada arsitektur yang kolaps di L{level} "
                f"({n_seed} seed, semua 0/{n_seed}): {', '.join(aman)}. "
                f"Ambang kolaps: {metric} < {collapse_thresh}.")
    frasa = ", ".join(f"{a} {n}/{t}" for a, n, t in kolaps)
    teks = (f"> Kolaps dari scratch di L{level} ({n_seed} seed, ambang "
            f"{metric} < {collapse_thresh}): {frasa}.")
    if aman:
        teks += f" Tidak pernah kolaps (0/{n_seed}): {', '.join(aman)}."
    teks += (" Rata-rata pada baris yang kolaps tidak bermakna — laporkan "
             "jumlah kolaps dan rata-rata run sehat secara terpisah.")
    return teks


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
