import argparse
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cvl.report import (
    pivot_markdown, plot_accuracy_vs_n, efficiency_markdown,
    scratch_trainability_markdown,
)

# `full` sudah tidak ada di grid (lihat src/cvl/config.py). Filter ini dipertahankan
# supaya results.csv lama — yang masih memuat baris Lfull — tetap terlaporkan sebagai
# L1–L4 saja.
DROP = ("full",)


def parse_args():
    p = argparse.ArgumentParser(
        description="Bangun laporan hasil eksperimen dari results/results.csv")
    p.add_argument(
        "--date", nargs="?", const="auto", default=None, metavar="TAG",
        help="sisipkan tanggal ke nama laporan DAN figure-nya, supaya run lama "
             "tidak ketimpa. Tanpa nilai = YYYY-MM-DD hari ini; boleh diisi tag "
             "sendiri, mis. --date 2026-08-05-rerun-warmup.")
    p.add_argument(
        "--out", default=None, metavar="PATH",
        help="path laporan eksplisit (menimpa penamaan dari --date).")
    return p.parse_args()


def resolve_suffix(date_arg) -> str:
    """'' bila --date tidak dipakai; '-2026-08-05' atau '-<tag>' bila dipakai."""
    if date_arg is None:
        return ""
    tag = datetime.now().strftime("%Y-%m-%d") if date_arg == "auto" else date_arg
    return f"-{tag}"


def main():
    args = parse_args()
    suffix = resolve_suffix(args.date)
    out_md = Path(args.out) if args.out else Path(f"dokumentasi/08-hasil-eksperimen{suffix}.md")
    acc_png = f"acc_vs_n_pretrained{suffix}.png"

    df = pd.read_csv("results/results.csv")
    fig = Path("results/figures"); fig.mkdir(parents=True, exist_ok=True)
    parts = ["# Hasil Eksperimen — Perbandingan Arsitektur Writer-ID CVL\n",
             f"\n_Dibuat {datetime.now():%Y-%m-%d %H:%M} dari `results/results.csv` "
             f"({len(df)} run)._\n"]

    # === Pretrained: hasil utama, ablasi ukuran data L1–L4 ===
    plot_accuracy_vs_n(df, "pretrained", fig / acc_png, exclude_levels=DROP)
    parts += ["\n## Mode: pretrained (transfer learning)\n",
              "\nHasil utama. Ablasi ukuran data latih L1–L4.\n",
              "\n### Top-1 (halaman)\n", pivot_markdown(df, "top1_page", "pretrained", DROP),
              "\n\n### Top-5 (halaman)\n", pivot_markdown(df, "top5_page", "pretrained", DROP),
              "\n\n### Macro-F1 (halaman)\n", pivot_markdown(df, "macro_f1_page", "pretrained", DROP),
              "\n\n### mAP (retrieval, baris)\n", pivot_markdown(df, "map_line", "pretrained", DROP),
              "\n\n### Efisiensi (rata-rata lintas level/seed)\n", efficiency_markdown(df, "pretrained"),
              f"\n\n![acc](../results/figures/{acc_png})\n"]

    # === Scratch: temuan sekunder — trainability dari nol di data penuh ===
    parts += ["\n## Mode: scratch (dari nol) — trainability\n",
              "\nDilatih dari inisialisasi acak dengan resep sama (LR warmup=3) untuk "
              "SEMUA arsitektur. Dilaporkan hanya pada **L4** (data latih terbanyak, "
              "kondisi terbaik untuk scratch); data lebih sedikit hanya memperparah. "
              "Kolom **kolaps** = jumlah seed dengan top-1 < 0.05 (prediksi ~1 kelas).\n",
              "\n### Top-1 per seed @ L4\n",
              scratch_trainability_markdown(df, level=4, metric="top1_page"),
              "\n\n### Macro-F1 per seed @ L4\n",
              scratch_trainability_markdown(df, level=4, metric="macro_f1_page"),
              "\n\n> Swin-Tiny (kolaps 3/3) dan ConvNeXt-Tiny (kolaps 2/3) gagal konvergen "
              "dari scratch bahkan dengan warmup + data L4, sementara CNN (ResNet, "
              "EfficientNet) dan ViT tidak pernah kolaps (0/3). Arsitektur hierarkis "
              "modern menuntut pretraining pada skala dataset ini.\n"]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(parts))
    print(f"report written to {out_md}")
    print(f"figure written to results/figures/{acc_png}")

if __name__ == "__main__":
    main()
