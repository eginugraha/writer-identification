import argparse
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cvl.config import add_date_args, date_suffix
from src.cvl.report import (
    pivot_markdown, plot_accuracy_vs_n, efficiency_markdown,
    scratch_trainability_markdown, ringkasan_kolaps,
)

# `full` sudah tidak ada di grid (lihat src/cvl/config.py). Filter ini dipertahankan
# supaya results.csv lama — yang masih memuat baris Lfull — tetap terlaporkan sebagai
# L1–L4 saja.
DROP = ("full",)


def parse_args():
    p = argparse.ArgumentParser(
        description="Bangun laporan hasil eksperimen dari results/results*.csv")
    add_date_args(p)
    p.add_argument(
        "--results", default=None, metavar="PATH",
        help="path results.csv sumber (default: mengikuti --date, "
             "jatuh ke results/results.csv bila yang bertanggal belum ada).")
    p.add_argument(
        "--out", default=None, metavar="PATH",
        help="path laporan eksplisit (menimpa penamaan dari --date).")
    return p.parse_args()


def resolve_results_csv(explicit, suffix) -> Path:
    """CSV sumber. Dengan --date, pakai results/results<suffix>.csv bila ada;
    kalau belum ada, mundur ke yang kanonik — tapi bilang, jangan diam-diam."""
    if explicit:
        return Path(explicit)
    dated = Path(f"results/results{suffix}.csv")
    if suffix and not dated.exists():
        print(f"[!] {dated} tidak ada -> pakai results/results.csv")
        return Path("results/results.csv")
    return dated


def bagian_pretrained(df, fig_dir, acc_png):
    plot_accuracy_vs_n(df, "pretrained", fig_dir / acc_png, exclude_levels=DROP)
    return ["\n## Mode: pretrained (transfer learning)\n",
            "\nHasil utama. Ablasi ukuran data latih L1–L4.\n",
            "\n### Top-1 (halaman)\n", pivot_markdown(df, "top1_page", "pretrained", DROP),
            "\n\n### Top-5 (halaman)\n", pivot_markdown(df, "top5_page", "pretrained", DROP),
            "\n\n### Macro-F1 (halaman)\n", pivot_markdown(df, "macro_f1_page", "pretrained", DROP),
            "\n\n### mAP (retrieval, baris)\n", pivot_markdown(df, "map_line", "pretrained", DROP),
            "\n\n### Efisiensi (rata-rata lintas level/seed)\n", efficiency_markdown(df, "pretrained"),
            f"\n\n![acc](../results/figures/{acc_png})\n"]


def bagian_scratch(df, fig_dir, acc_png):
    plot_accuracy_vs_n(df, "scratch", fig_dir / acc_png, exclude_levels=DROP)
    return ["\n## Mode: scratch (dari nol) — trainability\n",
            "\nDilatih dari inisialisasi acak dengan resep sama (LR warmup=3) untuk "
            "SEMUA arsitektur. Dilaporkan hanya pada **L4** (data latih terbanyak, "
            "kondisi terbaik untuk scratch); data lebih sedikit hanya memperparah. "
            "Kolom **kolaps** = jumlah seed dengan top-1 < 0.05 (prediksi ~1 kelas).\n",
            "\n### Top-1 per seed @ L4\n",
            scratch_trainability_markdown(df, level=4, metric="top1_page"),
            "\n\n### Macro-F1 per seed @ L4\n",
            scratch_trainability_markdown(df, level=4, metric="macro_f1_page"),
            "\n\n", ringkasan_kolaps(df, level=4, metric="top1_page"),
            f"\n\n![acc](../results/figures/{acc_png})\n"]


# Bagian yang bisa ditulis, dipilih menurut mode yang benar-benar ada di CSV.
# Pembagian dua server membuat tiap berkas hanya memuat satu mode; meminta
# bagian yang datanya tidak ada menghasilkan tabel kosong dan figure bersumbu
# kosong, bukan error.
BAGIAN = {"pretrained": bagian_pretrained, "scratch": bagian_scratch}


def main():
    args = parse_args()
    suffix = date_suffix(args.date)
    src_csv = resolve_results_csv(args.results, suffix)
    out_md = Path(args.out) if args.out else Path(f"dokumentasi/08-hasil-eksperimen{suffix}.md")

    df = pd.read_csv(src_csv)
    ada = [m for m in BAGIAN if m in set(df["mode"].astype(str))]
    if not ada:
        raise SystemExit(
            f"{src_csv}: tidak ada mode yang dikenali di kolom `mode` "
            f"(ditemukan: {sorted(set(df['mode'].astype(str)))}; "
            f"diharapkan salah satu dari {sorted(BAGIAN)})")

    fig = Path("results/figures"); fig.mkdir(parents=True, exist_ok=True)
    parts = ["# Hasil Eksperimen — Perbandingan Arsitektur Writer-ID CVL\n",
             f"\n_Dibuat {datetime.now():%Y-%m-%d %H:%M} dari `{src_csv}` "
             f"({len(df)} run, mode: {', '.join(ada)})._\n"]

    png = []
    for mode in ada:
        acc_png = f"acc_vs_n_{mode}{suffix}.png"
        parts += BAGIAN[mode](df, fig, acc_png)
        png.append(acc_png)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(parts))
    print(f"report written to {out_md}")
    for p in png:
        print(f"figure written to results/figures/{p}")
    lewat = [m for m in BAGIAN if m not in ada]
    if lewat:
        print(f"(mode {', '.join(lewat)} tidak ada di CSV ini -> bagiannya dilewati)")

if __name__ == "__main__":
    main()
