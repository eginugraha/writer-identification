import sys
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

def main():
    df = pd.read_csv("results/results.csv")
    fig = Path("results/figures"); fig.mkdir(parents=True, exist_ok=True)
    parts = ["# Hasil Eksperimen — Perbandingan Arsitektur Writer-ID CVL\n"]

    # === Pretrained: hasil utama, ablasi ukuran data L1–L4 ===
    plot_accuracy_vs_n(df, "pretrained", fig / "acc_vs_n_pretrained.png", exclude_levels=DROP)
    parts += ["\n## Mode: pretrained (transfer learning)\n",
              "\nHasil utama. Ablasi ukuran data latih L1–L4.\n",
              "\n### Top-1 (halaman)\n", pivot_markdown(df, "top1_page", "pretrained", DROP),
              "\n\n### Top-5 (halaman)\n", pivot_markdown(df, "top5_page", "pretrained", DROP),
              "\n\n### Macro-F1 (halaman)\n", pivot_markdown(df, "macro_f1_page", "pretrained", DROP),
              "\n\n### mAP (retrieval, baris)\n", pivot_markdown(df, "map_line", "pretrained", DROP),
              "\n\n### Efisiensi (rata-rata lintas level/seed)\n", efficiency_markdown(df, "pretrained"),
              "\n\n![acc](../results/figures/acc_vs_n_pretrained.png)\n"]

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

    Path("dokumentasi/08-hasil-eksperimen.md").write_text("\n".join(parts))
    print("report written to dokumentasi/08-hasil-eksperimen.md")

if __name__ == "__main__":
    main()
