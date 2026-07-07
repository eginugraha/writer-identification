from pathlib import Path
import pandas as pd
from src.cvl.report import pivot_markdown, plot_accuracy_vs_n

def main():
    df = pd.read_csv("results/results.csv")
    fig = Path("results/figures"); fig.mkdir(parents=True, exist_ok=True)
    parts = ["# Hasil Eksperimen — Perbandingan Arsitektur Writer-ID CVL\n"]
    for mode in ["pretrained", "scratch"]:
        plot_accuracy_vs_n(df, mode, fig / f"acc_vs_n_{mode}.png")
        parts += [f"\n## Mode: {mode}\n",
                  "\n### Top-1 (halaman)\n", pivot_markdown(df, "top1_page", mode),
                  "\n\n### mAP (retrieval, baris)\n", pivot_markdown(df, "map_line", mode),
                  f"\n\n![acc](../results/figures/acc_vs_n_{mode}.png)\n"]
    Path("dokumentasi/08-hasil-eksperimen.md").write_text("\n".join(parts))
    print("report written to dokumentasi/08-hasil-eksperimen.md")

if __name__ == "__main__":
    main()
