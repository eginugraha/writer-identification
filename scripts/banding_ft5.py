"""Bandingkan FT0 -> FT5 -> FT1 secara berpasangan per seed.

Menjawab pertanyaan yang diajukan pembimbing: dari kenaikan FT1 atas FT0,
berapa bagian yang sebenarnya sudah dijelaskan oleh 9-crop averaging saat
inferensi saja (FT5), dan berapa yang tersisa untuk sliding-window training.

    python scripts/banding_ft5.py --arch swin_tiny

Tidak menarik torch/timm, jadi bisa dijalankan di laptop atas CSV yang
disalin dari server.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cvl.banding import baca_kondisi, sejajarkan, uji_t, porsi

METRIK = ["top1_page", "top5_page", "macro_f1_page", "map_line", "top1_retrieval"]

# Ambang dua sisi 5% untuk df = n-1, dipakai hanya untuk menandai baris.
AMBANG = {4: 2.776}


def parse_args():
    p = argparse.ArgumentParser(
        description="Perbandingan berpasangan FT0/FT5/FT1 per seed")
    p.add_argument("--ft0", default="results/results-pretrained.csv", metavar="PATH")
    p.add_argument("--ft5", default="results/results-evalonly-swin.csv", metavar="PATH")
    p.add_argument("--ft1", default="results/results-finetune-swin.csv", metavar="PATH")
    p.add_argument("--arch", default="swin_tiny", metavar="NAMA")
    p.add_argument("--level", type=int, default=1, metavar="N")
    return p.parse_args()


def main():
    a = parse_args()
    kondisi = sejajarkan({
        "FT0": baca_kondisi(a.ft0, a.arch, a.level, "pretrained"),
        "FT5": baca_kondisi(a.ft5, a.arch, a.level, "FT5"),
        "FT1": baca_kondisi(a.ft1, a.arch, a.level, "FT1"),
    })
    seeds = list(kondisi["FT0"].index)
    print(f"{a.arch} L{a.level} | seed {seeds} (n={len(seeds)}, berpasangan)\n")

    print("rata-rata +- simpangan baku")
    print(f"{'kondisi':<8}" + "".join(f"{m:>19}" for m in METRIK))
    for nama, d in kondisi.items():
        sel = "".join(f"{d[m].mean():>11.4f} +-{d[m].std():.3f}" for m in METRIK)
        print(f"{nama:<8}{sel}")

    print("\nselisih berpasangan (poin persentase; * = |t| > 2,776, df=4)")
    print(f"{'perbandingan':<14}" + "".join(f"{m:>21}" for m in METRIK))
    pasangan = [("FT5 - FT0", "FT5", "FT0"), ("FT1 - FT5", "FT1", "FT5"),
                ("FT1 - FT0", "FT1", "FT0")]
    for label, x, y in pasangan:
        sel = ""
        for m in METRIK:
            h = uji_t(kondisi[x][m], kondisi[y][m])
            tanda = "*" if abs(h["t"]) > AMBANG.get(h["n"] - 1, 2.776) else " "
            sel += f"{h['delta_pp']:>+10.2f} t={h['t']:>7.2f}{tanda}"
        print(f"{label:<14}{sel}")

    print("\nbagian kenaikan FT1 atas FT0 yang dijelaskan 9-crop saja (FT5)")
    for m in METRIK:
        f = porsi(kondisi["FT0"][m].mean(), kondisi["FT5"][m].mean(),
                  kondisi["FT1"][m].mean())
        print(f"  {m:<16} {f*100:6.1f}%".replace(".", ",", 1)
              if f == f else f"  {m:<16}      -  (FT1 tidak di atas FT0)")

    print("\nCatatan: putuskan dari top1_page dan macro_f1_page. Sebagian kenaikan")
    print("map_line bersifat mekanis — retrieval atas rata-rata sembilan jendela")
    print("memang lebih stabil daripada atas satu potongan, terlepas dari bobotnya.")


if __name__ == "__main__":
    main()
