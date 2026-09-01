"""Bandingkan kondisi Studi 2 secara berpasangan per seed.

Dua pertanyaan yang dijawab:

1. **Membelah FT1** menjadi efek protokol uji dan efek geometri latih, lewat
   tabel 2x2 FT0 / FT5 / FT6 / FT1 beserta suku interaksinya. Dua komponen yang
   masing-masing menutup keterbatasan yang sama tidak akan menjumlah, dan
   hanya sel keempat yang bisa memperlihatkannya.
2. **Membandingkan bobot lain di protokol yang sama.** Peringkat Studi 2 disusun
   di bawah `eval_crops=1`, padahal protokol uji sendirian bernilai belasan
   poin. AUG dan FT4 yang sudah dinilai ulang dengan 9-crop karena itu
   dibandingkan terhadap FT5 — bukan FT0 — karena hanya pasangan itu yang
   protokol ujinya identik.

    python scripts/banding_protokol.py --arch swin_tiny

Tidak menarik torch/timm, jadi bisa dijalankan di laptop atas CSV yang
disalin dari server.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cvl.banding import baca_kondisi, sejajarkan, uji_t, porsi, interaksi

METRIK = ["top1_page", "top5_page", "macro_f1_page", "map_line", "top1_retrieval"]

# Ambang dua sisi 5% untuk df = n-1, dipakai hanya untuk menandai baris.
AMBANG = {4: 2.776}

# label -> (berkas, slot pada run_id, wajib ada?)
KONDISI = [
    ("FT0", "grid", "pretrained", True),
    ("FT1", "finetune", "FT1", True),
    ("FT5", "evalonly", "FT5", True),
    ("FT6", "evalonly", "FT6-from-FT1", False),
    ("AUG@9", "evalonly", "FT5-from-AUG", False),
    ("FT4@9", "evalonly", "FT7-from-FT4", False),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Perbandingan berpasangan antar kondisi Studi 2, per seed")
    p.add_argument("--grid", default="results/results-pretrained.csv", metavar="PATH")
    p.add_argument("--finetune", default="results/results-finetune-swin.csv", metavar="PATH")
    p.add_argument("--evalonly", default="results/results-evalonly-swin.csv", metavar="PATH")
    p.add_argument("--arch", default="swin_tiny", metavar="NAMA")
    p.add_argument("--level", type=int, default=1, metavar="N")
    return p.parse_args()


def _tanda(h):
    return "*" if abs(h["t"]) > AMBANG.get(h["n"] - 1, 2.776) else " "


def _baris_selisih(label, k, x, y):
    sel = ""
    for m in METRIK:
        h = uji_t(k[x][m], k[y][m])
        sel += f"{h['delta_pp']:>+10.2f} t={h['t']:>7.2f}{_tanda(h)}"
    return f"{label:<14}{sel}"


def main():
    a = parse_args()
    berkas = {"grid": a.grid, "finetune": a.finetune, "evalonly": a.evalonly}

    ada, hilang = {}, []
    for label, sumber, slot, wajib in KONDISI:
        try:
            ada[label] = baca_kondisi(berkas[sumber], a.arch, a.level, slot)
        except SystemExit:
            if wajib:
                raise
            hilang.append(label)
    k = sejajarkan(ada)
    seeds = list(next(iter(k.values())).index)
    print(f"{a.arch} L{a.level} | seed {seeds} (n={len(seeds)}, berpasangan)")
    if hilang:
        print(f"belum dijalankan, dilewati: {', '.join(hilang)}")
    print()

    print("rata-rata +- simpangan baku")
    print(f"{'kondisi':<8}" + "".join(f"{m:>19}" for m in METRIK))
    for nama, d in k.items():
        print(f"{nama:<8}" + "".join(f"{d[m].mean():>11.4f} +-{d[m].std():.3f}"
                                    for m in METRIK))

    if "FT6" in k:
        print("\ntabel 2x2 top1_page (latih x uji)")
        print(f"{'':<18}{'uji 1 potongan':>18}{'uji 9 jendela':>18}")
        print(f"{'latih center':<18}{k['FT0']['top1_page'].mean():>18.4f}"
              f"{k['FT5']['top1_page'].mean():>18.4f}")
        print(f"{'latih linewindow':<18}{k['FT6']['top1_page'].mean():>18.4f}"
              f"{k['FT1']['top1_page'].mean():>18.4f}")

    print("\nselisih berpasangan (poin persentase; * = |t| > 2,776, df=4)")
    print(f"{'perbandingan':<14}" + "".join(f"{m:>21}" for m in METRIK))
    pasangan = [("FT5 - FT0", "FT5", "FT0")]
    if "FT6" in k:
        pasangan += [("FT6 - FT0", "FT6", "FT0"), ("FT1 - FT6", "FT1", "FT6")]
    pasangan += [("FT1 - FT5", "FT1", "FT5"), ("FT1 - FT0", "FT1", "FT0")]
    for label, x, y in pasangan:
        print(_baris_selisih(label, k, x, y))

    print("\nbagian kenaikan FT1 atas FT0 yang dijelaskan tiap komponen sendirian")
    print(f"{'komponen':<24}" + "".join(f"{m:>19}" for m in METRIK))
    komponen = [("9-crop saja (FT5)", "FT5")]
    if "FT6" in k:
        komponen.append(("latih linewindow (FT6)", "FT6"))
    for label, x in komponen:
        sel = ""
        for m in METRIK:
            f = porsi(k["FT0"][m].mean(), k[x][m].mean(), k["FT1"][m].mean())
            sel += f"{f*100:>18.1f}%" if f == f else f"{'-':>19}"
        print(f"{label:<24}{sel}")

    if "FT6" in k:
        print("\ninteraksi (FT1-FT6) - (FT5-FT0), poin persentase")
        print(f"{'':<24}" + "".join(
            f"{interaksi(k['FT0'][m].mean(), k['FT5'][m].mean(), k['FT6'][m].mean(), k['FT1'][m].mean()):>+18.2f} "
            for m in METRIK))
        print("Negatif = kedua komponen tumpang tindih: keduanya menutup")
        print("keterbatasan yang sama, jadi yang kedua jauh berkurang gunanya.")

    lain = [n for n in ("AUG@9", "FT4@9") if n in k]
    if lain:
        print("\nbobot lain di bawah protokol 9-crop yang sama, terhadap FT5")
        print(f"{'perbandingan':<14}" + "".join(f"{m:>21}" for m in METRIK))
        for n in lain + ["FT1"]:
            print(_baris_selisih(f"{n} - FT5", k, n, "FT5"))
        print("FT5 memakai bobot FT0. Perbandingan terhadap FT0 langsung tidak sah")
        print("di sini: protokol ujinya berbeda, jadi selisihnya bercampur.")

    print("\nCatatan: putuskan dari top1_page dan macro_f1_page. Sebagian kenaikan")
    print("map_line bersifat mekanis — retrieval atas rata-rata sembilan jendela")
    print("memang lebih stabil daripada atas satu potongan, terlepas dari bobotnya.")


if __name__ == "__main__":
    main()
