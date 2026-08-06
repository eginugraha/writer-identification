"""Periksa dan bersihkan CSV hasil yang tertulis oleh dua proses sekaligus.

Menjalankan run_all.py dua kali pada tag --date yang sama membuat dua proses
menulis ke CSV dan folder checkpoint yang sama tanpa penguncian. Akibatnya:
baris header nyasar di tengah berkas (keduanya mengira berkasnya belum ada),
run_id ganda, dan checkpoint yang ditulis berebut sehingga metriknya tidak
bisa dipercaya.

    python scripts/cek_csv.py results/results-pretrained.csv
    python scripts/cek_csv.py results/results-pretrained.csv --bersihkan

Mode --bersihkan menyimpan cadangan, membuang header nyasar, dan membuang
SELURUH baris milik run_id yang ganda — bukan menyisakan salah satunya —
karena keduanya berasal dari checkpoint yang sama-sama diragukan. Run yang
dibuang akan dilatih ulang sendiri saat perintah run_all.py diulang.
"""
import argparse
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd


def periksa(df: pd.DataFrame):
    """(baris_header_nyasar, daftar_run_id_ganda)"""
    nyasar = df[df["run_id"] == "run_id"]
    asli = df[df["run_id"] != "run_id"]
    ganda = asli[asli.duplicated("run_id", keep=False)]
    return nyasar, sorted(ganda["run_id"].unique())


def main():
    p = argparse.ArgumentParser(description="Periksa/bersihkan CSV hasil yang berebut")
    p.add_argument("csv", help="mis. results/results-pretrained.csv")
    p.add_argument("--bersihkan", action="store_true",
                   help="tulis ulang CSV tanpa baris rusak (membuat cadangan)")
    args = p.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"{path} belum ada — tidak ada yang perlu diperiksa.")

    df = pd.read_csv(path)
    nyasar, ids_ganda = periksa(df)
    print(f"{path}: {len(df)} baris")
    print(f"  header nyasar di tengah berkas : {len(nyasar)}")
    print(f"  run_id ganda                   : {len(ids_ganda)}")
    for r in ids_ganda[:10]:
        print(f"      {r}")
    if len(ids_ganda) > 10:
        print(f"      ... dan {len(ids_ganda) - 10} lagi")

    if not nyasar.empty or ids_ganda:
        if not args.bersihkan:
            print("\nJalankan ulang dengan --bersihkan untuk memperbaiki.")
            return
    else:
        print("\nBersih — tidak ada tanda dua proses menulis bersamaan.")
        return

    cadangan = path.with_suffix(".csv.sebelum-bersih")
    shutil.copy(path, cadangan)
    bersih = df[(df["run_id"] != "run_id") & (~df["run_id"].isin(ids_ganda))]
    bersih.to_csv(path, index=False)
    print(f"\ncadangan : {cadangan}")
    print(f"tersisa  : {len(bersih)} baris (dibuang {len(df) - len(bersih)})")
    if ids_ganda:
        ckpt = path.stem.replace("results", "checkpoints")
        print(f"\nHapus juga checkpoint yang ditulis berebut, lalu ulangi run_all.py:")
        for r in ids_ganda:
            print(f"  rm -rf results/{ckpt}/{r}")


if __name__ == "__main__":
    main()
