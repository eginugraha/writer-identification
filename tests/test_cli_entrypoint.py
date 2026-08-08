"""Kedua CLI runner harus benar-benar memanggil main() saat dijalankan.

Commit 54c61bd tidak sengaja meng-indent guard `if __name__ == "__main__"`
ke dalam `main()`. Akibatnya `python scripts/run_all.py` keluar dengan status
0 tanpa melatih apa pun: log kosong, CSV tidak pernah dibuat, dan tidak ada
pesan error yang menunjukkan ada yang salah. Run 20 jam yang "selesai" dalam
sedetik hanya ketahuan kalau ada yang memeriksa isi log.

Tes ini memakai AST, bukan menjalankan skripnya, supaya tetap jalan di mesin
tanpa torch/timm terpasang.
"""
import ast
import builtins
from pathlib import Path
import pytest

CLIS = ["scripts/run_all.py", "scripts/run_scenarios.py"]
ROOT = Path(__file__).resolve().parents[1]


def _pohon(nama):
    return ast.parse((ROOT / nama).read_text(), filename=nama)


def _guard_main(node):
    """True kalau node adalah `if __name__ == "__main__":`."""
    return (isinstance(node, ast.If)
            and any(isinstance(n, ast.Name) and n.id == "__name__"
                    for n in ast.walk(node.test)))


@pytest.mark.parametrize("nama", CLIS)
def test_guard_main_ada_di_level_modul(nama):
    pohon = _pohon(nama)
    assert any(_guard_main(n) for n in pohon.body), (
        f"{nama}: guard __main__ tidak ada di level modul -> main() tidak pernah "
        "dipanggil dan skrip keluar diam-diam tanpa menjalankan apa pun"
    )


@pytest.mark.parametrize("nama", CLIS)
def test_guard_main_tidak_bersarang_di_dalam_fungsi(nama):
    for fn in ast.walk(_pohon(nama)):
        if not isinstance(fn, ast.FunctionDef):
            continue
        bersarang = [n for n in ast.walk(fn) if _guard_main(n)]
        assert not bersarang, (
            f"{nama}: guard __main__ ada di dalam def {fn.name}() baris "
            f"{bersarang[0].lineno} -- kemungkinan salah indentasi"
        )


@pytest.mark.parametrize("nama", CLIS)
def test_semua_nama_yang_dipakai_sudah_diimpor(nama):
    """run_scenarios.py sempat memakai kunci_eksklusif tanpa mengimpornya."""
    pohon = _pohon(nama)
    terdefinisi = set(dir(builtins)) | {"__name__", "__file__", "__doc__"}
    for n in ast.walk(pohon):
        if isinstance(n, ast.alias):
            terdefinisi.add((n.asname or n.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            terdefinisi.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            terdefinisi.add(n.id)
        elif isinstance(n, ast.arg):
            terdefinisi.add(n.arg)

    dipakai = {n.id for n in ast.walk(pohon)
               if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert not (dipakai - terdefinisi), (
        f"{nama}: nama dipakai tapi tidak pernah diimpor/didefinisikan: "
        f"{sorted(dipakai - terdefinisi)}"
    )
