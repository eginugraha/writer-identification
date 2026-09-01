"""Registry skenario Studi 2 (fine-tuning ConvNeXt-Tiny di L1).

Satu skenario = satu mekanisme yang diubah. `Scenario()` tanpa argumen
adalah pipeline apa adanya, sehingga FT0 tidak perlu dijalankan ulang dan
barisnya bisa disalin dari hasil grid utama.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    geometry: str = "center"            # "center" | "linewindow"
    aug: str = "baseline"               # "baseline" | "strong"
    drop_path: float = 0.0
    label_smoothing: float = 0.0
    freeze_strategy: str | None = None  # kunci di finetune.STRATEGIES
    head: str = "linear"                # "linear" | "arcface"
    eval_crops: int = 1


SCENARIOS: dict[str, Scenario] = {
    # baseline: diambil dari results-pretrained.csv, tidak dijalankan ulang
    "FT0": Scenario(),
    # cakupan baris: jendela acak saat latih, 9 jendela dirata-rata saat uji
    "FT1": Scenario(geometry="linewindow", eval_crops=9),
    # regularisasi bawaan ConvNeXt
    "FT2": Scenario(drop_path=0.2, label_smoothing=0.1),
    # transfer learning klasik (S3 = beku stem+stages.0-1, LLRD 0.7)
    "FT3": Scenario(freeze_strategy="S3"),
    # head margin sudut
    "FT4": Scenario(head="arcface"),
    # protokol uji FT1 di atas pipeline latih FT0: 9 jendela dirata-rata saat
    # uji, training tidak disentuh. Memisahkan efek test-time ensemble dari
    # efek sliding-window training, yang di FT1 tercampur jadi satu angka.
    "FT5": Scenario(eval_crops=9),
    # augmentasi kuat
    "AUG": Scenario(aug="strong"),
}


# Skenario yang tidak melatih apa pun: hanya mengganti protokol uji di atas
# checkpoint yang sudah ada, lewat scripts/eval_only.py. Kalau ikut jalur latih,
# bobotnya jadi model baru — bukan lagi bobot FT0 yang justru jadi intinya.
EVAL_ONLY = frozenset({"FT5"})


def skenario_latih() -> list[str]:
    """Skenario yang perlu dilatih: semua kecuali FT0 dan yang eval-only.

    Satu sumber kebenaran untuk runner dan CLI. Filter `!= "FT0"` yang ditulis
    ulang di scripts/ akan diam-diam ikut melatih setiap skenario eval-only yang
    ditambahkan kemudian.
    """
    return [n for n in SCENARIOS if n != "FT0" and n not in EVAL_ONLY]


def scenario_run_id(name: str, seed: int, arch: str, level) -> str:
    """Identitas satu run Studi 2.

    Mengikuti pola grid utama ({arch}_L{level}_{mode}_s{seed}) dengan nama
    skenario di posisi mode. Arsitektur wajib ikut: tanpa itu dua arsitektur
    yang menulis ke CSV yang sama membuat `already_done` melewati run kedua
    sebagai "sudah selesai", dan folder checkpoint-nya saling menimpa.
    """
    return f"{arch}_L{level}_{name}_s{seed}"
