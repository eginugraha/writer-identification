"""Uji unit untuk src/cvl/finetune.py.

Menguji mekanisme freeze/LLRD langsung pada model, bukan lewat training,
supaya cepat dan tidak bergantung pada dataset. Tanpa berkas ini, skenario
FT3 bisa diam-diam mendegradasi jadi FT0 (tidak ada yang dibekukan, satu
grup LR seragam) dan test training tetap lolos karena hanya memeriksa
`epochs_ran >= 1`.
"""
import timm
from src.cvl.finetune import freeze_layers, build_param_groups, STRATEGIES


def _model():
    # pretrained=False supaya tidak ada unduhan; num_classes sesuai studi.
    return timm.create_model("convnext_tiny", pretrained=False, num_classes=308)


def test_freeze_layers_s3_membekukan_prefix_yang_benar():
    model = _model()
    freeze_layers(model, "S3")
    frozen_prefixes = STRATEGIES["S3"]["freeze"]  # ("stem", "stages.0", "stages.1")

    beku = [name for name, p in model.named_parameters() if not p.requires_grad]
    hidup = [name for name, p in model.named_parameters() if p.requires_grad]

    # Arah 1: semua parameter berprefix stem/stages.0/stages.1 harus beku.
    for name, p in model.named_parameters():
        if name.startswith(frozen_prefixes):
            assert not p.requires_grad, f"{name} seharusnya beku di S3"

    # Arah 2: minimal satu parameter DI LUAR prefix tersebut harus tetap
    # hidup — kalau tidak, test "arah 1" saja lolos meski semuanya dibekukan.
    assert any(not name.startswith(frozen_prefixes) for name in hidup)
    # dan kebalikannya: tidak semua parameter ikut beku.
    assert len(hidup) > 0
    assert len(beku) > 0


def test_build_param_groups_s3_llrd_menurun_sesuai_urutan():
    model = _model()
    freeze_layers(model, "S3")
    groups = build_param_groups(model, "S3", base_lr=1e-4)

    # lebih dari satu grup: baseline (S0) hanya satu, S3 harus terdiferensiasi.
    assert len(groups) > 1

    lrs = [g["lr"] for g in groups]
    # menurun tegas mengikuti _LLRD_ORDER (head -> stages.3 -> stages.2 -> ...)
    assert all(a > b for a, b in zip(lrs, lrs[1:]))

    # Nilai yang didokumentasikan di rencana: head 1e-4, stages.3 7e-5, stages.2 4.9e-5.
    assert abs(lrs[0] - 1.0e-4) < 1e-9
    assert abs(lrs[1] - 7.0e-5) < 1e-9
    assert abs(lrs[2] - 4.9e-5) < 1e-9

    # Tidak ada grup yang menyelundupkan parameter beku (requires_grad=False).
    for g in groups:
        for p in g["params"]:
            assert p.requires_grad


def test_build_param_groups_s0_tetap_satu_grup_seragam():
    model = _model()
    freeze_layers(model, "S0")  # S0: freeze=() -> semua tetap hidup
    groups = build_param_groups(model, "S0", base_lr=3e-4)

    assert len(groups) == 1
    assert abs(groups[0]["lr"] - 3e-4) < 1e-9
    for p in groups[0]["params"]:
        assert p.requires_grad
