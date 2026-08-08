"""Uji unit untuk src/cvl/finetune.py pada model timm sungguhan.

Menguji mekanisme freeze/LLRD langsung pada model, bukan lewat training,
supaya cepat dan tidak bergantung pada dataset. Tanpa berkas ini, skenario
FT3 bisa diam-diam mendegradasi jadi FT0 (tidak ada yang dibekukan, satu
grup LR seragam) dan test training tetap lolos karena hanya memeriksa
`epochs_ran >= 1`.

Semuanya dijalankan untuk setiap arsitektur di LAYER_MAP. Tes berbasis nama
palsu ada di tests/test_finetune_arch.py dan jalan tanpa timm; berkas ini
yang membuktikan peta lapisannya cocok dengan model asli.
"""
import pytest
import timm
from src.cvl.config import ALL_ARCHITECTURES
from src.cvl.finetune import (
    LAYER_MAP, STRATEGIES, freeze_layers, build_param_groups, freeze_prefixes,
)

ARCHS = sorted(LAYER_MAP)


def _model(arch):
    # pretrained=False supaya tidak ada unduhan; num_classes sesuai studi.
    return timm.create_model(ALL_ARCHITECTURES[arch], pretrained=False, num_classes=308)


@pytest.mark.parametrize("arch", ARCHS)
def test_freeze_layers_s3_membekukan_prefix_yang_benar(arch):
    model = _model(arch)
    freeze_layers(model, "S3", arch=arch)
    frozen_prefixes = freeze_prefixes(arch, "S3")

    beku = [name for name, p in model.named_parameters() if not p.requires_grad]
    hidup = [name for name, p in model.named_parameters() if p.requires_grad]

    # Arah 1: semua parameter berprefix stem + 2 stage pertama harus beku.
    for name, p in model.named_parameters():
        if name.startswith(frozen_prefixes):
            assert not p.requires_grad, f"{name} seharusnya beku di S3"

    # Arah 2: minimal satu parameter DI LUAR prefix tersebut harus tetap
    # hidup — kalau tidak, test "arah 1" saja lolos meski semuanya dibekukan.
    assert any(not name.startswith(frozen_prefixes) for name in hidup)
    # dan kebalikannya: tidak semua parameter ikut beku.
    assert len(hidup) > 0
    assert len(beku) > 0


@pytest.mark.parametrize("arch", ARCHS)
def test_build_param_groups_s3_llrd_menurun_sesuai_urutan(arch):
    model = _model(arch)
    freeze_layers(model, "S3", arch=arch)
    groups = build_param_groups(model, "S3", base_lr=1e-4, arch=arch)

    # lebih dari satu grup: baseline (S0) hanya satu, S3 harus terdiferensiasi.
    assert len(groups) > 1

    lrs = [g["lr"] for g in groups]
    # menurun tegas mengikuti llrd_order (head -> stage terdalam -> ... -> stem)
    assert all(a > b for a, b in zip(lrs, lrs[1:]))

    # Nilai yang didokumentasikan di rencana: head 1e-4, stage-3 7e-5, stage-2 4.9e-5.
    assert abs(lrs[0] - 1.0e-4) < 1e-9
    assert abs(lrs[1] - 7.0e-5) < 1e-9
    assert abs(lrs[2] - 4.9e-5) < 1e-9

    # Tidak ada grup yang menyelundupkan parameter beku (requires_grad=False).
    for g in groups:
        for p in g["params"]:
            assert p.requires_grad


@pytest.mark.parametrize("arch", ARCHS)
def test_build_param_groups_s0_tetap_satu_grup_seragam(arch):
    model = _model(arch)
    freeze_layers(model, "S0", arch=arch)  # S0: tidak membekukan apa pun
    groups = build_param_groups(model, "S0", base_lr=3e-4, arch=arch)

    assert len(groups) == 1
    assert abs(groups[0]["lr"] - 3e-4) < 1e-9
    for p in groups[0]["params"]:
        assert p.requires_grad


@pytest.mark.parametrize("arch", ARCHS)
def test_s1_menyisakan_hanya_kepala(arch):
    """S1 = feature extraction: seluruh backbone beku, hanya kepala yang latih."""
    model = _model(arch)
    freeze_layers(model, "S1", arch=arch)
    hidup = [n for n, p in model.named_parameters() if p.requires_grad]
    assert hidup, "S1 membekukan segalanya — kepala pun ikut, tidak ada yang bisa dilatih"
    assert not any(n.startswith(freeze_prefixes(arch, "S1")) for n in hidup)
    n_hidup = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    assert n_hidup < 0.5 * n_total, "S1 seharusnya melatih sebagian kecil parameter saja"


@pytest.mark.parametrize("arch", ARCHS)
def test_strategi_beku_benar_benar_menghentikan_gradien(arch):
    """Bukti akhir: parameter beku tidak boleh punya .grad setelah backward."""
    import torch
    model = _model(arch)
    freeze_layers(model, "S3", arch=arch)
    out = model(torch.randn(2, 3, 224, 224))
    torch.nn.functional.cross_entropy(out, torch.tensor([1, 2])).backward()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, f"{n} beku tapi tetap menerima gradien"


def test_semua_strategi_punya_kunci_yang_sama():
    """Menambah strategi tanpa salah satu kunci akan meledak jauh di dalam run."""
    kunci = {"n_freeze", "base_lr", "llrd_decay", "label_smoothing"}
    for nama, cfg in STRATEGIES.items():
        assert set(cfg) == kunci, f"{nama}: kunci strategi tidak lengkap"
