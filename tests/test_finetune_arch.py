"""Strategi freeze/LLRD harus mengikuti pohon modul arsitekturnya.

STRATEGIES lama menuliskan prefiks ConvNeXt secara harfiah ("stem",
"stages.0"). Dipakai pada Swin-Tiny -- yang menamai bloknya "patch_embed"
dan "layers.N" -- tidak ada prefiks yang cocok, sehingga:

  * freeze_layers membekukan NOL parameter, dan
  * _prefix_of melempar semua parameter ke bucket "head" sehingga LLRD
    runtuh jadi satu grup seragam.

Run-nya tetap selesai dan menghasilkan angka yang masuk akal, jadi FT3 diam-
diam berubah jadi "FT0 dengan LR 1e-4". Tes di bawah mengunci dua hal: peta
lapisan per arsitektur, dan kegagalan keras saat prefiks tidak cocok dengan
satu parameter pun.
"""
import importlib.util
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cvl.finetune import (
    STRATEGIES, freeze_prefixes, llrd_order, freeze_layers, build_param_groups,
)

# Nama parameter yang mewakili tiap arsitektur (subset yang cukup untuk diuji).
PARAM_CONVNEXT = [
    "stem.0.weight", "stages.0.blocks.0.conv_dw.weight",
    "stages.1.blocks.0.conv_dw.weight", "stages.2.blocks.0.conv_dw.weight",
    "stages.3.blocks.0.conv_dw.weight", "norm_pre.weight", "head.fc.weight",
]
PARAM_SWIN = [
    "patch_embed.proj.weight", "layers.0.blocks.0.attn.qkv.weight",
    "layers.1.blocks.0.attn.qkv.weight", "layers.2.blocks.0.attn.qkv.weight",
    "layers.3.blocks.0.attn.qkv.weight", "norm.weight", "head.fc.weight",
]


def _model_palsu(nama_param):
    """Modul bersarang yang named_parameters()-nya persis `nama_param`."""
    root = torch.nn.Module()
    for penuh in nama_param:
        *jalur, daun = penuh.split(".")
        cur = root
        for bagian in jalur:
            anak = cur._modules.get(bagian)
            if anak is None:
                anak = torch.nn.Module()
                cur.add_module(bagian, anak)
            cur = anak
        cur.register_parameter(daun, torch.nn.Parameter(torch.zeros(2)))
    return root


def _dibekukan(model):
    return sorted(n for n, p in model.named_parameters() if not p.requires_grad)


# --------------------------------------------------------------------------
# peta lapisan
# --------------------------------------------------------------------------

def test_prefiks_freeze_convnext_tetap_seperti_semula():
    """Regression: perilaku ConvNeXt tidak boleh berubah -- FT0..FT4 sudah
    dirancang atasnya dan baseline-nya disalin dari grid utama."""
    assert freeze_prefixes("convnext_tiny", "S3") == ("stem", "stages.0", "stages.1")


def test_prefiks_freeze_swin_memakai_patch_embed_dan_layers():
    assert freeze_prefixes("swin_tiny", "S3") == ("patch_embed", "layers.0", "layers.1")


def test_s1_membekukan_seluruh_backbone():
    assert freeze_prefixes("swin_tiny", "S1") == (
        "patch_embed", "layers.0", "layers.1", "layers.2", "layers.3")


def test_s0_tidak_membekukan_apa_pun():
    assert freeze_prefixes("swin_tiny", "S0") == ()


def test_arsitektur_di_luar_peta_ditolak():
    """Lebih baik gagal daripada diam-diam tidak membekukan apa pun."""
    with pytest.raises(ValueError, match="resnet50"):
        freeze_prefixes("resnet50", "S3")


def test_llrd_urut_dari_kepala_ke_lapisan_awal():
    assert llrd_order("swin_tiny") == (
        "head", "layers.3", "layers.2", "layers.1", "layers.0", "patch_embed")


# --------------------------------------------------------------------------
# guard: prefiks yang tidak cocok harus meledak
# --------------------------------------------------------------------------

def test_freeze_menolak_model_yang_tidak_cocok_dengan_arch():
    """Inti bug-nya: arch salah tidak boleh berakhir 'nol parameter dibekukan'."""
    model = _model_palsu(PARAM_SWIN)
    with pytest.raises(ValueError, match="tidak cocok"):
        freeze_layers(model, "S3", arch="convnext_tiny")


def test_freeze_swin_membekukan_patch_embed_dan_dua_stage_pertama():
    model = _model_palsu(PARAM_SWIN)
    freeze_layers(model, "S3", arch="swin_tiny")
    assert _dibekukan(model) == [
        "layers.0.blocks.0.attn.qkv.weight",
        "layers.1.blocks.0.attn.qkv.weight",
        "patch_embed.proj.weight",
    ]


def test_freeze_convnext_masih_membekukan_stem_dan_dua_stage_pertama():
    model = _model_palsu(PARAM_CONVNEXT)
    freeze_layers(model, "S3", arch="convnext_tiny")
    assert _dibekukan(model) == [
        "stages.0.blocks.0.conv_dw.weight",
        "stages.1.blocks.0.conv_dw.weight",
        "stem.0.weight",
    ]


# --------------------------------------------------------------------------
# LLRD
# --------------------------------------------------------------------------

def test_llrd_swin_memberi_lr_menurun_ke_lapisan_awal():
    model = _model_palsu(PARAM_SWIN)
    freeze_layers(model, "S3", arch="swin_tiny")
    grup = build_param_groups(model, "S3", base_lr=3e-4, arch="swin_tiny")
    lr = sorted(g["lr"] for g in grup)
    assert len(grup) == 3, "head+norm, layers.3, layers.2 -- sisanya beku"
    assert lr == sorted(set(lr)), "LLRD runtuh: ada grup dengan LR sama"
    assert max(lr) == pytest.approx(STRATEGIES["S3"]["base_lr"])
    assert min(lr) == pytest.approx(STRATEGIES["S3"]["base_lr"] * 0.7 ** 2)


def test_llrd_tidak_memasukkan_parameter_beku():
    model = _model_palsu(PARAM_SWIN)
    freeze_layers(model, "S3", arch="swin_tiny")
    grup = build_param_groups(model, "S3", base_lr=3e-4, arch="swin_tiny")
    n = sum(len(g["params"]) for g in grup)
    # layers.2, layers.3, norm, head.fc -- patch_embed/layers.0/layers.1 beku
    assert n == 4, "hanya layers.2, layers.3, dan kepala yang boleh ikut"


def test_s2_tanpa_llrd_tetap_satu_grup():
    model = _model_palsu(PARAM_SWIN)
    freeze_layers(model, "S2", arch="swin_tiny")
    grup = build_param_groups(model, "S2", base_lr=3e-4, arch="swin_tiny")
    assert len(grup) == 1 and grup[0]["lr"] == pytest.approx(1e-4)


# --------------------------------------------------------------------------
# peta harus cocok dengan model timm sungguhan (hanya jalan di pod)
# --------------------------------------------------------------------------

butuh_timm = pytest.mark.skipif(
    importlib.util.find_spec("timm") is None,
    reason="timm hanya terpasang di pod GPU -- tes ini wajib dijalankan di sana",
)


@butuh_timm
@pytest.mark.parametrize("arch,timm_name", [
    ("convnext_tiny", "convnext_tiny"),
    ("swin_tiny", "swin_tiny_patch4_window7_224"),
])
def test_peta_lapisan_cocok_dengan_model_timm_asli(arch, timm_name):
    """Peta di atas ditulis dari ingatan soal penamaan timm; ini yang
    membuktikannya. Tiap prefiks wajib cocok dengan minimal satu parameter."""
    import timm
    model = timm.create_model(timm_name, pretrained=False, num_classes=10)
    nama = [n for n, _ in model.named_parameters()]
    for pfx in llrd_order(arch):
        assert any(n.startswith(pfx) for n in nama), (
            f"{timm_name}: tidak ada parameter berawalan '{pfx}' -- peta lapisan salah")
