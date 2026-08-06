import pytest
import torch
from src.cvl.models import build_model, forward_features, count_params

def test_forward_logits_shape():
    m = build_model("resnet50", num_classes=7, pretrained=False).eval()
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 7)

def test_forward_features_shape():
    m = build_model("resnet50", num_classes=7, pretrained=False).eval()
    f = forward_features(m, torch.randn(2, 3, 224, 224))
    assert f.dim() == 2 and f.shape[0] == 2

def test_count_params_positive():
    m = build_model("vit_small", num_classes=5, pretrained=False)
    assert count_params(m) > 1_000_000

def test_drop_path_diteruskan():
    m = build_model("convnext_tiny", num_classes=7, pretrained=False, drop_path=0.2)
    rates = [mod.drop_prob for mod in m.modules()
             if mod.__class__.__name__ == "DropPath"]
    assert rates and max(rates) > 0.0


def test_drop_path_default_nol():
    m = build_model("convnext_tiny", num_classes=7, pretrained=False)
    rates = [mod.drop_prob for mod in m.modules()
             if mod.__class__.__name__ == "DropPath"]
    assert not rates or max(rates) == 0.0


def test_swin_drop_path_default_sama_dengan_timm_tanpa_kwarg():
    """convnext_tiny kebal terhadap bug ini karena default timm-nya memang 0.

    swin_tiny berbeda: `SwinTransformer.__init__` default `drop_path_rate=0.1`.
    Kalau `build_model` selalu meneruskan `drop_path_rate=0.0` secara eksplisit,
    stochastic depth bawaan Swin mati diam-diam. Bandingkan langsung dengan
    `timm.create_model` tanpa kwarg `drop_path_rate` sama sekali -- itu acuan
    perilaku pra-branch yang wajib direproduksi oleh jalur default."""
    import timm
    acuan = timm.create_model("swin_tiny_patch4_window7_224",
                              pretrained=False, num_classes=7)
    rates_acuan = [mod.drop_prob for mod in acuan.modules()
                   if mod.__class__.__name__ == "DropPath"]

    m = build_model("swin_tiny", num_classes=7, pretrained=False)
    rates = [mod.drop_prob for mod in m.modules()
             if mod.__class__.__name__ == "DropPath"]

    assert rates and max(rates) == pytest.approx(0.1)
    assert max(rates) == pytest.approx(max(rates_acuan))


def test_swin_drop_path_eksplisit_tetap_dihormati():
    m = build_model("swin_tiny", num_classes=7, pretrained=False, drop_path=0.2)
    rates = [mod.drop_prob for mod in m.modules()
             if mod.__class__.__name__ == "DropPath"]
    assert rates and max(rates) == pytest.approx(0.2)


def test_head_arcface_bentuk_logit():
    m = build_model("resnet50", num_classes=7, pretrained=False, head="arcface").eval()
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 7)


def test_forward_features_bekerja_pada_arcface():
    m = build_model("resnet50", num_classes=7, pretrained=False, head="arcface").eval()
    f = forward_features(m, torch.randn(2, 3, 224, 224))
    assert f.dim() == 2 and f.shape[0] == 2


def test_set_arcface_margin_aman_untuk_head_linear():
    from src.cvl.models import set_arcface_margin
    m = build_model("resnet50", num_classes=7, pretrained=False)
    set_arcface_margin(m, 0.1)  # tidak boleh melempar
