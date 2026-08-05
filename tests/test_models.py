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
