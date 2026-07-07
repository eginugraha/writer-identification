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
