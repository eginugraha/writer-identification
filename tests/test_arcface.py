import torch
from src.cvl.arcface import ArcFaceHead, ArcFaceModel


def test_head_bentuk_logit():
    h = ArcFaceHead(in_features=16, num_classes=5)
    out = h(torch.randn(4, 16))
    assert out.shape == (4, 5)


def test_head_margin_menurunkan_logit_kelas_benar():
    """Dengan label diberikan, logit kelas target dikurangi margin sudut."""
    torch.manual_seed(0)
    h = ArcFaceHead(in_features=16, num_classes=5, s=30.0, m=0.3)
    feats = torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 3])
    tanpa = h(feats)
    dengan = h(feats, labels)
    idx = torch.arange(4)
    assert (dengan[idx, labels] < tanpa[idx, labels]).all()


def test_set_margin_nol_menyamai_kosinus_polos():
    torch.manual_seed(0)
    h = ArcFaceHead(in_features=16, num_classes=5, s=30.0, m=0.3)
    h.set_margin(0.0)
    feats = torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 3])
    assert torch.allclose(h(feats, labels), h(feats), atol=1e-5)


def test_model_forward_tanpa_label():
    import timm
    backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
    m = ArcFaceModel(backbone, ArcFaceHead(backbone.num_features, 5)).eval()
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 5)


def test_model_forward_features_dua_dimensi():
    import timm
    backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
    m = ArcFaceModel(backbone, ArcFaceHead(backbone.num_features, 5)).eval()
    f = m.forward_features(torch.randn(2, 3, 224, 224))
    assert f.dim() == 2 and f.shape[0] == 2
