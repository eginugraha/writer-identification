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


def test_gradien_tetap_finite_saat_cos_persis_satu_dalam_fp16():
    """Di bawah AMP (torch.amp.autocast), F.linear mengeluarkan fp16. Fp16
    tidak bisa merepresentasikan 1.0 - 1e-7 -- ia dibulatkan jadi 1.0 persis,
    sehingga clamp(-1+1e-7, 1-1e-7) jadi no-op di titik itu, acos(1.0)
    berada tepat di batas domainnya, dan gradiennya meledak jadi tak-hingga.

    Disimulasikan langsung dengan tensor fp16 (bukan lewat autocast CUDA,
    yang tidak tersedia di CPU) supaya reproducible tanpa GPU: baris weight
    kelas target disamakan persis dengan feats-nya sehingga kosinusnya
    membulat ke 1.0 dalam fp16."""
    torch.manual_seed(0)
    h = ArcFaceHead(in_features=8, num_classes=2, s=30.0, m=0.3).half()
    feats = torch.randn(2, 8, dtype=torch.float16, requires_grad=True)
    with torch.no_grad():
        h.weight[0].copy_(feats[0])  # sejajar persis -> cos membulat ke 1.0
    labels = torch.tensor([0, 1])
    out = h(feats, labels)
    out.sum().backward()
    assert torch.isfinite(feats.grad).all()
