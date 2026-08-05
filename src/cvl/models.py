import timm
import torch
from .arcface import ArcFaceHead, ArcFaceModel
from .config import ALL_ARCHITECTURES


def build_model(arch_key: str, num_classes: int, pretrained: bool,
                drop_path: float = 0.0, head: str = "linear"):
    name = ALL_ARCHITECTURES[arch_key]
    if head == "linear":
        return timm.create_model(name, pretrained=pretrained,
                                 num_classes=num_classes,
                                 drop_path_rate=drop_path)
    if head == "arcface":
        backbone = timm.create_model(name, pretrained=pretrained,
                                     num_classes=0, drop_path_rate=drop_path)
        return ArcFaceModel(backbone, ArcFaceHead(backbone.num_features, num_classes))
    raise ValueError(f"head tidak dikenal: {head}")


def set_arcface_margin(model, m: float) -> None:
    """Setel margin bila model memakai head ArcFace; selain itu tidak apa-apa."""
    if isinstance(model, ArcFaceModel):
        model.head.set_margin(m)


def forward_features(model, x):
    if isinstance(model, ArcFaceModel):
        return model.forward_features(x)
    feats = model.forward_features(x)
    # samakan ke [B, D]: pool spatial/token via head pre_logits jika tersedia
    if hasattr(model, "forward_head"):
        pooled = model.forward_head(feats, pre_logits=True)
    else:
        pooled = feats.mean(dim=tuple(range(2, feats.dim()))) if feats.dim() > 2 else feats
    return pooled


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
