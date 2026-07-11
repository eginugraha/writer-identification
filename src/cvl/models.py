import timm
import torch
from .config import ALL_ARCHITECTURES

def build_model(arch_key: str, num_classes: int, pretrained: bool):
    return timm.create_model(
        ALL_ARCHITECTURES[arch_key], pretrained=pretrained, num_classes=num_classes
    )

def forward_features(model, x):
    feats = model.forward_features(x)
    # samakan ke [B, D]: pool spatial/token via head pre_logits jika tersedia
    if hasattr(model, "forward_head"):
        pooled = model.forward_head(feats, pre_logits=True)
    else:
        pooled = feats.mean(dim=tuple(range(2, feats.dim()))) if feats.dim() > 2 else feats
    return pooled

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
