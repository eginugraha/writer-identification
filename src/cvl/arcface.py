"""Head ArcFace (additive angular margin) untuk skenario FT4.

Writer identification pada L1 adalah 308 identitas dengan ~7 contoh per kelas
— rezim tempat head margin lazim dipakai. `s` dan `m` dipatok di muka pada
nilai standar papernya dan tidak disetel setelah melihat hasil.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int,
                 s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def set_margin(self, m: float) -> None:
        """Dipakai untuk menaikkan margin bertahap selama epoch warmup."""
        self.m = m

    def forward(self, feats, labels=None):
        cos = F.linear(F.normalize(feats), F.normalize(self.weight))
        if labels is None or self.m == 0.0:
            return self.s * cos
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos)
        margin = torch.zeros_like(theta)
        margin.scatter_(1, labels.view(-1, 1), self.m)
        return self.s * torch.cos(theta + margin)


class ArcFaceModel(nn.Module):
    """Backbone timm (num_classes=0) + head ArcFace.

    `forward` menerima `labels` opsional supaya margin hanya aktif saat latih;
    tanpa label ia mengembalikan skor kosinus terskala yang aman di-softmax
    saat evaluasi.
    """

    def __init__(self, backbone, head: ArcFaceHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, labels=None):
        return self.head(self.backbone(x), labels)

    def forward_features(self, x):
        return self.backbone(x)
