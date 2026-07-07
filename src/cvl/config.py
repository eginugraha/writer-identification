from pathlib import Path

EXCLUDE_WRITERS = {"0431", "0161"}
MIN_PAGES = 5
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ARCHITECTURES = {
    "resnet50": "resnet50",
    "convnext_tiny": "convnext_tiny",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "vit_small": "vit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}
ABLATION_LEVELS = [1, 2, 3, 4, None]  # None = full
SEEDS = [0, 1, 2]

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def lines_root() -> Path:
    return project_root() / "cvl-database-1-1"
