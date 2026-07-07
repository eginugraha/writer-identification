import torch
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.dataset import LineDataset, build_transforms

def test_transform_shape():
    from PIL import Image
    t = build_transforms(train=False)
    out = t(Image.new("RGB", (300, 60)))
    assert out.shape == (3, 224, 224)

def test_dataset_item(tiny_lines):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    train = m[m.split == "train"]
    ds = LineDataset(train, train=True)
    x, y = ds[0]
    assert x.shape == (3, 224, 224) and isinstance(y, int)
    assert 0 <= y < 2
