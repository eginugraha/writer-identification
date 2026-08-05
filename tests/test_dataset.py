import torch
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.dataset import LineDataset, build_transforms
from src.cvl.dataset import ResizeHeight

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


def test_resize_height_menjaga_rasio(wide_line_image):
    out = ResizeHeight(224)(wide_line_image)
    assert out.size[1] == 224
    # 1740/140 * 224 = 2784
    assert out.size[0] == 2784


def test_resize_height_tidak_pernah_lebih_sempit_dari_tinggi():
    from PIL import Image
    out = ResizeHeight(224)(Image.new("RGB", (10, 500)))
    assert out.size[0] >= 224 and out.size[1] == 224


def test_geometri_center_identik_dengan_perilaku_lama(wide_line_image):
    """Penjaga utama: FT0 harus sama persis dengan pipeline sebelum perubahan."""
    import torch
    torch.manual_seed(0)
    a = build_transforms(train=False)(wide_line_image)
    torch.manual_seed(0)
    b = build_transforms(train=False, geometry="center", aug="baseline")(wide_line_image)
    assert torch.equal(a, b)
    assert a.shape == (3, 224, 224)


def test_linewindow_menghasilkan_jendela_berbeda(wide_line_image):
    """RandomResizedCrop lama selalu mengembalikan potongan identik; ini
    memastikan geometri baru benar-benar mengacak posisi."""
    import torch
    t = build_transforms(train=True, geometry="linewindow")
    outs = []
    for seed in range(8):
        torch.manual_seed(seed)
        outs.append(t(wide_line_image))
    assert all(o.shape == (3, 224, 224) for o in outs)
    unik = {o.numpy().tobytes() for o in outs}
    assert len(unik) > 1, "jendela linewindow tidak pernah berpindah"


def test_linewindow_eval_bentuk_benar(wide_line_image):
    t = build_transforms(train=False, geometry="linewindow")
    assert t(wide_line_image).shape == (3, 224, 224)


def test_geometri_tidak_dikenal_ditolak(wide_line_image):
    import pytest
    with pytest.raises(ValueError):
        build_transforms(train=True, geometry="entahlah")


def test_aug_tidak_dikenal_ditolak(wide_line_image):
    import pytest
    with pytest.raises(ValueError):
        build_transforms(train=True, aug="entahlah")
