import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from src.cvl.config import IMAGENET_MEAN, IMAGENET_STD
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.dataset import LineDataset, build_transforms
from src.cvl.dataset import ResizeHeight


def _wide_line_image_bertekstur():
    """Citra baris 1740x140 dengan tekstur acak (bukan warna polos).

    `wide_line_image` (fixture bersama) warnanya rata, sehingga rotasi,
    translasi, dan skala kecil dari RandomAffine tidak mengubah satu piksel
    pun di area yang ter-crop -- dua pipeline yang berbeda parameternya bisa
    menghasilkan tensor yang sama persis secara kebetulan. Citra ini dipakai
    khusus di test penjaga geometri "center" supaya perbedaan parameter
    benar-benar terlihat di outputnya.
    """
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(140, 1740, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")

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


def _pipeline_pra_refactor(train: bool, image_size: int = 224):
    """Salinan literal build_transforms sebelum refactor Task 3.

    Sengaja diduplikasi di dalam test: gunanya justru sebagai pembanding
    independen, supaya kesalahan di _geometry_stage/_aug_stage tidak ikut
    tercermin di sisi acuan.
    """
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(image_size),
            T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(), norm,
        ])
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(), norm,
    ])


def test_geometri_center_identik_dengan_perilaku_lama_saat_train():
    """Penjaga utama (train=True): bandingkan dengan reproduksi literal dan
    independen dari pipeline pra-refactor, bukan dengan build_transforms
    versi baru itu sendiri, agar bug di _geometry_stage/_aug_stage tidak
    lolos hanya karena kedua sisi salah dengan cara yang sama.

    Memakai citra bertekstur (bukan `wide_line_image` yang warnanya rata)
    supaya perubahan parameter RandomAffine/urutan tahap benar-benar
    mengubah nilai piksel, bukan hanya bentuk tensor."""
    img = _wide_line_image_bertekstur()
    torch.manual_seed(0)
    a = _pipeline_pra_refactor(train=True)(img)
    torch.manual_seed(0)
    b = build_transforms(train=True, geometry="center", aug="baseline")(img)
    assert torch.equal(a, b)
    assert a.shape == (3, 224, 224)


def test_geometri_center_identik_dengan_perilaku_lama_saat_eval():
    """Penjaga utama (train=False): sama seperti di atas, jalur eval."""
    img = _wide_line_image_bertekstur()
    torch.manual_seed(0)
    a = _pipeline_pra_refactor(train=False)(img)
    torch.manual_seed(0)
    b = build_transforms(train=False, geometry="center", aug="baseline")(img)
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


def test_geometri_tidak_dikenal_ditolak_saat_train(wide_line_image):
    import pytest
    with pytest.raises(ValueError):
        build_transforms(train=True, geometry="entahlah")


def test_geometri_tidak_dikenal_ditolak_saat_eval(wide_line_image):
    import pytest
    with pytest.raises(ValueError):
        build_transforms(train=False, geometry="entahlah")


def test_aug_tidak_dikenal_ditolak_saat_train(wide_line_image):
    import pytest
    with pytest.raises(ValueError):
        build_transforms(train=True, aug="entahlah")


def test_aug_tidak_dikenal_ditolak_saat_eval(wide_line_image):
    """aug harus divalidasi tanpa syarat, bukan hanya saat _aug_stage
    benar-benar dipanggil (yaitu hanya saat train=True)."""
    import pytest
    with pytest.raises(ValueError):
        build_transforms(train=False, aug="entahlah")


def test_aug_strong_bentuk_benar(wide_line_image):
    import torch
    torch.manual_seed(0)
    out = build_transforms(train=True, aug="strong")(wide_line_image)
    assert out.shape == (3, 224, 224)


def test_aug_strong_benar_benar_mengacak():
    """Crop pada AUG dipasang setelah strip tengah 224x246 supaya batasan
    rasio bisa dipenuhi — di geometri baseline crop selalu identik.

    Memakai citra bertekstur (bukan `wide_line_image` yang warnanya rata)
    supaya keacakan RandomResizedCrop/RandomErasing benar-benar terlihat di
    nilai piksel, bukan hanya lewat ColorJitter yang tetap mengubah warna
    solid."""
    img = _wide_line_image_bertekstur()
    t = build_transforms(train=True, aug="strong")
    outs = []
    for seed in range(8):
        torch.manual_seed(seed)
        outs.append(t(img))
    assert len({o.numpy().tobytes() for o in outs}) > 1


def test_aug_strong_tidak_mengubah_geometri(wide_line_image):
    """AUG hanya boleh menaikkan kekuatan augmentasi, bukan memperluas
    bagian baris yang terlihat — kalau tidak, ia rancu dengan FT1."""
    from src.cvl.dataset import _strip_width
    assert _strip_width(224) == 246


def test_aug_strong_center_menyematkan_strip_sebelum_crop():
    """Penjaga anti-rancu: pastikan _aug_stage("center", 224, "strong")
    benar-benar memasang CenterCrop 224x246 SEBELUM RandomResizedCrop,
    bukan cuma bahwa _strip_width() menghitung 246 dengan benar (itu
    hanya aritmetika berdiri sendiri, tidak membuktikan apa pun tentang
    daftar tahap yang sesungguhnya dipakai). Kalau CenterCrop ini hilang
    atau tertukar urutan dengan RandomResizedCrop, sumbu aug jadi ikut
    memperluas cakupan baris dan rancu dengan sumbu geometry (FT1)."""
    from src.cvl.dataset import _aug_stage

    steps = _aug_stage("center", 224, "strong")
    center_crop_idx = [i for i, s in enumerate(steps) if isinstance(s, T.CenterCrop)]
    resized_crop_idx = [i for i, s in enumerate(steps) if isinstance(s, T.RandomResizedCrop)]
    assert len(center_crop_idx) == 1, "harus ada tepat satu CenterCrop penyemat strip"
    assert len(resized_crop_idx) == 1, "harus ada tepat satu RandomResizedCrop"
    assert steps[center_crop_idx[0]].size == (224, 246)
    assert center_crop_idx[0] < resized_crop_idx[0], (
        "CenterCrop harus dipasang SEBELUM RandomResizedCrop, kalau tidak "
        "strip tengah tidak benar-benar menyemat wilayah yang terlihat"
    )


def test_eval_crops_menghasilkan_tumpukan(tiny_lines):
    from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    test = m[m.split == "test"]
    ds = LineDataset(test, train=False, geometry="linewindow", eval_crops=9)
    x, y = ds[0]
    assert x.shape == (9, 3, 224, 224) and isinstance(y, int)


def test_eval_crops_satu_tetap_tiga_dimensi(tiny_lines):
    from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    ds = LineDataset(m[m.split == "test"], train=False, eval_crops=1)
    x, _ = ds[0]
    assert x.shape == (3, 224, 224)


def test_jendela_evaluasi_merata_dan_berbeda():
    """Jendela harus benar-benar berasal dari posisi berbeda dan menutup
    baris dari ujung ke ujung. Diuji dengan citra bergradien — pada citra
    polos semua potongan identik sehingga uji isi tidak membuktikan apa pun."""
    from PIL import Image
    from src.cvl.dataset import even_windows
    grad = Image.new("L", (1740, 140))
    grad.putdata([x % 256 for _ in range(140) for x in range(1740)])
    grad = grad.convert("RGB")

    wins = even_windows(grad, size=224, k=9)
    assert len(wins) == 9
    assert all(w.size == (224, 140) for w in wins)
    # kesembilan jendela berbeda isinya
    assert len({w.tobytes() for w in wins}) == 9
    # jendela pertama mulai di kiri, terakhir berakhir di kanan
    assert wins[0].tobytes() == grad.crop((0, 0, 224, 140)).tobytes()
    assert wins[-1].tobytes() == grad.crop((1740 - 224, 0, 1740, 140)).tobytes()


def test_aug_strong_linewindow_tidak_memakai_centercrop_tambahan():
    """Mirror dari test di atas: pada geometry="linewindow" jendela sudah
    224x224 (hasil ResizeHeight + RandomCrop di _geometry_stage), jadi
    _aug_stage TIDAK boleh menambahkan CenterCrop lagi — itu akan
    mempersempit cakupan, bukan menyematkannya, dan mengubah wilayah yang
    terlihat model secara tak sengaja."""
    from src.cvl.dataset import _aug_stage

    steps = _aug_stage("linewindow", 224, "strong")
    assert not any(isinstance(s, T.CenterCrop) for s in steps)
