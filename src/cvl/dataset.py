from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class ResizeHeight:
    """Setel tinggi ke `height`, jaga rasio aspek.

    Berbeda dari `T.Resize(224)` yang menyetel sisi *pendek*: pada citra baris
    CVL (rasio ~12:1) keduanya kebetulan sama, tapi ResizeHeight tetap benar
    untuk baris yang tidak wajar (lebih tinggi daripada lebar) karena hasilnya
    dijamin tidak lebih sempit dari `height`.
    """

    def __init__(self, height: int):
        self.height = height

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w = max(self.height, int(round(w * self.height / h)))
        return img.resize((new_w, self.height), Image.BILINEAR)


def _geometry_stage(train: bool, image_size: int, geometry: str):
    """Tahap yang menentukan *bagian mana* dari baris yang dilihat model."""
    if geometry == "center":
        # Perilaku lama. T.Resize(int) menyetel sisi pendek; pada baris 12:1
        # hasilnya ~3284x224 dan crop 224 mengambil bagian tengah saja.
        if train:
            return [T.Resize(image_size)]
        return [T.Resize(image_size), T.CenterCrop(image_size)]
    if geometry == "linewindow":
        # Tinggi dipaskan ke image_size, lalu jendela selebar image_size
        # diambil acak (latih) atau di tengah (uji, saat eval_crops=1).
        if train:
            return [ResizeHeight(image_size), T.RandomCrop(image_size)]
        return [ResizeHeight(image_size), T.CenterCrop(image_size)]
    raise ValueError(f"geometry tidak dikenal: {geometry}")


def _strip_width(image_size: int) -> int:
    """Lebar strip tengah yang dilihat pipeline baseline.

    Torchvision jatuh ke fallback `w = round(h * max(ratio))` ketika batasan
    RandomResizedCrop tidak terpenuhi. Dengan ratio maks 1.1 dan tinggi 224,
    lebarnya 246 — angka ini yang dipakai AUG supaya cakupan barisnya sama
    persis dengan baseline.
    """
    return int(round(image_size * 1.1))


def _aug_stage(geometry: str, image_size: int, aug: str):
    """Tahap augmentasi (hanya dipakai saat train=True)."""
    if aug == "baseline":
        if geometry == "center":
            return [
                T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        # linewindow: jendela sudah 224x224, affine + jitter saja
        return [
            T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    if aug == "strong":
        steps = [
            T.RandomAffine(degrees=6, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        ]
        if geometry == "center":
            # Potong dulu ke strip tengah 224x246 — wilayah yang sama dengan
            # baseline — supaya RandomResizedCrop punya ruang untuk memenuhi
            # batasan rasio dan benar-benar mengacak.
            steps.append(T.CenterCrop((image_size, _strip_width(image_size))))
        steps += [
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
            T.ColorJitter(brightness=0.4, contrast=0.4),
        ]
        return steps
    raise ValueError(f"aug tidak dikenal: {aug}")


def build_transforms(train: bool, image_size: int = IMAGE_SIZE,
                     geometry: str = "center", aug: str = "baseline"):
    """PIL -> Tensor [3, image_size, image_size].

    `geometry` mengatur bagian mana dari baris yang terlihat; `aug` mengatur
    seberapa keras citra diacak. Dua sumbu ini sengaja dipisah agar skenario
    FT1 dan AUG menguji mekanisme yang berbeda tanpa saling mencemari.
    """
    # Validasi tanpa syarat: nilai `aug` salah ketik harus ditolak baik saat
    # train maupun eval, bukan hanya saat _aug_stage benar-benar dipanggil.
    if aug not in ("baseline", "strong"):
        raise ValueError(f"aug tidak dikenal: {aug}")
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    steps = [T.Grayscale(num_output_channels=3)]
    steps += _geometry_stage(train, image_size, geometry)
    if train:
        steps += _aug_stage(geometry, image_size, aug)
    steps += [T.ToTensor()]
    if train and aug == "strong":
        # RandomErasing bekerja pada tensor, bukan PIL, jadi harus setelah
        # ToTensor.
        steps.append(T.RandomErasing(p=0.25))
    steps += [norm]
    return T.Compose(steps)


class LineDataset(Dataset):
    def __init__(self, manifest_subset, train: bool,
                 geometry: str = "center", aug: str = "baseline",
                 eval_crops: int = 1):
        self.rows = manifest_subset.reset_index(drop=True)
        self.train = train
        self.geometry = geometry
        self.eval_crops = eval_crops
        self.tf = build_transforms(train, geometry=geometry, aug=aug)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), int(r["label"])
