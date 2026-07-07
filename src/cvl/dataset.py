from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

def build_transforms(train: bool, image_size: int = IMAGE_SIZE):
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

class LineDataset(Dataset):
    def __init__(self, manifest_subset, train: bool):
        self.rows = manifest_subset.reset_index(drop=True)
        self.tf = build_transforms(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), int(r["label"])
