import csv
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_image_mask_pairs(root_path):
    root_path = Path(root_path)
    image_paths = sorted(path for path in root_path.glob("train_images/*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    mask_paths = sorted(path for path in root_path.glob("train_masks/*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Found {len(image_paths)} images but {len(mask_paths)} masks under {root_path}."
        )
    if not image_paths:
        raise ValueError(f"No training pairs found under {root_path}.")
    return image_paths, mask_paths


def split_dataset(root_path, train_ratio=0.8, seed=42):
    image_paths, mask_paths = get_image_mask_pairs(root_path)
    indices = list(range(len(image_paths)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    num_train = int(len(indices) * train_ratio)
    train_idx = indices[:num_train]
    valid_idx = indices[num_train:]

    train_images = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    valid_images = [image_paths[i] for i in valid_idx]
    valid_masks = [mask_paths[i] for i in valid_idx]
    return train_images, train_masks, valid_images, valid_masks


def read_pair_manifest(manifest_path, root_path):
    """Read image/mask pairs from a frozen CSV manifest."""
    manifest_path = Path(manifest_path)
    root_path = Path(root_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    images, masks = [], []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "mask"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain image and mask columns: {manifest_path}")
        for row in reader:
            image = Path(row["image"])
            mask = Path(row["mask"])
            image = image if image.is_absolute() else root_path / image
            mask = mask if mask.is_absolute() else root_path / mask
            if not image.exists() or not mask.exists():
                raise FileNotFoundError(f"Missing manifest pair: image={image}, mask={mask}")
            images.append(image)
            masks.append(mask)

    if not images:
        raise ValueError(f"Manifest contains no pairs: {manifest_path}")
    return images, masks


def build_train_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.15,
                rotate_limit=20,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15, contrast_limit=0.15, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=6, sat_shift_limit=12, val_shift_limit=10, p=1.0
                    ),
                ],
                p=0.6,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
                ],
                p=0.25,
            ),
        ]
    )


def build_valid_transforms(img_size):
    return A.Compose([A.Resize(img_size, img_size)])


class WoundSegDataset(Dataset):
    def __init__(self, images, masks, transform=None, normalize=True):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(str(self.images[index]))
        if image is None:
            raise ValueError(f"Failed to read image: {self.images[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.masks[index]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {self.masks[index]}")

        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
            image = sample["image"]
            mask = sample["mask"]

        image = image.astype(np.float32) / 255.0 if self.normalize else image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()


def build_datasets(root_path, img_size=256, train_ratio=0.8, seed=42):
    train_images, train_masks, valid_images, valid_masks = split_dataset(
        root_path=root_path,
        train_ratio=train_ratio,
        seed=seed,
    )

    train_dataset = WoundSegDataset(
        train_images,
        train_masks,
        transform=build_train_transforms(img_size),
    )
    valid_dataset = WoundSegDataset(
        valid_images,
        valid_masks,
        transform=build_valid_transforms(img_size),
    )
    return train_dataset, valid_dataset
