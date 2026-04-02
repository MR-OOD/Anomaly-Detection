import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import nibabel as nib


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_valid_transforms():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def _is_supported_file(path):
    lower = path.lower()
    return lower.endswith((".png", ".jpg", ".jpeg", ".nii", ".nii.gz"))


def _list_supported_files(folder):
    return sorted([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and _is_supported_file(f)
    ])


def _first_existing_dir(*paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"None of these directories exists: {paths}")


def _resolve_base_root(data_root):
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    return os.path.abspath(data_root)


def load_nifti_image(path, transform=None):
    lower = path.lower()

    if lower.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(path).convert("RGB")
        return transform(img) if transform else img

    nifti_img = nib.load(path)
    image = nifti_img.get_fdata()

    if image.shape == (224, 224, 1, 3):
        image = image[:, :, 0, :]
    else:
        raise ValueError(
            f"Unsupported image shape: {image.shape}. Expected (224, 224, 1, 3)"
        )

    image_normalized = np.zeros_like(image)
    for i in range(3):
        slice_i = image[:, :, i]
        slice_i = (slice_i - np.min(slice_i)) / (np.max(slice_i) - np.min(slice_i) + 1e-8) * 255
        image_normalized[:, :, i] = slice_i

    image = image_normalized.astype(np.uint8)
    image_pil = Image.fromarray(image)

    return transform(image_pil) if transform else image_pil


def load_nifti_mask(path, transform=None):
    lower = path.lower()

    if lower.endswith((".png", ".jpg", ".jpeg")):
        mask_pil = Image.open(path).convert("L")
        mask_tensor = transforms.ToTensor()(mask_pil)
        mask_tensor = torch.where(mask_tensor >= 0.1, 1.0, 0.0)

        if transform:
            mask_pil = transforms.ToPILImage()(mask_tensor)
            mask_tensor = transform(mask_pil)
            mask_tensor = torch.where(mask_tensor >= 0.1, 1.0, 0.0)

        return mask_tensor

    nifti_mask = nib.load(path)
    mask = nifti_mask.get_fdata()

    if mask.shape != (224, 224):
        raise ValueError(f"Unsupported mask shape: {mask.shape}. Expected (224, 224)")

    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-8) * 255
    mask = mask.astype(np.uint8)
    mask_pil = Image.fromarray(mask).convert("L")

    mask_tensor = transforms.ToTensor()(mask_pil)
    mask_tensor = torch.where(mask_tensor >= 0.1, 1.0, 0.0)

    if transform:
        mask_pil = transforms.ToPILImage()(mask_tensor)
        mask_tensor = transform(mask_pil)
        mask_tensor = torch.where(mask_tensor >= 0.1, 1.0, 0.0)

    return mask_tensor


def _find_mask_path(mask_dir, fname):
    direct = os.path.join(mask_dir, fname)
    if os.path.exists(direct):
        return direct

    stem = fname
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = os.path.splitext(stem)[0]

    candidates = [
        os.path.join(mask_dir, stem + ".png"),
        os.path.join(mask_dir, stem + ".jpg"),
        os.path.join(mask_dir, stem + ".jpeg"),
        os.path.join(mask_dir, stem + ".nii.gz"),
        os.path.join(mask_dir, stem + ".nii"),
    ]

    for c in candidates:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(f"Mask not found for abnormal sample: {fname}")


class TrainDataset(data.Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = _resolve_base_root(data_root)
        self.transform = transform

        self.train_root = _first_existing_dir(
            os.path.join(self.data_root, "train", "good", "img"),
            os.path.join(self.data_root, "train", "good"),
        )

        self.data_list = _list_supported_files(self.train_root)

    def __getitem__(self, idx):
        img_path = os.path.join(self.train_root, self.data_list[idx])
        img = load_nifti_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data_list)


class ValidDataset(data.Dataset):
    def __init__(self, data_root, transform=None, size=256):
        self.data_root = _resolve_base_root(data_root)
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

        self.good = _first_existing_dir(
            os.path.join(self.data_root, "valid", "good", "img"),
            os.path.join(self.data_root, "valid", "good"),
        )
        self.ungood = _first_existing_dir(
            os.path.join(self.data_root, "valid", "Ungood", "img"),
            os.path.join(self.data_root, "valid", "Ungood"),
        )
        self.mask = os.path.join(self.data_root, "valid", "Ungood", "label")

        self.good_list = _list_supported_files(self.good)
        self.ungood_list = _list_supported_files(self.ungood)

    def __getitem__(self, idx):
        if idx < len(self.ungood_list):
            fname = self.ungood_list[idx]
            img_path = os.path.join(self.ungood, fname)
            img = load_nifti_image(img_path, self.transform)

            if os.path.isdir(self.mask):
                mask_path = _find_mask_path(self.mask, fname)
                mask = load_nifti_mask(mask_path, self.mask_transform)
            else:
                mask = torch.zeros(1, 256, 256)

            label = torch.tensor(1, dtype=torch.long)
        else:
            fname = self.good_list[idx - len(self.ungood_list)]
            img_path = os.path.join(self.good, fname)
            img = load_nifti_image(img_path, self.transform)
            mask = torch.zeros(1, 256, 256)
            label = torch.tensor(0, dtype=torch.long)

        return {"image": img, "mask": mask, "label": label, "path": str(img_path)}

    def __len__(self):
        return len(self.good_list) + len(self.ungood_list)


class TestDataset(data.Dataset):
    def __init__(self, data_root, transform=None, size=256):
        self.data_root = _resolve_base_root(data_root)
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

        self.good = _first_existing_dir(
            os.path.join(self.data_root, "test", "good", "img"),
            os.path.join(self.data_root, "test", "good"),
        )
        self.ungood = _first_existing_dir(
            os.path.join(self.data_root, "test", "Ungood", "img"),
            os.path.join(self.data_root, "test", "Ungood"),
        )
        self.mask = os.path.join(self.data_root, "test", "Ungood", "label")

        self.good_list = _list_supported_files(self.good)
        self.ungood_list = _list_supported_files(self.ungood)

    def __getitem__(self, idx):
        if idx < len(self.ungood_list):
            fname = self.ungood_list[idx]
            img_path = os.path.join(self.ungood, fname)
            img = load_nifti_image(img_path, self.transform)

            if os.path.isdir(self.mask):
                mask_path = _find_mask_path(self.mask, fname)
                mask = load_nifti_mask(mask_path, self.mask_transform)
            else:
                mask = torch.zeros(1, 256, 256)

            label = torch.tensor(1, dtype=torch.long)
        else:
            fname = self.good_list[idx - len(self.ungood_list)]
            img_path = os.path.join(self.good, fname)
            img = load_nifti_image(img_path, self.transform)
            mask = torch.zeros(1, 256, 256)
            label = torch.tensor(0, dtype=torch.long)

        return {"image": img, "mask": mask, "label": label, "path": str(img_path)}

    def __len__(self):
        return len(self.good_list) + len(self.ungood_list)