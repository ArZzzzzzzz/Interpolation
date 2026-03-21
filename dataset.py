"""
dataset.py — Vimeo-90K Triplet Dataset for Video Frame Interpolation

Structure on disk:
    vimeo_triplet/
        sequences/          # triplets: I0, Imid, I1
            00001/0001/im1.png  (I0)
            00001/0001/im2.png  (Imid)
            00001/0001/im3.png  (I1)
        tri_trainlist.txt
        tri_testlist.txt

Each line in the txt files: "00001/0001"
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class PairedRandomCrop:
    """Apply the same random crop to all images in a tuple."""
    def __init__(self, size: int):
        self.size = size

    def __call__(self, imgs):
        i, j, h, w = transforms.RandomCrop.get_params(imgs[0], (self.size, self.size))
        return tuple(TF.crop(img, i, j, h, w) for img in imgs)


class PairedRandomHFlip:
    """Apply the same random horizontal flip to all images in a tuple."""
    def __call__(self, imgs):
        if random.random() > 0.5:
            return tuple(TF.hflip(img) for img in imgs)
        return imgs


class PairedRandomVFlip:
    """Apply the same random vertical flip to all images in a tuple."""
    def __call__(self, imgs):
        if random.random() > 0.5:
            return tuple(TF.vflip(img) for img in imgs)
        return imgs


class ToTensorNorm:
    """
    Convert PIL images to tensors and normalise to [-1, 1].
    Vimeo-90K frames are 448×256. We first resize to at least crop_size,
    then the crop is applied separately if used in a Paired transform.
    """
    def __call__(self, img: Image.Image) -> torch.Tensor:
        t = TF.to_tensor(img)       # [0, 1]
        t = t * 2.0 - 1.0           # [-1, 1]
        return t


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Vimeo90KTriplet(Dataset):
    """
    Vimeo-90K Triplet dataset.

    Args:
        root        : path to vimeo_triplet/ folder
        split       : 'train', 'val', or 'test'
        crop_size   : spatial size for random crop (default 256)
        augment     : whether to apply random flips (train only)
        val_fraction: fraction of training list used for validation
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: int = 256,
        augment: bool = True,
        val_fraction: float = 0.05,
        max_samples: int = None,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and (split == "train")

        # Read sequence list
        if split in ("train", "val"):
            list_file = self.root / "tri_trainlist.txt"
        else:
            list_file = self.root / "tri_testlist.txt"

        with open(list_file) as f:
            all_sequences = [line.strip() for line in f if line.strip()]

        # Train/val split from training list
        if split in ("train", "val"):
            n_val = max(1, int(len(all_sequences) * val_fraction))
            # Deterministic split: last n_val sequences go to validation
            if split == "train":
                self.sequences = all_sequences[:-n_val]
            else:
                self.sequences = all_sequences[-n_val:]
        else:
            self.sequences = all_sequences

        # Limiter le nombre de séquences si demandé
        if max_samples is not None:
            self.sequences = self.sequences[:max_samples]

        # Shared transforms
        self.to_tensor = ToTensorNorm()
        self.paired_crop = PairedRandomCrop(crop_size)
        self.paired_hflip = PairedRandomHFlip()
        self.paired_vflip = PairedRandomVFlip()

        # Resize: Vimeo frames are 448×256, we resize shorter side to crop_size
        self.resize = transforms.Resize(
            crop_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def _load_triplet(self, seq: str):
        base = self.root / "sequences" / seq
        I0   = load_image(str(base / "im1.png"))
        Imid = load_image(str(base / "im2.png"))
        I1   = load_image(str(base / "im3.png"))
        return I0, Imid, I1

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        I0, Imid, I1 = self._load_triplet(seq)

        # Resize all frames consistently
        I0   = self.resize(I0)
        Imid = self.resize(Imid)
        I1   = self.resize(I1)

        # Augmentation (train only): paired crop + flips
        if self.augment:
            I0, Imid, I1 = self.paired_crop((I0, Imid, I1))
            I0, Imid, I1 = self.paired_hflip((I0, Imid, I1))
            I0, Imid, I1 = self.paired_vflip((I0, Imid, I1))
        else:
            # Centre crop for val/test
            I0   = TF.center_crop(I0,   self.crop_size)
            Imid = TF.center_crop(Imid, self.crop_size)
            I1   = TF.center_crop(I1,   self.crop_size)

        # To tensor, normalise [-1, 1]
        I0   = self.to_tensor(I0)
        Imid = self.to_tensor(Imid)
        I1   = self.to_tensor(I1)

        return {"I0": I0, "Imid": Imid, "I1": I1, "seq": seq}


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    root: str,
    crop_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    val_fraction: float = 0.05,
    pin_memory: bool = True,
    max_samples: int = None,
):
    """
    Returns train, val, and test DataLoaders for Vimeo-90K Triplet.

    Args:
        root        : path to vimeo_triplet/
        crop_size   : spatial crop size
        batch_size  : samples per batch
        num_workers : DataLoader workers
        val_fraction: fraction of train list used for validation
        pin_memory  : pin memory for faster GPU transfer
        max_samples : limite le nombre de séquences (None = tout)

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = Vimeo90KTriplet(root, "train",  crop_size, augment=True,  val_fraction=val_fraction, max_samples=max_samples)
    val_ds   = Vimeo90KTriplet(root, "val",    crop_size, augment=False, val_fraction=val_fraction)
    test_ds  = Vimeo90KTriplet(root, "test",   crop_size, augment=False, val_fraction=val_fraction)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Dataset splits — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "./vimeo_triplet"
    train_loader, val_loader, test_loader = get_dataloaders(root, batch_size=4, num_workers=0)

    batch = next(iter(train_loader))
    print("I0   shape:", batch["I0"].shape)    # [4, 3, 256, 256]
    print("Imid shape:", batch["Imid"].shape)
    print("I1   shape:", batch["I1"].shape)
    print("Value range:", batch["I0"].min().item(), batch["I0"].max().item())
