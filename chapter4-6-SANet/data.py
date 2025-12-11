# chapter4-6-CycleGAN/data.py

import os
import random
from glob import glob
from io import BytesIO
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class UnpairedImageDataset(Dataset):
    """
    Minimal unpaired dataset for CycleGAN.
    Expects:
      dir_A/xxx.png, dir_B/yyy.png ...
    Returns a dict with real_A and real_B tensors.
    """

    def __init__(
        self,
        dir_A: str,
        dir_B: str,
        transform: Optional[transforms.Compose] = None,
        serial_batches: bool = False,
        load_to: str = "disk",
    ):
        super().__init__()
        self.files_A = sorted(glob(os.path.join(dir_A, "*.*")))
        self.files_B = sorted(glob(os.path.join(dir_B, "*.*")))
        if not self.files_A:
            raise RuntimeError(f"No images found in {dir_A}")
        if not self.files_B:
            raise RuntimeError(f"No images found in {dir_B}")

        if load_to not in {"disk", "memory"}:
            raise ValueError("load_to must be either 'disk' or 'memory'")

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor()
                ]
            )
        else:
            self.transform = transform

        self.serial_batches = serial_batches
        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)
        self.load_to = load_to

        # Optionally cache image bytes in memory to reduce disk I/O
        self.data_A = None
        self.data_B = None
        if self.load_to == "memory":
            self.data_A = [self._read_bytes(path) for path in self.files_A]
            self.data_B = [self._read_bytes(path) for path in self.files_B]

    @staticmethod
    def _read_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def __len__(self):
        # Use the larger set length to avoid overfitting to the smaller domain
        return max(self.len_A, self.len_B)

    def __getitem__(self, idx):
        idx_A = idx % self.len_A
        idx_B = idx if self.serial_batches else random.randint(0, self.len_B - 1)
        if self.load_to == "memory":
            img_A = Image.open(BytesIO(self.data_A[idx_A])).convert("RGB")
            img_B = Image.open(BytesIO(self.data_B[idx_B % self.len_B])).convert("RGB")
        else:
            img_A = Image.open(self.files_A[idx_A]).convert("RGB")
            img_B = Image.open(self.files_B[idx_B % self.len_B]).convert("RGB")
        return {
            "content": self.transform(img_A),
            "style": self.transform(img_B),
        }


def create_dataloader(
    dir_A: str,
    dir_B: str,
    img_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    serial_batches: bool = False,
    load_to: str = "disk",
):
    dataset = UnpairedImageDataset(
        dir_A=dir_A,
        dir_B=dir_B,
        transform=transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ]
        ),
        serial_batches=serial_batches,
        load_to=load_to,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

