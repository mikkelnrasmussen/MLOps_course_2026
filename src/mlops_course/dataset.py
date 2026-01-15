from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import torchvision.transforms.v2 as transforms


class MnistDataset(Dataset):
    """MNIST dataset for PyTorch.

    Args:
        data_folder: Path to the data folder.
        train: Whether to load training or test data.
        img_transform: Image transformation to apply.
        target_transform: Target transformation to apply.
    """

    name: str = "MNIST"

    def __init__(
        self,
        data_folder: str = "data",
        train: bool = True,
        img_transform: transforms.Transform | None = None,
        target_transform: transforms.Transform | None = None,
    ) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.train = train
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.load_data()

    def load_data(self) -> None:
        """Load images and targets from disk."""
        if self.train:
            self.images = torch.load(f"{self.data_folder}/train_images.pt")
            self.target = torch.load(f"{self.data_folder}/train_target.pt")
        else:
            self.images = torch.load(f"{self.data_folder}/test_images.pt")
            self.target = torch.load(f"{self.data_folder}/test_target.pt")

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return image and target tensor."""
        img, target = self.images[idx], self.target[idx]
        if self.img_transform:
            img = self.img_transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return self.images.shape[0]
