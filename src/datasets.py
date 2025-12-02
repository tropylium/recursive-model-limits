import os
from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_ROOT = Path(os.environ["DATASET_DIR"])


def CIFAR10(train: bool = True, transform: Optional[Callable] = None, **kwargs):
    """
    Simple wrapper around CIFAR10
    """
    dataset = datasets.CIFAR10(
        root=DATASET_ROOT / "CIFAR10",
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        ) if transform is None else transform,
        **kwargs
    )
    return dataset


def CIFAR100(train: bool = True, transform: Optional[Callable] = None, **kwargs):
    """
    Simple wrapper around CIFAR100
    """
    dataset = datasets.CIFAR100(
        root=DATASET_ROOT / "CIFAR100",
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
                ),
            ]
        ) if transform is None else transform,
        **kwargs
    )
    return dataset


def dataset_factory(identifier: str):
    if identifier == "CIFAR10":
        return CIFAR10
    elif identifier == "CIFAR100":
        return CIFAR100
    else:
        raise ValueError(f"Invalid dataset identifier: {identifier}")
