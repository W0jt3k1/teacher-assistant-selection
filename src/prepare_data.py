from torchvision.datasets import CIFAR100
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
)
from torch.utils.data import random_split, Subset, DataLoader
import torch

mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)


def get_data():
    test_set = CIFAR100(
        root="./data", train=False, download=False, transform=testval_transform()
    )
    full_train_set = CIFAR100(
        root="./data", train=True, download=False, transform=train_transform()
    )
    full_val_set = CIFAR100(
        root="./data", train=True, download=False, transform=testval_transform()
    )
    train_set, val_set = random_split(
        full_train_set, [45_000, 5_000], generator=torch.Generator().manual_seed(42)
    )
    val_set = Subset(full_val_set, val_set.indices)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader


def train_transform():
    transform = [
        RandomCrop(size=32, padding=4),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ]
    return Compose(transform)


def testval_transform():
    transform = [
        ToTensor(),
        Normalize(mean=mean, std=std),
    ]
    return Compose(transform)
