from torchvision.datasets import CIFAR100
from torch.utils.data import random_split
from torch import Generator


def get_data():
    test_set = CIFAR100(root="./data", train=False, download=False)
    full_train_set = CIFAR100(root="./data", train=True, download=False)
    train_set, val_set = random_split(
        full_train_set, [45_000, 5_000], generator=Generator().manual_seed(42)
    )
    return train_set, val_set, test_set
