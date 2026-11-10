import pytest
import torch

from torchvision.datasets import CIFAR100
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
)
from src.prepare_data import get_data


@pytest.fixture(scope="module")
def loaders():
    return get_data()


@pytest.fixture(scope="module")
def datasets(loaders):
    return tuple(loader.dataset for loader in loaders)


def test_full_train_set_size():
    train_data = CIFAR100(root="./data", train=True, download=False)
    assert len(train_data) == 50_000


def test_test_set_size():
    test_data = CIFAR100(root="./data", train=False, download=False)
    assert len(test_data) == 10_000


def test_image_size():
    train_data = CIFAR100(root="./data", train=True, download=False)
    image, _ = train_data[0]
    assert image.size == (32, 32)


def test_label_range():
    train_data = CIFAR100(root="./data", train=True, download=False)
    _, label = train_data[0]

    assert isinstance(label, int)
    assert 0 <= label < 100


def test_split_sizes(datasets):
    train, val, test = datasets

    assert len(train) == 45_000
    assert len(val) == 5_000
    assert len(test) == 10_000


def test_split_disjoint(datasets):
    train, val, _ = datasets

    train_indices = set(train.indices)
    val_indices = set(val.indices)

    assert train_indices.isdisjoint(val_indices)
    assert len(train_indices | val_indices) == 50_000


def test_batch_shapes(loaders):
    train_loader, _, _ = loaders

    images, labels = next(iter(train_loader))

    assert images.shape == (128, 3, 32, 32)
    assert labels.shape == (128,)


def test_batch_dtypes(loaders):
    train_loader, _, _ = loaders

    images, labels = next(iter(train_loader))

    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_batch_label_range(loaders):
    train_loader, _, _ = loaders

    _, labels = next(iter(train_loader))

    assert torch.all((labels >= 0) & (labels < 100))


def test_loader_shuffle(loaders):
    train_loader, val_loader, test_loader = loaders

    from torch.utils.data import RandomSampler, SequentialSampler

    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)


def test_transforms(datasets):
    train, val, test = datasets

    train_transform = train.dataset.transform
    val_transform = val.dataset.transform
    test_transform = test.transform

    assert isinstance(train_transform, Compose)
    assert isinstance(val_transform, Compose)
    assert isinstance(test_transform, Compose)

    assert any(isinstance(t, RandomCrop) for t in train_transform.transforms)
    assert any(isinstance(t, RandomHorizontalFlip) for t in train_transform.transforms)

    for transform in (val_transform, test_transform):
        assert len(transform.transforms) == 2
        assert isinstance(transform.transforms[0], ToTensor)
        assert isinstance(transform.transforms[1], Normalize)


def test_split_reproducibility(datasets):
    train, val, _ = datasets

    train_loader2, val_loader2, _ = get_data()

    assert train.indices == train_loader2.dataset.indices
    assert val.indices == val_loader2.dataset.indices
