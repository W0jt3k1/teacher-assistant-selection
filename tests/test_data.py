from torchvision.datasets import CIFAR100

train_data = CIFAR100(root="./data", train=True, download=False)
test_data = CIFAR100(root="./data", train=False, download=False)


def test_train_set_size():
    assert (
        len(train_data) == 50_000
    ), f"Expected 50,000 training samples, got {len(train_data)}"


def test_test_set_size():
    assert (
        len(test_data) == 10_000
    ), f"Expected 10,000 testing samples, got {len(test_data)}"


def test_image_size():
    image, _ = train_data[0]
    assert image.size == (32, 32), f"Expected image size 32x32 got {image.size}"


def test_label_range():
    _, label = train_data[0]
    assert isinstance(label, int), f"label should be whole number got {label}"
    assert 0 <= label <= 99, f"label should be in [0,99] got {label}"
