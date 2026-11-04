from torchvision.datasets import CIFAR100

train_dataset = CIFAR100(root="./data", train=True, download=True)
test_dataset = CIFAR100(root="./data", train=False, download=True)
