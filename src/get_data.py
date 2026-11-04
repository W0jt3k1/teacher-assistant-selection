from torchvision.datasets import CIFAR100

print("pobieranie z pytorcha moze zostac ograniczone i trwać długo ")
# pytorch sam pilnuje zeby nie pobrac datasetu 2 raz jesli istnieje pod root
train_dataset = CIFAR100(root="./data", train=True, download=True)
test_dataset = CIFAR100(root="./data", train=False, download=True)
