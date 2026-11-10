import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from src.models.resnet import ResNet
from src.training.trainer import train_one_epoch, evaluate
from src.prepare_data import get_data


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    train_loader, val_loader, test_loader = get_data()
    resnet8 = ResNet(n=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        resnet8.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, foreach=False
    )
    epochs = 10

    print(f"Training on: {device}\n")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            resnet8, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate(resnet8, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.2%} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.2%}"
        )


if __name__ == "__main__":
    main()
