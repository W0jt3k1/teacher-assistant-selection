import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path
from src.models.resnet import ResNet
from src.training.trainer import train_one_epoch, evaluate
from src.prepare_data import get_data


def main():
    best_val_acc = 0.0
    checkpoints_path = Path("checkpoints")
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    train_loader, val_loader, test_loader = get_data()
    resnet110 = ResNet(n=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        resnet110.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, foreach=False
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[5, 10],
        gamma=0.1,
    )
    epochs = 11

    print(f"Training on: {device}\n")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            resnet110, train_loader, optimizer, criterion, device
        )
        scheduler.step()
        val_loss, val_acc = evaluate(resnet110, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": resnet110.state_dict(),
                    "val_acc": val_acc,
                },
                checkpoints_path / "resnet110_best.pt",
            )
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": resnet110.state_dict(),
                    "val_acc": val_acc,
                    "scheduler_state_dict": scheduler.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                checkpoints_path / "resnet110_last.pt",
            )
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"LR:{current_lr}"
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.2%} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.2%}"
        )


if __name__ == "__main__":
    main()
