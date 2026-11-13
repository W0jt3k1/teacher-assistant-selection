import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from src.models.resnet import ResNet
from src.prepare_data import get_data
from src.training.trainer import train_one_epoch, evaluate


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # Configuration
    set_seed(42)

    epochs = 5
    patience = 30
    min_delta = 0.001
    early_stopping_enabled = False

    best_val_acc = 0.0
    early_stopping_best = 0.0
    epochs_without_improvement = 0

    model_name = "ResNet-8"
    checkpoints_path = Path("checkpoints")
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    checkpoints_path = Path("checkpoints") / model_name
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    # Data and model
    train_loader, val_loader, _ = get_data()

    model = ResNet(n=1).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        foreach=False,
    )

    scheduler = MultiStepLR(
        optimizer,
        milestones=[75, 112],
        gamma=0.1,
    )

    print(f"Training on: {device}\n")

    # Training
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                },
                checkpoints_path / "best.pt",
            )

        # Early stopping counter
        if val_acc > early_stopping_best + min_delta:
            early_stopping_best = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Learning rate scheduler
        scheduler.step()

        # Save training checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "early_stopping_best": early_stopping_best,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                checkpoints_path / "last.pt",
            )

        # Logging
        print(
            f"Epoch {epoch}/{epochs} | "
            f"LR: {current_lr:.5f} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.2%} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.2%}"
        )

        # Early stopping
        if early_stopping_enabled and epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch}")

            # Save final state before stopping
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "early_stopping_best": early_stopping_best,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                checkpoints_path / "last.pt",
            )
            break

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
