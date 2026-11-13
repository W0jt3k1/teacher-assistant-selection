import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.optim.lr_scheduler import MultiStepLR
from dataclasses import asdict
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.config import TrainingConfig
from src.models.resnet import ResNet
from src.prepare_data import get_data
from src.training.trainer import train_one_epoch, evaluate


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    print(f"Training on: {device}\n")
    config = TrainingConfig(
        epochs=4,
        checkpoint_interval=1,
        run_name="resume-test",
        resume_from_checkpoint=True,
    )
    # Configuration
    set_seed(config.seed)

    epochs = config.epochs
    min_delta = config.min_delta

    best_val_acc = 0.0
    early_stopping_best = 0.0
    epochs_without_improvement = 0

    model_name = config.get_run_name()
    model = ResNet(n=config.model_n).to(device)
    checkpoints_path = Path("checkpoints") / model_name
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        foreach=False,
    )

    scheduler = MultiStepLR(
        optimizer,
        milestones=list(config.milestones),
        gamma=config.gamma,
    )

    # Resume from checkpoint if specified
    start_epoch = 1

    if not config.resume_from_checkpoint:
        if (checkpoints_path / "best.pt").exists() or (
            checkpoints_path / "last.pt"
        ).exists():
            raise FileExistsError(f"Checkpoints already exist in {checkpoints_path}")
    if config.resume_from_checkpoint:
        last_checkpoint_path = checkpoints_path / "last.pt"

        checkpoint = load_checkpoint(last_checkpoint_path, device)

        saved_config = checkpoint["config"]

        if (
            saved_config["architecture"] != config.architecture
            or saved_config["model_n"] != config.model_n
            or saved_config["method"] != config.method
        ):
            raise ValueError("Checkpoint configuration mismatch")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        best_val_acc = checkpoint["best_val_acc"]
        early_stopping_best = checkpoint["early_stopping_best"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]

        start_epoch = checkpoint["epoch"] + 1

        print(f"Resumed training from epoch {start_epoch}")

    # Data and model
    train_loader, val_loader, _ = get_data()

    # Training
    for epoch in range(start_epoch, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
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
        if (
            epoch % config.checkpoint_interval == 0
            or epoch == epochs
            or (
                epochs_without_improvement >= config.patience
                and config.early_stopping_enabled
            )
        ):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "early_stopping_best": early_stopping_best,
                    "epochs_without_improvement": epochs_without_improvement,
                    "config": asdict(config),
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
        if (
            config.early_stopping_enabled
            and epochs_without_improvement >= config.patience
        ):
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
