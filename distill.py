from pathlib import Path

import torch
from torch import nn, optim

from src.models.resnet import ResNet
from src.prepare_data import get_data
from src.training.trainer import evaluate
from src.training.distillation import distill_one_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    train_loader, val_loader, _ = get_data()

    teacher = ResNet(n=18).to(device)
    student = ResNet(n=1).to(device)

    # Set this to the location of your existing teacher checkpoint.
    checkpoint_path = Path("checkpoints/ResNet-110/best.pt")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    teacher.requires_grad_(False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        student.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        foreach=False,
    )

    for epoch in range(1, 6):
        train_loss, train_acc = distill_one_epoch(
            teacher,
            student,
            train_loader,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate(
            student,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch}/5 | "
            f"Train KD loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.2%} | "
            f"Val CE loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.2%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
