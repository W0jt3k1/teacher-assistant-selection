import csv
from pathlib import Path

METRIC_FIELDS = [
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
]


def log_metrics(
    path: Path,
    epoch: int,
    learning_rate: float,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        # Prevent duplicated rows after resuming from a checkpoint.
        if rows and int(rows[-1]["epoch"]) >= epoch:
            rows = [row for row in rows if int(row["epoch"]) < epoch]

            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
