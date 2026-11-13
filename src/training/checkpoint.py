from pathlib import Path
import torch


def save_checkpoint(state_dict: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return torch.load(path, map_location=device, weights_only=True)
