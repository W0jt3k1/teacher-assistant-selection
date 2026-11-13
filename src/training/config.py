from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration for training a model.
    """

    method: str = "supervised"
    architecture: str = "ResNet"
    model_n: int = 1
    run_name: str | None = None
    milestones: list = (75, 112)
    gamma: float = 0.1
    checkpoint_interval: int = 10
    epochs: int = 150
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    seed: int = 42
    patience: int = 30
    min_delta: float = 0.001
    early_stopping_enabled: bool = False
    resume_from_checkpoint: bool = False

    def get_run_name(self) -> str:
        """
        Returns the run name based on the run_name or generated name.
        """
        if self.run_name is not None:
            return self.run_name
        return f"{self.architecture}-{6*self.model_n+2}-{self.method}-seed{self.seed}"
