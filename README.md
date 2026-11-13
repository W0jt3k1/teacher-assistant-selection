### Slow Download Fix

If downloading CIFAR-100 through `torchvision` is extremely slow, download it manually from the [Zenodo mirror](https://zenodo.org/records/10089977/files/cifar-100-python.tar.gz?download=1).

Extract the archive into the `data/` directory. `torchvision` will automatically detect the existing dataset and skip downloading it again.

# Teacher Assistant Selection

Research project investigating Knowledge Distillation (KD) and Teacher Assistant Knowledge Distillation (TAKD) on CIFAR-100.

The project supports supervised training and knowledge distillation, with experiment configurations, checkpoint management, training resumption, and metrics logging.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Compatible GPU (NVIDIA CUDA or Intel XPU), or CPU
- Git

## Installation

Clone the repository:

```bash
git clone git@github.com:W0jt3k1/teacher-assistant-selection.git
cd teacher-assistant-selection
```

Install Python if needed:

```bash
uv python install 3.12
```

Synchronize dependencies for your hardware.

**NVIDIA GPU (CUDA):**

```bash
uv sync --extra cuda
```

**Intel GPU (XPU):**

```bash
uv sync --extra xpu
```

The appropriate PyTorch backend is selected through the corresponding optional dependency group.

## Dataset

The project uses [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), containing 100 image classes.

The dataset is split into:

| Split | Samples |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Test | 10,000 |

Download CIFAR-100 using torchvision:

```bash
uv run --extra cuda python -c "from torchvision.datasets import CIFAR100; CIFAR100(root='./data', train=True, download=True); CIFAR100(root='./data', train=False, download=True)"
```

For Intel XPU, replace `--extra cuda` with `--extra xpu`.

The dataset should be stored in:

```text
data/
└── cifar-100-python/
    ├── train
    ├── test
    └── meta
```

If the official download is unavailable, the CIFAR-100 Python archive can be downloaded manually and extracted into `data/`.

Dataset files are excluded from Git.

## Experiment Configuration

Experiments are defined in `experiments.py` as a list of `TrainingConfig` objects.

Each configuration specifies the model, training method, hyperparameters, checkpoint settings, and optional distillation parameters.

Example:

```python
from src.training.config import TrainingConfig, DistillationConfig

experiments = [
    TrainingConfig(
        method="supervised",
        model_n=18,
        epochs=150,
        run_name="teacher-resnet110",
    ),
    TrainingConfig(
        method="supervised",
        model_n=1,
        epochs=150,
        run_name="student-resnet8-baseline",
    ),
    TrainingConfig(
        method="distillation",
        model_n=1,
        epochs=150,
        run_name="student-resnet8-kd",
        distillation=DistillationConfig(
            teacher_model_n=18,
            teacher_checkpoint="checkpoints/teacher-resnet110/best.pt",
            alpha=0.5,
            temperature=4.0,
        ),
    ),
]
```

Experiments run **sequentially, in the order they appear in the list**.

For knowledge distillation, the teacher checkpoint must exist before the corresponding experiment starts. The example above trains the teacher first, followed by the baseline student and the distilled student.

### Supported training methods

- `supervised` — standard training using ground-truth labels and cross-entropy loss.
- `distillation` — knowledge distillation using a pretrained teacher and a combination of cross-entropy and KL divergence.

The ResNet depth is determined by `6n + 2`:

| `model_n` | Model |
|---|---|
| `1` | ResNet-8 |
| `5` | ResNet-32 |
| `18` | ResNet-110 |

## Running Experiments

**NVIDIA CUDA:**

```bash
uv run --extra cuda python train.py
```

**Intel XPU:**

```bash
uv run --extra xpu python train.py
```

All experiments in `experiments.py` are executed automatically, one after another.

To run only a subset of experiments, remove or comment out the other entries in the list.

To save console output to a file in PowerShell:

```powershell
uv run --extra cuda python -u train.py 2>&1 | Tee-Object -FilePath training.log
```

## Checkpoints and Metrics

Each experiment creates a separate directory under `checkpoints/`, based on its `run_name`.

Example:

```text
checkpoints/
├── teacher-resnet110/
│   ├── best.pt
│   ├── last.pt
│   └── metrics.csv
├── student-resnet8-baseline/
│   ├── best.pt
│   ├── last.pt
│   └── metrics.csv
└── student-resnet8-kd/
    ├── best.pt
    ├── last.pt
    └── metrics.csv
```

- `best.pt` — model weights from the epoch with the highest validation accuracy.
- `last.pt` — training checkpoint containing model, optimizer and scheduler states, as well as the experiment configuration.
- `metrics.csv` — epoch-level training and validation loss, accuracy, and learning rate.

The checkpoint save interval is controlled by `checkpoint_interval` in `TrainingConfig`.

Checkpoints and datasets are excluded from Git.

## Resuming Training

To continue an interrupted experiment, set:

```python
resume_from_checkpoint=True
```

in the corresponding `TrainingConfig`.

Keep the same `run_name` and other training hyperparameters. The number of epochs may be increased.

Example:

```python
TrainingConfig(
    method="supervised",
    model_n=18,
    epochs=150,
    run_name="teacher-resnet110",
    resume_from_checkpoint=True,
)
```

The training script loads `last.pt` and resumes from the following epoch.

**Important:** starting a new experiment with an existing checkpoint directory raises an error to prevent accidental overwriting. Completed experiments are not automatically skipped; remove them from the experiment list when they no longer need to run.

## Running Tests

Execute the project tests with:

```bash
uv run --extra cuda pytest
```

For Intel XPU, use `--extra xpu`.

## Notes

- The best checkpoint is selected using validation accuracy.
- The test dataset is reserved for final evaluation.
- Direct distillation and supervised training share the same training infrastructure.
- Teacher Assistant Knowledge Distillation can be implemented as two sequential distillation experiments: Teacher → Assistant and Assistant → Student.
- Full deterministic reproduction across interrupted runs is not yet guaranteed.
