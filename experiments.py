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
