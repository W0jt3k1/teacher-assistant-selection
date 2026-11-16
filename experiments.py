from src.training.config import TrainingConfig, DistillationConfig

experiments = [
    TrainingConfig(
        method="distillation",
        model_n=3,
        epochs=150,
        run_name="assistant-resnet20-kd",
        milestones=(80, 120),
        nesterov=True,
        distillation=DistillationConfig(
            teacher_model_n=18,
            teacher_checkpoint="checkpoints/teacher-resnet110/best.pt",
            alpha=0.5,
            temperature=4.0,
        ),
    ),
    TrainingConfig(
        method="distillation",
        model_n=1,
        epochs=150,
        run_name="student-resnet8-takd-ta20",
        milestones=(80, 120),
        nesterov=True,
        distillation=DistillationConfig(
            teacher_model_n=3,
            teacher_checkpoint="checkpoints/assistant-resnet20-kd/best.pt",
            alpha=0.5,
            temperature=4.0,
        ),
    ),
]
