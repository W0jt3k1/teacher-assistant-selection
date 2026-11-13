import torch.nn.functional as F
import torch
from tqdm import tqdm


def kd_loss(student_logits, teacher_logits, targets, alpha=0.5, temperature=4.0):

    ce_loss = F.cross_entropy(student_logits, targets)

    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)

    return (1 - alpha) * ce_loss + alpha * kd_loss


def distill_one_epoch(
    teacher,
    student,
    train_loader,
    optimizer,
    device,
    alpha=0.5,
    temperature=4.0,
):
    teacher.eval()
    student.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(train_loader, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        # Teacher does not receive gradients.
        with torch.no_grad():
            teacher_logits = teacher(images)

        student_logits = student(images)

        loss = kd_loss(
            student_logits,
            teacher_logits,
            targets,
            alpha,
            temperature,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += (student_logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return total_loss / total, correct / total
