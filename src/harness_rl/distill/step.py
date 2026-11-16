"""One privileged self-distillation update of the student's QLoRA adapter."""

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import StrEnum

import torch

from harness_rl.distill.losses import chunked_jsd_sum, sampled_token_loss_sum
from harness_rl.distill.teacher_context import DistillExample
from harness_rl.training.logprobs import selected_token_logps
from harness_rl.training.qlora import PolicyTrainer, lora_norm


class DistillLoss(StrEnum):
    """Which objective to optimise."""

    FULL_JSD = "full_jsd"
    SAMPLED_TOKEN = "sampled_token"


class TeacherWeights(StrEnum):
    """Which frozen weights the privileged teacher uses for this update."""

    BASE = "base"  # adapter disabled: the shared initial checkpoint
    STUDENT_SNAPSHOT = "student_snapshot"  # current adapter, no grad


@dataclass(frozen=True, slots=True)
class DistillSettings:
    """Distillation hyper-parameters."""

    loss: DistillLoss = DistillLoss.FULL_JSD
    teacher: TeacherWeights = TeacherWeights.BASE
    beta: float = 0.5
    chunk_size: int = 512
    max_grad_norm: float = 1.0


@dataclass(frozen=True, slots=True)
class DistillMetrics:
    """Evidence that the teacher produced a usable gradient."""

    examples: int
    trained_tokens: int
    loss: float
    grad_norm: float
    lora_norm_before: float
    lora_norm_after: float


def completion_positions(total_length: int, mask: Sequence[int]) -> torch.Tensor:
    """Return absolute indices of trained completion tokens at the sequence end."""
    start = total_length - len(mask)
    indices = [start + i for i, keep in enumerate(mask) if keep and start + i > 0]
    return torch.tensor(indices, device="cuda", dtype=torch.long)


def final_hidden(trainer: PolicyTrainer, token_ids: Sequence[int]) -> torch.Tensor:
    """Run the transformer body and return ``[length, d]`` final hidden states."""
    ids = torch.tensor(token_ids, device="cuda")[None]
    return trainer.model.get_base_model().model(input_ids=ids).last_hidden_state[0]


def teacher_weights(
    trainer: PolicyTrainer, teacher: TeacherWeights
) -> AbstractContextManager:
    """Return a context that selects the teacher's (frozen) weights."""
    if teacher is TeacherWeights.BASE:
        return trainer.model.disable_adapter()
    return nullcontext()


def example_loss_sum(
    trainer: PolicyTrainer, example: DistillExample, settings: DistillSettings
) -> tuple[torch.Tensor, int]:
    """Return the summed loss and trained-token count for one example."""
    student_pos = completion_positions(
        len(example.student_ids), example.completion_mask
    )
    teacher_pos = completion_positions(
        len(example.teacher_ids), example.completion_mask
    )
    with torch.no_grad(), teacher_weights(trainer, settings.teacher):
        teacher_hidden = final_hidden(trainer, example.teacher_ids)[teacher_pos - 1]
    student_hidden = final_hidden(trainer, example.student_ids)[student_pos - 1]
    weight = trainer.model.get_base_model().get_output_embeddings().weight
    if settings.loss is DistillLoss.FULL_JSD:
        loss = chunked_jsd_sum(
            student_hidden, teacher_hidden, weight, settings.beta, settings.chunk_size
        )
        return loss, len(student_pos)
    targets = torch.tensor(example.student_ids, device="cuda")[student_pos]
    student = selected_token_logps(
        student_hidden, weight, targets, 1.0, settings.chunk_size
    )
    with torch.no_grad():
        teacher = selected_token_logps(
            teacher_hidden, weight, targets, 1.0, settings.chunk_size
        )
    return sampled_token_loss_sum(student, teacher), len(student_pos)


def distill_step(
    trainer: PolicyTrainer,
    examples: Sequence[DistillExample],
    settings: DistillSettings,
) -> DistillMetrics:
    """Accumulate the distillation loss over examples and step once."""
    total = sum(sum(e.completion_mask) for e in examples)  # normaliser, as in GRPO
    norm_before = lora_norm(trainer.model)
    trainer.model.train()
    loss_total = 0.0
    for example in examples:
        loss_sum, _ = example_loss_sum(trainer, example, settings)
        (loss_sum / max(total, 1)).backward()
        loss_total += float(loss_sum)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in trainer.model.parameters() if p.requires_grad],
        settings.max_grad_norm,
    )
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    return DistillMetrics(
        examples=len(examples),
        trained_tokens=total,
        loss=loss_total / max(total, 1),
        grad_norm=float(grad_norm),
        lora_norm_before=norm_before,
        lora_norm_after=lora_norm(trainer.model),
    )
