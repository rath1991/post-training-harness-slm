"""Distillation objectives: full-distribution JSD and a sampled-token alternative.

These are deliberately separate functions. A sampled-token log-ratio is a
one-sample estimate whose *gradient* matches reverse KL in expectation; it is
not the same quantity as a full-vocabulary KL and is reported separately.
"""

import math

import torch
from torch.utils.checkpoint import checkpoint


def kl_divergence(p_logps: torch.Tensor, q_logps: torch.Tensor) -> torch.Tensor:
    """Return KL(p || q) per position from log-probabilities over the vocab."""
    return (p_logps.exp() * (p_logps - q_logps)).sum(-1)


def generalized_jsd(
    student_logps: torch.Tensor, teacher_logps: torch.Tensor, beta: float
) -> torch.Tensor:
    """Per-position generalized JSD with TRL's beta convention.

    ``beta=0`` is forward KL(teacher || student), ``beta=1`` is reverse
    KL(student || teacher), and values in between use the beta-mixture.
    """
    if beta == 0.0:
        return kl_divergence(teacher_logps, student_logps)
    if beta == 1.0:
        return kl_divergence(student_logps, teacher_logps)
    mixture = torch.logsumexp(
        torch.stack(
            [student_logps + math.log(1 - beta), teacher_logps + math.log(beta)]
        ),
        dim=0,
    )
    return beta * kl_divergence(teacher_logps, mixture) + (1 - beta) * kl_divergence(
        student_logps, mixture
    )


def _chunk_jsd(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    weight: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Return the summed JSD for one chunk of aligned positions."""
    student = (student_hidden @ weight.T).float().log_softmax(-1)
    teacher = (teacher_hidden @ weight.T).float().log_softmax(-1)
    return generalized_jsd(student, teacher, beta).sum()


def chunked_jsd_sum(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    beta: float,
    chunk_size: int,
) -> torch.Tensor:
    """Sum generalized JSD over positions without materialising full logits.

    Args:
        student_hidden: ``[n, d]`` student states at completion positions.
        teacher_hidden: ``[n, d]`` teacher states at the same positions
            (computed without grad: the teacher is frozen for this update).
        lm_head_weight: Shared ``[vocab, d]`` output projection.
        beta: JSD interpolation (0 forward KL, 1 reverse KL).
        chunk_size: Positions per chunk.

    Returns:
        Scalar sum; divide by the total number of trained tokens.
    """
    teacher_hidden = teacher_hidden.detach()
    return sum(
        checkpoint(
            _chunk_jsd,
            student_hidden[start : start + chunk_size],
            teacher_hidden[start : start + chunk_size],
            lm_head_weight,
            beta,
            use_reentrant=False,
        )
        for start in range(0, student_hidden.shape[0], chunk_size)
    )


def sampled_token_loss_sum(
    student_logps: torch.Tensor, teacher_logps: torch.Tensor
) -> torch.Tensor:
    """Policy-gradient form of on-policy distillation over sampled tokens.

    Each sampled token's reward is ``log p_teacher - log p_student`` (detached),
    so the gradient is a single-sample estimate of reverse KL's gradient.
    """
    advantage = (teacher_logps - student_logps).detach()
    return -(advantage * student_logps).sum()
