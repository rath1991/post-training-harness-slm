"""GRPO clipped surrogate over sampled tokens (beta=0, no reference model).

Matches TRL AsyncGRPOTrainer's ``score_centering=False`` branch: the vLLM
sampler's log-probs are the old policy, tokens whose importance ratio falls
outside ``[is_clip_min, is_clip_max]`` are dropped, and the loss is a sum over
tokens that the caller normalises by the step's total trained-token count.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ClipConfig:
    """PPO-style ratio clipping plus the sampler-mismatch mask."""

    epsilon_low: float
    epsilon_high: float
    is_clip_min: float
    is_clip_max: float


@dataclass(frozen=True, slots=True)
class LossTerms:
    """Summed loss and the per-token statistics that explain it."""

    loss_sum: torch.Tensor
    n_tokens: int
    ratio_sum: float
    clipped: int
    is_masked: int
    abs_logp_diff_sum: float


def grpo_loss_sum(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantage: float,
    clip: ClipConfig,
) -> LossTerms:
    """Return the summed clipped-surrogate loss for one sequence's trained tokens.

    On the first optimizer step the trainer's policy equals the sampler's, so
    ``abs_logp_diff`` then measures pure inference/training mismatch
    (BF16 vLLM vs. NF4 QLoRA), the scope's main numerical risk.
    """
    log_ratio = new_logps - old_logps
    ratio = log_ratio.exp()
    is_masked = (ratio < clip.is_clip_min) | (ratio > clip.is_clip_max)
    clipped_ratio = ratio.clamp(1 - clip.epsilon_low, 1 + clip.epsilon_high)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    loss_sum = (-surrogate * (~is_masked)).sum()  # keeps the autograd graph
    with torch.no_grad():
        return LossTerms(
            loss_sum=loss_sum,
            n_tokens=int(new_logps.numel()),
            ratio_sum=float(ratio.sum()),
            clipped=int((clipped_ratio != ratio).sum()),
            is_masked=int(is_masked.sum()),
            abs_logp_diff_sum=float(log_ratio.abs().sum()),
        )
