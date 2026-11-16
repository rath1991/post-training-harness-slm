"""One GRPO optimizer step over captured rows, with evidence that weights moved."""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from harness_rl.config import GrpoConfig
from harness_rl.rollout.traces import TokenSequence, TrainingRow
from harness_rl.training.grpo_loss import ClipConfig, LossTerms, grpo_loss_sum
from harness_rl.training.logprobs import selected_token_logps, trained_positions
from harness_rl.training.qlora import PolicyTrainer, lora_norm


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """What happened during one optimizer step."""

    rows: int
    rows_skipped_too_long: int
    trained_tokens: int
    loss: float
    mean_ratio: float
    clip_fraction: float
    is_masked_fraction: float
    mean_abs_logp_diff: float
    grad_norm: float
    lora_norm_before: float
    lora_norm_after: float
    seconds: float
    torch_peak_alloc_mib: float


def clip_config(grpo: GrpoConfig) -> ClipConfig:
    """Extract the loss's clipping settings from the run config."""
    return ClipConfig(
        grpo.epsilon_low, grpo.epsilon_high, grpo.is_clip_min, grpo.is_clip_max
    )


def sequence_logps(
    trainer: PolicyTrainer, sequence: TokenSequence, grpo: GrpoConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (current policy log-probs, sampler log-probs) at trained tokens."""
    device = torch.device("cuda")
    input_ids = torch.tensor(sequence.input_ids, device=device)
    positions = trained_positions(torch.tensor(sequence.loss_mask, device=device))
    causal_lm = trainer.model.get_base_model()
    hidden = causal_lm.model(input_ids=input_ids[None]).last_hidden_state[0]
    new_logps = selected_token_logps(
        hidden[positions - 1],
        causal_lm.get_output_embeddings().weight,
        input_ids[positions],
        grpo.temperature,
        grpo.logprob_chunk,
    )
    old_logps = torch.tensor(sequence.old_logps, device=device)[positions]
    return new_logps, old_logps


def per_token_means(terms: Sequence[LossTerms], total_tokens: int) -> dict[str, float]:
    """Average per-sequence loss statistics over the step's trained tokens."""
    n = max(total_tokens, 1)
    return {
        "loss": sum(float(t.loss_sum) for t in terms) / n,
        "mean_ratio": sum(t.ratio_sum for t in terms) / n,
        "clip_fraction": sum(t.clipped for t in terms) / n,
        "is_masked_fraction": sum(t.is_masked for t in terms) / n,
        "mean_abs_logp_diff": sum(t.abs_logp_diff_sum for t in terms) / n,
    }


def train_step(
    trainer: PolicyTrainer, rows: Sequence[TrainingRow], grpo: GrpoConfig
) -> StepMetrics:
    """Accumulate gradients over every row, then take one optimizer step.

    Each row is its own micro-batch (agent traces vary hugely in length), and
    the loss is normalised by the step's total trained tokens, as in TRL.
    """
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    usable = [r for r in rows if len(r.sequence.input_ids) <= grpo.max_train_tokens]
    total_tokens = sum(row.sequence.n_trained for row in usable)
    norm_before = lora_norm(trainer.model)
    trainer.model.train()
    terms = []
    for row in usable:
        new_logps, old_logps = sequence_logps(trainer, row.sequence, grpo)
        term = grpo_loss_sum(new_logps, old_logps, row.advantage, clip_config(grpo))
        (term.loss_sum / max(total_tokens, 1)).backward()
        terms.append(term)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in trainer.model.parameters() if p.requires_grad], grpo.max_grad_norm
    )
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    return StepMetrics(
        rows=len(usable),
        rows_skipped_too_long=len(rows) - len(usable),
        trained_tokens=total_tokens,
        **per_token_means(terms, total_tokens),
        grad_norm=float(grad_norm),
        lora_norm_before=norm_before,
        lora_norm_after=lora_norm(trainer.model),
        seconds=time.monotonic() - started,
        torch_peak_alloc_mib=torch.cuda.max_memory_allocated() / 2**20,
    )


@torch.no_grad()
def probe_logps(
    trainer: PolicyTrainer, sequence: TokenSequence, grpo: GrpoConfig
) -> tuple[float, ...]:
    """Score a fixed probe sequence with the trainer's current policy.

    Compared later against vLLM's log-probs for the same tokens under the newly
    loaded adapter, this proves the inference engine is serving the update.
    """
    trainer.model.eval()
    new_logps, _ = sequence_logps(trainer, sequence, grpo)
    return tuple(new_logps.float().cpu().tolist())
