"""Memory-bounded log-probabilities over a large vocabulary.

Qwen3's vocabulary is ~151k tokens: full fp32 logits for a 12k-token agent
trace would be ~7 GB. Logits are therefore computed chunk by chunk, only at
trained positions, and recomputed in backward instead of being stored.
"""

import torch
from torch.utils.checkpoint import checkpoint


def _chunk_logps(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return log p(target) for one chunk of positions."""
    logits = (hidden @ weight.T).float() / temperature
    return logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def selected_token_logps(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    """Compute log-probs of ``targets`` from final hidden states, chunked.

    Args:
        hidden: ``[n, d]`` hidden states at the positions that predict targets.
        lm_head_weight: ``[vocab, d]`` output projection.
        targets: ``[n]`` token ids to score.
        temperature: Sampling temperature used by vLLM for these tokens.
        chunk_size: Positions per chunk; bounds peak logits memory.

    Returns:
        ``[n]`` float32 log-probabilities.
    """
    chunks = [
        checkpoint(
            _chunk_logps,
            hidden[start : start + chunk_size],
            lm_head_weight,
            targets[start : start + chunk_size],
            temperature,
            use_reentrant=False,
        )
        for start in range(0, hidden.shape[0], chunk_size)
    ]
    return torch.cat(chunks)


def trained_positions(loss_mask: torch.Tensor) -> torch.Tensor:
    """Return indices of trained tokens (position 0 can never be predicted)."""
    positions = loss_mask.nonzero().squeeze(-1)
    return positions[positions > 0]
