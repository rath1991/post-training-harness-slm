"""Verifier rewards and group-relative advantages (same rule as TRL async GRPO)."""

import statistics
from collections.abc import Sequence

# Population std as in TRL's AsyncRolloutLoop.compute_advantages.
STD_EPSILON = 1e-8


def correctness_reward(env_reward: float | None) -> float | None:
    """Use the Harbor verifier's all-or-nothing score as the reward.

    ``None`` means the verifier never ran; it is unscorable, not a failure.
    """
    return None if env_reward is None else float(env_reward)


def group_advantages(rewards: Sequence[float | None]) -> tuple[float | None, ...]:
    """Standardise rewards within a group; unscored rollouts get ``None``."""
    scored = [reward for reward in rewards if reward is not None]
    if len(scored) < 2:
        return tuple(None if r is None else 0.0 for r in rewards)
    mean = statistics.fmean(scored)
    std = statistics.pstdev(scored)
    return tuple(
        None if reward is None else (reward - mean) / (std + STD_EPSILON)
        for reward in rewards
    )


def has_learning_signal(rewards: Sequence[float | None]) -> bool:
    """Return True if the group's scored rewards are not all identical.

    A group where every attempt passed (or every attempt failed) has zero
    advantage everywhere and contributes no gradient.
    """
    return len({reward for reward in rewards if reward is not None}) > 1
