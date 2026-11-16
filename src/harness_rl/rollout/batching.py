"""Pure helpers that turn rollout groups into a training batch and a probe."""

import dataclasses
from collections.abc import Iterable, Sequence
from typing import Any

from harness_rl.rollout.collect import PromptRow, RolloutGroup
from harness_rl.rollout.rewards import group_advantages, has_learning_signal
from harness_rl.rollout.traces import TokenSequence, TrainingRow, rows_for_rollout

PROBE_TOKENS = 512


def prompts_for_step(
    prompts: Sequence[PromptRow], step: int, per_step: int
) -> tuple[PromptRow, ...]:
    """Walk the task list in order, wrapping around (one epoch = len/per_step)."""
    start = step * per_step
    return tuple(prompts[(start + i) % len(prompts)] for i in range(per_step))


def training_rows(groups: Iterable[RolloutGroup]) -> tuple[TrainingRow, ...]:
    """Keep groups with reward variance; give each rollout its group advantage."""
    return tuple(
        row
        for group in groups
        if has_learning_signal(group.rewards)
        for result, advantage in zip(
            group.results, group_advantages(group.rewards), strict=True
        )
        if advantage is not None
        for row in rows_for_rollout(result.turns, advantage, result.rollout_id)
    )


def probe_from_groups(groups: Iterable[RolloutGroup]) -> TokenSequence | None:
    """Freeze the first captured call as a fixed probe for weight-sync checks.

    Every position after the first is scored, so the trainer and vLLM's
    ``prompt_logprobs`` cover exactly the same tokens.
    """
    first = next((r.turns[0] for g in groups for r in g.results if r.turns), None)
    if first is None:
        return None
    ids = (*first.prompt_ids, *first.completion_ids)[:PROBE_TOKENS]
    return TokenSequence(ids, (0.0,) * len(ids), (0, *(1,) * (len(ids) - 1)))


def mean_abs_difference(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the mean absolute element-wise difference."""
    pairs = list(zip(first, second, strict=True))
    return sum(abs(a - b) for a, b in pairs) / max(len(pairs), 1)


def rollout_summary(groups: Sequence[RolloutGroup]) -> dict[str, Any]:
    """Summarise rewards, failures and tool use for the step log."""
    results = [r for g in groups for r in g.results]
    scored = [r.reward for r in results if r.reward is not None]
    per_rollout = [
        {
            **{f.name: getattr(r, f.name) for f in dataclasses.fields(r)},
            "turns": len(r.turns),  # token ids are too large for the log
        }
        for r in results
    ]
    return {
        "rollouts": len(results),
        "errors": sum(r.error is not None for r in results),
        "unscored": len(results) - len(scored),
        "mean_reward": sum(scored) / max(len(scored), 1),
        "groups_with_signal": sum(has_learning_signal(g.rewards) for g in groups),
        "groups": len(groups),
        "mean_tool_calls": sum(r.tool_calls for r in results) / max(len(results), 1),
        "per_rollout": per_rollout,
    }
