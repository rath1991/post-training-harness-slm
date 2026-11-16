import pytest

from harness_rl.rollout.batching import (
    mean_abs_difference,
    probe_from_groups,
    prompts_for_step,
    training_rows,
)
from harness_rl.rollout.collect import PromptRow, RolloutGroup, RolloutResult
from harness_rl.rollout.rewards import group_advantages, has_learning_signal
from harness_rl.rollout.traces import TurnTokens

TURN = TurnTokens((1, 2), (3,), (-0.5,), (1,))


def result(reward: float | None, rollout_id: str = "r") -> RolloutResult:
    return RolloutResult(rollout_id, "task", reward, (TURN,), 2, 1.0)


def test_advantages_are_group_standardised() -> None:
    advantages = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert advantages == pytest.approx([1.0, -1.0, 1.0, -1.0])


def test_unscored_rollouts_are_excluded_not_zeroed() -> None:
    advantages = group_advantages([1.0, None, 0.0])
    assert advantages[1] is None
    assert advantages[0] == pytest.approx(1.0)


def test_uniform_groups_have_no_signal() -> None:
    assert not has_learning_signal([1.0, 1.0, None])
    assert not has_learning_signal([0.0, 0.0])
    assert has_learning_signal([0.0, 1.0])


def test_training_rows_skip_groups_without_variance() -> None:
    flat = RolloutGroup("a", (result(0.0), result(0.0)))
    mixed = RolloutGroup("b", (result(1.0, "win"), result(0.0, "loss")))
    rows = training_rows([flat, mixed])
    assert {row.rollout_id: row.advantage > 0 for row in rows} == {
        "win": True,
        "loss": False,
    }


def test_prompts_wrap_around_the_task_list() -> None:
    prompts = [PromptRow((), f"t{i}", i) for i in range(3)]
    assert [p.task_name for p in prompts_for_step(prompts, 1, 2)] == ["t2", "t0"]


def test_probe_scores_every_token_after_the_first() -> None:
    probe = probe_from_groups([RolloutGroup("a", (result(1.0),))])
    assert probe.input_ids == (1, 2, 3)
    assert probe.loss_mask == (0, 1, 1)


def test_mean_abs_difference() -> None:
    assert mean_abs_difference([0.0, -1.0], [0.5, -1.5]) == pytest.approx(0.5)
