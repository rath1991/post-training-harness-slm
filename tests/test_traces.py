import pytest

from harness_rl.rollout.traces import (
    MalformedTurnError,
    TurnTokens,
    merge_turns,
    rows_for_rollout,
)


def turn(
    prompt: list[int], completion: list[int], mask: list[int] | None = None
) -> TurnTokens:
    mask = mask if mask is not None else (1,) * len(completion)
    logps = tuple(-0.1 * (i + 1) for i in range(len(completion)))
    return TurnTokens(tuple(prompt), tuple(completion), logps, tuple(mask))


def test_prompt_tokens_are_never_trained() -> None:
    (sequence,) = merge_turns([turn([1, 2, 3], [10, 11])])
    assert sequence.input_ids == (1, 2, 3, 10, 11)
    assert sequence.loss_mask == (0, 0, 0, 1, 1)
    assert sequence.old_logps[:3] == (0.0, 0.0, 0.0)


def test_partial_completion_mask_is_preserved() -> None:
    (sequence,) = merge_turns([turn([10, 11], [20, 21, 22], mask=[1, 0, 1])])
    assert sequence.loss_mask == (0, 0, 1, 0, 1)  # the OpenEnv docs example


def test_prefix_turns_merge_and_tool_output_is_masked() -> None:
    first = turn([1, 2], [10, 11])
    # Second call's prompt = whole first sequence + tool output [50, 51].
    second = turn([1, 2, 10, 11, 50, 51], [12])
    (sequence,) = merge_turns([first, second])
    assert sequence.input_ids == (1, 2, 10, 11, 50, 51, 12)
    assert sequence.loss_mask == (0, 0, 1, 1, 0, 0, 1)


def test_rewritten_history_starts_a_new_sequence() -> None:
    first = turn([1, 2], [10])
    compacted = turn([1, 99], [11])  # harness summarised its context
    assert len(merge_turns([first, compacted])) == 2


def test_fully_masked_sequences_are_dropped() -> None:
    assert merge_turns([turn([1], [10], mask=[0])]) == ()


def test_misaligned_turn_is_rejected() -> None:
    bad = TurnTokens((1,), (10, 11), (-0.1,), (1, 1))
    with pytest.raises(MalformedTurnError):
        merge_turns([bad])


def test_rows_share_the_rollout_advantage() -> None:
    rows = rows_for_rollout([turn([1], [10]), turn([7], [8])], 0.5, "r0")
    assert [row.advantage for row in rows] == [0.5, 0.5]
