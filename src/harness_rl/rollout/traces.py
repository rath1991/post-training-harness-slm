"""Turn captured model calls into token-level training rows.

Index convention used everywhere: ``loss_mask[i] == 1`` means token
``input_ids[i]`` was sampled by the policy and is trained, so its log-prob is
predicted from ``input_ids[:i]``. Everything else (system prompt, tool schemas,
tool output, file contents) has mask 0 and never enters the policy loss.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TurnTokens:
    """One captured model call, exactly as the inference engine saw it."""

    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    completion_logps: tuple[float, ...]
    completion_mask: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TokenSequence:
    """A merged chain of turns: one forward pass covers all of them."""

    input_ids: tuple[int, ...]
    old_logps: tuple[float, ...]
    loss_mask: tuple[int, ...]

    @property
    def n_trained(self) -> int:
        """Return how many tokens contribute to the loss."""
        return sum(self.loss_mask)


@dataclass(frozen=True, slots=True)
class TrainingRow:
    """A token sequence with the advantage of the rollout that produced it."""

    sequence: TokenSequence
    advantage: float
    rollout_id: str


class MalformedTurnError(ValueError):
    """A captured turn whose token, log-prob and mask lengths disagree."""


def validate_turn(turn: TurnTokens) -> TurnTokens:
    """Return the turn unchanged, or raise if its arrays are misaligned.

    Raises:
        MalformedTurnError: If completion arrays have different lengths.
    """
    lengths = {
        len(turn.completion_ids),
        len(turn.completion_logps),
        len(turn.completion_mask),
    }
    if len(lengths) != 1:
        raise MalformedTurnError(f"completion arrays differ in length: {lengths}")
    return turn


def sequence_from_turn(turn: TurnTokens) -> TokenSequence:
    """Start a new sequence: the prompt is context, the completion is trained."""
    n_prompt = len(turn.prompt_ids)
    return TokenSequence(
        input_ids=(*turn.prompt_ids, *turn.completion_ids),
        old_logps=(*(0.0,) * n_prompt, *turn.completion_logps),
        loss_mask=(*(0,) * n_prompt, *turn.completion_mask),
    )


def extends(sequence: TokenSequence, turn: TurnTokens) -> bool:
    """Return True if the turn's prompt starts with the whole sequence."""
    n = len(sequence.input_ids)
    return turn.prompt_ids[:n] == sequence.input_ids


def append_turn(sequence: TokenSequence, turn: TurnTokens) -> TokenSequence:
    """Extend a sequence with a turn whose prompt continues it.

    The new prompt tail (tool results, harness messages) is context only.
    """
    gap = turn.prompt_ids[len(sequence.input_ids) :]
    return TokenSequence(
        input_ids=(*sequence.input_ids, *gap, *turn.completion_ids),
        old_logps=(*sequence.old_logps, *(0.0,) * len(gap), *turn.completion_logps),
        loss_mask=(*sequence.loss_mask, *(0,) * len(gap), *turn.completion_mask),
    )


def merge_turns(turns: Iterable[TurnTokens]) -> tuple[TokenSequence, ...]:
    """Merge turns linked by exact token prefix into as few sequences as possible.

    Harnesses that rewrite or compact their history break the prefix, which
    starts a new sequence, mirroring OpenEnv's own turn-graph rule.
    """
    sequences: list[TokenSequence] = []
    for turn in map(validate_turn, turns):
        parent = next(
            (i for i, seq in enumerate(sequences) if extends(seq, turn)), None
        )
        if parent is None:
            sequences.append(sequence_from_turn(turn))
        else:
            sequences[parent] = append_turn(sequences[parent], turn)
    return tuple(seq for seq in sequences if seq.n_trained > 0)


def rows_for_rollout(
    turns: Sequence[TurnTokens], advantage: float, rollout_id: str
) -> tuple[TrainingRow, ...]:
    """Build training rows for one rollout; every row shares its advantage."""
    return tuple(
        TrainingRow(sequence, advantage, rollout_id) for sequence in merge_turns(turns)
    )
