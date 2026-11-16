"""Build privileged teacher inputs that score the student's own sampled tokens.

The teacher sees exactly the student's captured context plus one extra message
(the verified reference patch), inserted after the first chat message. The
completion tokens are identical in both sequences, so teacher and student
distributions line up position by position over the completion.

The reference only ever reaches the *teacher's* forward pass; it never enters
the Harbor sandbox (see ``tasks.harbor_layout.leaked_paths``).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from harness_rl.rollout.traces import TurnTokens


@dataclass(frozen=True, slots=True)
class DistillExample:
    """One captured turn, seen by the student and by the privileged teacher."""

    student_ids: tuple[int, ...]
    teacher_ids: tuple[int, ...]
    completion_mask: tuple[int, ...]

    @property
    def n_completion(self) -> int:
        """Return how many trailing tokens are the student's completion."""
        return len(self.completion_mask)


def privileged_message(reference_patch: str) -> str:
    """Wrap the verified solution as the teacher-only hint."""
    return (
        "Privileged reference (never shown to the agent): a verified solution "
        "for this task is below. Use it to judge which next actions are good.\n"
        f"{reference_patch}"
    )


def find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> int | None:
    """Return the start index of the first occurrence of pattern, if any."""
    width = len(pattern)
    return next(
        (
            i
            for i in range(len(sequence) - width + 1)
            if tuple(sequence[i : i + width]) == tuple(pattern)
        ),
        None,
    )


def insertion_point(prompt_ids: Sequence[int], message_end: Sequence[int]) -> int:
    """Return the index just after the first chat message (0 if none found)."""
    start = find_subsequence(prompt_ids, message_end)
    return 0 if start is None else start + len(message_end)


def distill_example(
    turn: TurnTokens, privileged_block: Sequence[int], message_end: Sequence[int]
) -> DistillExample:
    """Pair a captured turn with its privileged teacher sequence.

    Args:
        turn: A captured student model call.
        privileged_block: Token ids of a complete chat message carrying the
            privileged hint (already rendered with the chat template).
        message_end: Token ids that end a chat message (Qwen: ``<|im_end|>``
            followed by the newline token).

    Returns:
        Student and teacher token sequences sharing the same completion.
    """
    cut = insertion_point(turn.prompt_ids, message_end)
    teacher_prompt = (*turn.prompt_ids[:cut], *privileged_block, *turn.prompt_ids[cut:])
    return DistillExample(
        student_ids=(*turn.prompt_ids, *turn.completion_ids),
        teacher_ids=(*teacher_prompt, *turn.completion_ids),
        completion_mask=turn.completion_mask,
    )
