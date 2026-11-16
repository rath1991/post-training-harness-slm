"""Choose reproducible, codebase-disjoint, stratified task partitions."""

import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from harness_rl.tasks.manifest import TaskRecord

# Dataset split labels (README): training pools vs. held-out codebases.
TRAIN_LABELS = frozenset({"train", "new_train"})
EVAL_LABELS = frozenset({"new_test", "test_seen_types"})
OOD_LABELS = frozenset({"test_ood"})

type StratumKey = Callable[[TaskRecord], Hashable]


@dataclass(frozen=True, slots=True)
class SplitSizes:
    """Requested partition sizes."""

    train: int = 400
    eval: int = 100
    ood: int = 50
    smoke: int = 8


@dataclass(frozen=True, slots=True)
class Partitions:
    """The selected tasks for every partition, plus coverage notes."""

    train: tuple[TaskRecord, ...]
    eval: tuple[TaskRecord, ...]
    ood: tuple[TaskRecord, ...]
    smoke: tuple[TaskRecord, ...]
    notes: tuple[str, ...]


def tool_call_bucket(oracle_steps: int) -> str:
    """Bucket the oracle's step count as a proxy for expected tool calls."""
    if oracle_steps <= 1:
        return "1"
    return "2-3" if oracle_steps <= 3 else "4+"


def full_stratum(task: TaskRecord) -> tuple[str, str, int, str]:
    """Stratify by language, task type, difficulty and expected tool calls."""
    bucket = tool_call_bucket(task.oracle_steps)
    return (task.language, task.task_type, task.difficulty, bucket)


def coarse_stratum(task: TaskRecord) -> tuple[str, str]:
    """Stratify small samples by language and task type only."""
    return (task.language, task.task_type)


def group_by(
    tasks: Iterable[TaskRecord], key: StratumKey
) -> dict[Hashable, list[TaskRecord]]:
    """Group tasks by a key, preserving input order inside each group."""
    groups: dict[Hashable, list[TaskRecord]] = defaultdict(list)
    for task in tasks:
        groups[key(task)].append(task)
    return dict(groups)


def proportional_quotas(sizes: Mapping[Hashable, int], n: int) -> dict[Hashable, int]:
    """Allocate n picks across strata proportionally (largest-remainder method)."""
    total = sum(sizes.values())
    n = min(n, total)
    exact = {key: n * size / total for key, size in sizes.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    by_remainder = sorted(exact, key=lambda key: exact[key] - quotas[key], reverse=True)
    for key in by_remainder[: n - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def stratified_sample(
    tasks: Sequence[TaskRecord], n: int, key: StratumKey, rng: random.Random
) -> tuple[TaskRecord, ...]:
    """Sample n tasks so every stratum keeps its share of the population."""
    strata = group_by(sorted(tasks, key=lambda task: task.task_id), key)
    quotas = proportional_quotas({k: len(v) for k, v in strata.items()}, n)
    picked = [
        task
        for stratum_key in sorted(strata, key=repr)
        for task in rng.sample(strata[stratum_key], quotas[stratum_key])
    ]
    return tuple(sorted(picked, key=lambda task: task.task_id))


def codebases(tasks: Iterable[TaskRecord]) -> frozenset[str]:
    """Return the set of codebases used by some tasks."""
    return frozenset(task.codebase for task in tasks)


def assert_codebase_disjoint(
    first: Iterable[TaskRecord], second: Iterable[TaskRecord]
) -> None:
    """Fail if two partitions share a codebase.

    Raises:
        ValueError: Listing the shared codebases.
    """
    shared = codebases(first) & codebases(second)
    if shared:
        raise ValueError(f"partitions share codebases: {sorted(shared)}")


def build_partitions(
    tasks: Sequence[TaskRecord], sizes: SplitSizes, seed: int
) -> Partitions:
    """Select train/eval/OOD/smoke partitions from the dataset's own labels.

    The smoke set is drawn from the training partition, restricted to
    difficulty 1 so a 4B model has a chance of non-zero reward variance.
    """
    rng = random.Random(seed)
    pool = {
        "train": [t for t in tasks if t.split in TRAIN_LABELS],
        "eval": [t for t in tasks if t.split in EVAL_LABELS],
        "ood": [t for t in tasks if t.split in OOD_LABELS],
    }
    train = stratified_sample(pool["train"], sizes.train, full_stratum, rng)
    held_out = stratified_sample(pool["eval"], sizes.eval, full_stratum, rng)
    ood = tuple(sorted(pool["ood"], key=lambda task: task.task_id))
    easy = [t for t in train if t.difficulty == 1]
    smoke = stratified_sample(easy, sizes.smoke, coarse_stratum, rng)
    for other in (held_out, ood):
        assert_codebase_disjoint(train, other)
    return Partitions(train, held_out, ood, smoke, coverage_notes(sizes, ood))


def coverage_notes(sizes: SplitSizes, ood: Sequence[TaskRecord]) -> tuple[str, ...]:
    """Explain any partition that falls short of its requested size."""
    if len(ood) >= sizes.ood:
        return ()
    return (
        f"OOD has {len(ood)} native tasks (< {sizes.ood}); supplement with a "
        "separately verified external set. Held-out tasks are never relabelled.",
    )
