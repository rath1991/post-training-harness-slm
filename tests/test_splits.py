import random

import pytest

from harness_rl.tasks.manifest import TaskRecord, parse_task
from harness_rl.tasks.splits import (
    SplitSizes,
    assert_codebase_disjoint,
    build_partitions,
    full_stratum,
    proportional_quotas,
    stratified_sample,
)


def task(
    i: int, split: str, codebase: str | None = None, language: str = "go"
) -> TaskRecord:
    return TaskRecord(
        task_id=f"t{i:03d}",
        codebase=codebase or f"cb{i}",
        language=language,
        task_type="fix_bug",
        difficulty=1,
        split=split,
        oracle_steps=1,
        search=False,
        path=f"tasks/t{i:03d}",
    )


def dataset() -> list[TaskRecord]:
    train = [task(i, "new_train", language=("go", "c")[i % 2]) for i in range(60)]
    held_out = [task(100 + i, "new_test") for i in range(20)]
    ood = [task(200 + i, "test_ood") for i in range(5)]
    return train + held_out + ood


def test_quotas_sum_to_n_and_stay_proportional() -> None:
    quotas = proportional_quotas({"a": 30, "b": 10}, 8)
    assert quotas == {"a": 6, "b": 2}


def test_sampling_is_reproducible_for_a_seed() -> None:
    tasks = dataset()
    first = stratified_sample(tasks, 10, full_stratum, random.Random(1))
    second = stratified_sample(tasks, 10, full_stratum, random.Random(1))
    assert first == second


def test_partitions_respect_sizes_and_codebases() -> None:
    sizes = SplitSizes(train=40, eval=10, ood=50, smoke=6)
    parts = build_partitions(dataset(), sizes, seed=0)
    assert (len(parts.train), len(parts.eval), len(parts.smoke)) == (40, 10, 6)
    assert set(parts.smoke) <= set(parts.train)
    assert {t.split for t in parts.eval} == {"new_test"}


def test_short_ood_is_reported_not_padded() -> None:
    parts = build_partitions(dataset(), SplitSizes(train=10, eval=5), seed=0)
    assert len(parts.ood) == 5
    assert "OOD has 5 native tasks" in parts.notes[0]


def test_shared_codebase_is_rejected() -> None:
    with pytest.raises(ValueError, match="share codebases"):
        assert_codebase_disjoint([task(1, "train", "x")], [task(2, "new_test", "x")])


def test_manifest_null_search_means_false() -> None:
    row = {
        "id": "a",
        "codebase": "c",
        "language": "go",
        "task_type": "fix_bug",
        "difficulty": 2,
        "split": "train",
        "oracle_steps": None,
        "search": None,
        "path": "tasks/a",
    }
    assert parse_task(row).search is False
