"""Parse smol-software-harbor's manifest.jsonl into typed task records."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """One dataset task, as described by its manifest row."""

    task_id: str
    codebase: str
    language: str
    task_type: str
    difficulty: int
    split: str
    oracle_steps: int
    search: bool
    path: str


def parse_task(row: Mapping[str, Any]) -> TaskRecord:
    """Convert one manifest row into a TaskRecord.

    Raises:
        KeyError: If a required column is missing.
    """
    return TaskRecord(
        task_id=row["id"],
        codebase=row["codebase"],
        language=row["language"],
        task_type=row["task_type"],
        difficulty=int(row["difficulty"]),
        split=row["split"],
        oracle_steps=int(row.get("oracle_steps") or 0),
        # The manifest leaves `search` null on non-search tasks.
        search=bool(row.get("search")),
        path=row["path"],
    )


def read_manifest(path: Path) -> tuple[TaskRecord, ...]:
    """Read every task from a manifest.jsonl file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(parse_task(json.loads(line)) for line in lines if line.strip())
