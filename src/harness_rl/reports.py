"""Turn run logs into the Day 1 evidence: trace validity and strategy comparison."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from harness_rl.rollout.traces import TurnTokens, merge_turns, validate_turn

SWITCH_PHASES = frozenset({"to_inference", "to_training", "release_training"})


@dataclass(frozen=True, slots=True)
class TraceCheck:
    """Scope s.2 items 4-5: is the captured trace usable for training?"""

    turns: int
    sequences: int
    prompt_tokens_trained: int
    completion_tokens: int
    completion_tokens_trained: int
    logps_finite: bool
    problems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StrategyReport:
    """Scope s.4 measurements for one memory strategy."""

    name: str
    steps_trained: int
    gpu_peak_mib: int
    ram_peak_mib: int
    rollout_seconds: float
    rollouts: int
    switch_seconds: float
    train_step_seconds: float
    total_seconds: float
    mean_abs_logp_diff_first_step: float | None
    worst_served_vs_trainer: float | None
    min_served_change: float | None


def check_trace(turns: Sequence[TurnTokens], prompt_mask_sum: int) -> TraceCheck:
    """Verify captured turns: aligned arrays, masked prompts, trainable tokens."""
    problems = []
    if prompt_mask_sum:
        problems.append(f"{prompt_mask_sum} prompt tokens are marked trainable")
    valid = [validate_turn(turn) for turn in turns]
    completion = sum(len(t.completion_ids) for t in valid)
    trained = sum(sum(t.completion_mask) for t in valid)
    logps = [lp for t in valid for lp in t.completion_logps]
    finite = all(lp <= 0.0 and lp > float("-inf") for lp in logps)
    if not trained:
        problems.append("no completion token is trainable")
    if not finite:
        problems.append("log-probs out of range (expected finite, <= 0)")
    return TraceCheck(
        turns=len(valid),
        sequences=len(merge_turns(valid)),
        prompt_tokens_trained=prompt_mask_sum,
        completion_tokens=completion,
        completion_tokens_trained=trained,
        logps_finite=finite,
        problems=tuple(problems),
    )


def phases(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Flatten every phase record from a metrics.jsonl stream."""
    return [phase for record in records for phase in record.get("phases") or []]


def sync_values(records: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    """Collect one sync-check field across all steps that have one."""
    return [r["sync"][key] for r in records if r.get("sync")]


def strategy_report(name: str, records: Sequence[Mapping[str, Any]]) -> StrategyReport:
    """Summarise one strategy's metrics.jsonl for the comparison table."""
    all_phases = phases(records)
    trained = [r["train"] for r in records if r.get("train")]
    served_vs_trainer = sync_values(records, "served_vs_trainer")
    served_change = sync_values(records, "served_change")

    def seconds(names: frozenset[str]) -> float:
        return sum(p["seconds"] for p in all_phases if p["phase"] in names)

    return StrategyReport(
        name=name,
        steps_trained=len(trained),
        gpu_peak_mib=max((p["gpu_peak_mib"] for p in all_phases), default=0),
        ram_peak_mib=max((p["ram_peak_mib"] for p in all_phases), default=0),
        rollout_seconds=seconds(frozenset({"rollouts"})),
        rollouts=sum(r["rollouts"]["rollouts"] for r in records if r.get("rollouts")),
        switch_seconds=seconds(SWITCH_PHASES),
        train_step_seconds=seconds(frozenset({"train_step"})),
        total_seconds=sum(p["seconds"] for p in all_phases),
        mean_abs_logp_diff_first_step=(
            trained[0]["mean_abs_logp_diff"] if trained else None
        ),
        worst_served_vs_trainer=max(served_vs_trainer, default=None),
        min_served_change=min(served_change, default=None),
    )


def markdown_table(reports: Sequence[StrategyReport]) -> str:
    """Render reports side by side as a Markdown table."""
    fields = [f for f in StrategyReport.__dataclass_fields__ if f != "name"]
    header = "| metric | " + " | ".join(r.name for r in reports) + " |"
    divider = "|---|" + "---:|" * len(reports)
    rows = [
        f"| {field} | "
        + " | ".join(format_value(getattr(r, field)) for r in reports)
        + " |"
        for field in fields
    ]
    return "\n".join([header, divider, *rows])


def format_value(value: object) -> str:
    """Format a table cell compactly."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return "n/a" if value is None else str(value)
