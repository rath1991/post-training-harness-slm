"""Evaluate base vs. trained policies on the same tasks, harness and limits."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from harness_rl.adapters import AdapterRef
from harness_rl.config import RunConfig
from harness_rl.inference import vllm_server
from harness_rl.memory.strategies import launch_vllm
from harness_rl.rollout.collect import (
    RolloutGroup,
    collect_groups,
    make_factory_builder,
    prompt_rows,
)


@dataclass(frozen=True, slots=True)
class EvalSummary:
    """Aggregate results of one model on one split."""

    model: str
    split: str
    tasks: int
    attempts: int
    success_rate: float
    mean_tool_calls: float
    mean_seconds: float
    error_rate: float
    unscored_rate: float


@dataclass(frozen=True, slots=True)
class TaskComparison:
    """Per-task success change from base to trained."""

    improved: tuple[str, ...]
    regressed: tuple[str, ...]
    unchanged: int


def summarise_eval(
    groups: Sequence[RolloutGroup], model: str, split: str
) -> EvalSummary:
    """Reduce evaluation rollouts to headline numbers.

    Unscored rollouts (verifier never ran) count as failures for success rate
    but are also reported separately, so infrastructure faults stay visible.
    """
    results = [r for g in groups for r in g.results]
    n = max(len(results), 1)
    return EvalSummary(
        model=model,
        split=split,
        tasks=len(groups),
        attempts=len(results),
        success_rate=sum((r.reward or 0.0) >= 1.0 for r in results) / n,
        mean_tool_calls=sum(r.tool_calls for r in results) / n,
        mean_seconds=sum(r.seconds for r in results) / n,
        error_rate=sum(r.error is not None for r in results) / n,
        unscored_rate=sum(r.reward is None for r in results) / n,
    )


def task_success(groups: Sequence[RolloutGroup]) -> dict[str, float]:
    """Return each task's fraction of passing attempts."""
    return {
        g.task_name: sum((r or 0.0) >= 1.0 for r in g.rewards) / max(len(g.rewards), 1)
        for g in groups
    }


def compare_tasks(
    base: Mapping[str, float], trained: Mapping[str, float]
) -> TaskComparison:
    """List tasks whose success rate went up or down after training."""
    shared = sorted(base.keys() & trained.keys())
    return TaskComparison(
        improved=tuple(t for t in shared if trained[t] > base[t]),
        regressed=tuple(t for t in shared if trained[t] < base[t]),
        unchanged=sum(trained[t] == base[t] for t in shared),
    )


def run_eval(
    config: RunConfig, adapter: AdapterRef | None, attempts: int, temperature: float
) -> tuple[EvalSummary, tuple[RolloutGroup, ...]]:
    """Serve the model (base if no adapter) and attempt every task in the split."""
    model_name = config.model.served_name if adapter is None else adapter.name
    server = launch_vllm(config, adapter, sleep_mode=False)
    try:
        build = make_factory_builder(config.harbor, config.vllm.base_url, temperature)
        factory = build(model_name)
        groups = collect_groups(
            factory,
            prompt_rows(factory),
            attempts,
            config.harbor.max_concurrent_rollouts,
        )
    finally:
        vllm_server.stop_server(server)
    return summarise_eval(groups, model_name, config.harbor.split), groups
