"""Black-box GRPO loop: harness rollouts -> verifier rewards -> QLoRA update.

One iteration::

    to_inference(adapter vN)          # strategy-specific GPU hand-over
    rollouts via Pi in Harbor         # black box: OpenEnv captures tokens
    sync check on probe tokens        # does vLLM serve what we trained?
    to_training(vN) -> train_step     # GRPO on captured, masked tokens
    save adapter vN+1, release        # next rollouts use the new policy
"""

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from harness_rl.adapters import AdapterRef, adapter_ref
from harness_rl.config import RunConfig
from harness_rl.inference import vllm_server
from harness_rl.memory.monitor import PhaseRecord, measured
from harness_rl.memory.strategies import PolicyRuntime
from harness_rl.rollout.batching import (
    mean_abs_difference,
    probe_from_groups,
    prompts_for_step,
    rollout_summary,
    training_rows,
)
from harness_rl.rollout.collect import FactoryBuilder, PromptRow, collect_groups
from harness_rl.rollout.traces import TokenSequence, TrainingRow
from harness_rl.training.qlora import save_policy
from harness_rl.training.step import StepMetrics, probe_logps, train_step

type Sink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Everything an iteration needs that never changes during a run."""

    config: RunConfig
    runtime: PolicyRuntime
    build_factory: FactoryBuilder
    prompts: tuple[PromptRow, ...]


@dataclass(frozen=True, slots=True)
class LoopState:
    """What carries over from one iteration to the next."""

    step: int
    adapter: AdapterRef
    probe: TokenSequence | None = None
    trainer_probe: tuple[float, ...] | None = None
    served_probe: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class SyncCheck:
    """Evidence that the inference engine serves the freshly trained adapter.

    ``served_vs_trainer`` should be small (only BF16-vs-NF4 numerics);
    ``served_change`` must be non-zero after an update (outputs changed).
    """

    adapter: str
    adapter_listed: bool
    served_vs_trainer: float
    served_change: float


def sync_check(state: LoopState, served: tuple[float, ...], base_url: str) -> SyncCheck:
    """Compare the served adapter with the trainer that produced it."""
    return SyncCheck(
        adapter=state.adapter.name,
        adapter_listed=state.adapter.name in vllm_server.served_models(base_url),
        served_vs_trainer=mean_abs_difference(served, state.trainer_probe or served),
        served_change=mean_abs_difference(served, state.served_probe or served),
    )


def training_phase(
    context: PipelineContext, state: LoopState, rows: Sequence[TrainingRow]
) -> tuple[StepMetrics, tuple[float, ...] | None, AdapterRef, list[PhaseRecord]]:
    """Hand the GPU to the trainer, update once, save vN+1, hand it back."""
    config = context.config
    runtime = context.runtime
    trainer, to_train = measured(
        "to_training", partial(runtime.to_training, state.adapter)
    )
    metrics, train = measured(
        "train_step", partial(train_step, trainer, rows, config.grpo)
    )
    probe = (
        None if state.probe is None else probe_logps(trainer, state.probe, config.grpo)
    )
    next_ref = adapter_ref(config.output_dir, state.adapter.version + 1)
    adapter, save = measured("save_adapter", partial(save_policy, trainer, next_ref))
    del trainer  # the runtime decides whether the model leaves the GPU
    _, release = measured("release_training", runtime.release_training)
    return metrics, probe, adapter, [to_train, train, save, release]


def served_logps(
    config: RunConfig, adapter: AdapterRef, probe: TokenSequence | None
) -> tuple[float, ...] | None:
    """Score the probe with whatever vLLM serves under the adapter's name."""
    if probe is None:
        return None
    return vllm_server.prompt_logprobs(
        config.vllm.base_url, adapter.name, probe.input_ids
    )


def iteration(context: PipelineContext, state: LoopState) -> tuple[LoopState, dict]:
    """Run one rollout-and-update cycle and return the new state and its record."""
    config = context.config
    _, to_infer = measured(
        "to_inference", partial(context.runtime.to_inference, state.adapter)
    )
    batch = prompts_for_step(context.prompts, state.step, config.grpo.groups_per_step)
    groups, rollouts = measured(
        "rollouts",
        partial(
            collect_groups,
            context.build_factory(state.adapter.name),
            batch,
            config.grpo.num_generations,
            config.harbor.max_concurrent_rollouts,
        ),
    )
    probe = state.probe or probe_from_groups(groups)
    served = served_logps(config, state.adapter, probe)
    check = None
    if served is not None and state.step > 0:
        check = sync_check(state, served, config.vllm.base_url)
    record = {"step": state.step, "adapter": state.adapter.name, "sync": check}
    record["rollouts"] = rollout_summary(groups)
    after = dataclasses.replace(
        state, step=state.step + 1, probe=probe, served_probe=served
    )
    rows = training_rows(groups)
    if not rows:  # every group all-pass or all-fail: no gradient, keep policy
        return after, record | {"train": None, "phases": [to_infer, rollouts]}
    metrics, trainer_probe, adapter, phases = training_phase(context, after, rows)
    record |= {"train": metrics, "phases": [to_infer, rollouts, *phases]}
    return (
        dataclasses.replace(after, adapter=adapter, trainer_probe=trainer_probe),
        record,
    )


def to_jsonable(value: Any) -> Any:  # noqa: ANN401 - JSON encoder hook
    """Encode dataclasses and paths for the JSONL log."""
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return str(value) if isinstance(value, Path) else repr(value)


def jsonl_sink(path: Path) -> Sink:
    """Return a sink that appends one JSON record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(record: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=to_jsonable) + "\n")

    return write


def final_sync(context: PipelineContext, state: LoopState, sink: Sink) -> None:
    """Serve the last adapter once more to verify the final update reached vLLM."""
    if state.probe is None or state.trainer_probe is None:
        return
    context.runtime.to_inference(state.adapter)
    served = served_logps(context.config, state.adapter, state.probe)
    check = sync_check(state, served, context.config.vllm.base_url)
    sink({"step": state.step, "event": "final_sync", "sync": check})


def run_grpo(context: PipelineContext, sink: Sink) -> AdapterRef:
    """Run the configured number of GRPO steps; return the final adapter."""
    runtime = context.runtime
    try:
        adapter, start = measured("start", runtime.start)
        sink({"step": -1, "event": "start", "phases": [start]})
        state = LoopState(step=0, adapter=adapter)
        for _ in range(context.config.grpo.max_steps):
            state, record = iteration(context, state)
            sink(record)
        final_sync(context, state, sink)
        return state.adapter
    finally:
        runtime.close()
