"""Command-line entry points: the imperative shell around the pipeline.

harness-rl prepare-data --dataset-dir smol-software-harbor --out-dir data/harbor
harness-rl grpo --config configs/smoke_sleep.toml
harness-rl eval --config configs/eval.toml [--adapter runs/x/adapters/policy-v3]
harness-rl compare runs/eval/base.json runs/eval/trained.json
"""

import argparse
import dataclasses
import json
import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from pathlib import Path

from harness_rl.config import RunConfig, load_run_config
from harness_rl.tasks.harbor_layout import convert_task
from harness_rl.tasks.manifest import read_manifest
from harness_rl.tasks.splits import SplitSizes, build_partitions

TRACKED_PACKAGES = (
    "torch",
    "vllm",
    "transformers",
    "peft",
    "bitsandbytes",
    "trl",
    "openenv",
    "harbor",
)


def package_versions(names: Sequence[str]) -> dict[str, str]:
    """Return installed versions, marking packages that are absent."""

    def version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "not installed"

    return {name: version(name) for name in names}


def environment_record() -> dict[str, object]:
    """Capture what this run executed on, for reproducibility."""
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu,
        "packages": package_versions(TRACKED_PACKAGES),
    }


def write_json(path: Path, value: object) -> None:
    """Write a JSON document, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def prepare_data(args: argparse.Namespace) -> None:
    """Select partitions and write each in Harbor layout, one dir per split."""
    dataset = Path(args.dataset_dir)
    partitions = build_partitions(
        read_manifest(dataset / "manifest.jsonl"), SplitSizes(), args.seed
    )
    out = Path(args.out_dir)
    for split in ("train", "eval", "ood", "smoke"):
        for task in getattr(partitions, split):
            convert_task(dataset / task.path, out / split / Path(task.path).name)
    write_json(out / "partitions.json", dataclasses.asdict(partitions))
    for note in partitions.notes:
        print(f"NOTE: {note}")


def run_grpo_command(args: argparse.Namespace) -> None:
    """Run black-box GRPO with the configured memory strategy."""
    from harness_rl.memory.strategies import make_runtime
    from harness_rl.pipeline import PipelineContext, jsonl_sink, run_grpo
    from harness_rl.rollout.collect import make_factory_builder, prompt_rows

    config = load_run_config(Path(args.config))
    write_json(config.output_dir / "environment.json", environment_record())
    write_json(config.output_dir / "config.json", dataclasses.asdict(config))
    build = make_factory_builder(
        config.harbor, config.vllm.base_url, config.grpo.temperature
    )
    context = PipelineContext(
        config=config,
        runtime=make_runtime(config),
        build_factory=build,
        prompts=prompt_rows(build(config.model.served_name)),
    )
    final = run_grpo(context, jsonl_sink(config.output_dir / "metrics.jsonl"))
    print(f"final adapter: {final.path}")


def run_eval_command(args: argparse.Namespace) -> None:
    """Evaluate the base model, or a trained adapter, on the config's split."""
    from harness_rl.adapters import AdapterRef
    from harness_rl.evaluation import run_eval, task_success

    config: RunConfig = load_run_config(Path(args.config))
    adapter = None
    if args.adapter:
        path = Path(args.adapter).resolve()
        adapter = AdapterRef(version=-1, name=path.name, path=path)
    summary, groups = run_eval(config, adapter, args.attempts, args.temperature)
    write_json(
        Path(args.out),
        {"summary": dataclasses.asdict(summary), "tasks": task_success(groups)},
    )
    print(json.dumps(dataclasses.asdict(summary), indent=2))


def compare_command(args: argparse.Namespace) -> None:
    """Print per-task improvements and regressions between two eval files."""
    from harness_rl.evaluation import compare_tasks

    def tasks(path: str) -> Mapping[str, float]:
        return json.loads(Path(path).read_text(encoding="utf-8"))["tasks"]

    result = compare_tasks(tasks(args.base), tasks(args.trained))
    print(json.dumps(dataclasses.asdict(result), indent=2))


def inspect_trace_command(args: argparse.Namespace) -> None:
    """Validate a saved OpenEnv TrainingTrace before any training (stage 0).

    Raises:
        SystemExit: If the trace is not usable for training.
    """
    from openenv.core.harness import TrainingTrace

    from harness_rl.reports import check_trace
    from harness_rl.rollout.collect import turn_tokens

    trace = TrainingTrace.model_validate_json(Path(args.trace).read_text())
    prompt_mask_sum = sum(
        sum(turn.loss_mask[: len(turn.prompt_token_ids)]) for turn in trace.turns
    )
    check = check_trace([turn_tokens(t) for t in trace.turns], prompt_mask_sum)
    print(json.dumps(dataclasses.asdict(check), indent=2))
    if check.problems:
        raise SystemExit(1)


def compare_strategies_command(args: argparse.Namespace) -> None:
    """Print the Day 1 sleep-mode vs. sequential comparison table."""
    from harness_rl.reports import markdown_table, strategy_report

    def records(run_dir: str) -> list[dict]:
        lines = (Path(run_dir) / "metrics.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    reports = [strategy_report(Path(d).name, records(d)) for d in args.run_dirs]
    print(markdown_table(reports))


COMMANDS: Mapping[str, Callable[[argparse.Namespace], None]] = {
    "prepare-data": prepare_data,
    "inspect-trace": inspect_trace_command,
    "grpo": run_grpo_command,
    "compare-strategies": compare_strategies_command,
    "eval": run_eval_command,
    "compare": compare_command,
}


def parser() -> argparse.ArgumentParser:
    """Define the command-line interface."""
    root = argparse.ArgumentParser(prog="harness-rl")
    sub = root.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-data")
    prep.add_argument("--dataset-dir", required=True)
    prep.add_argument("--out-dir", default="data/harbor")
    prep.add_argument("--seed", type=int, default=0)
    sub.add_parser("inspect-trace").add_argument("trace")
    sub.add_parser("grpo").add_argument("--config", required=True)
    sub.add_parser("compare-strategies").add_argument("run_dirs", nargs="+")
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--adapter")
    evaluate.add_argument("--attempts", type=int, default=1)
    evaluate.add_argument("--temperature", type=float, default=0.7)
    evaluate.add_argument("--out", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("base")
    compare.add_argument("trained")
    return root


def main() -> None:
    """Dispatch to the chosen subcommand."""
    args = parser().parse_args()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
