"""Run configuration: frozen dataclasses parsed once from a TOML file."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

# TypeVar rather than PEP 695 syntax: pydocstyle cannot parse `def f[T](...)`.
Section = TypeVar("Section")


class MemoryStrategy(StrEnum):
    """How the single GPU is shared between vLLM and the QLoRA trainer."""

    SLEEP = "sleep"
    SEQUENTIAL = "sequential"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Student model and LoRA adapter shape."""

    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    served_name: str = "student-base"
    max_model_len: int = 12288
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True, slots=True)
class VllmConfig:
    """vLLM server launch settings."""

    host: str = "127.0.0.1"
    port: int = 8000
    gpu_memory_utilization: float = 0.62
    max_num_seqs: int = 4
    tool_call_parser: str = "hermes"
    enforce_eager: bool = False
    startup_timeout_sec: float = 600.0

    @property
    def base_url(self) -> str:
        """Return the server's HTTP root."""
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True, slots=True)
class HarborConfig:
    """OpenEnv Harbor server and per-rollout harness settings."""

    server_url: str = "http://127.0.0.1:8200"
    split: str = "smoke"
    harness: str = "pi"
    sandbox: str = "docker"
    agent_timeout_sec: float = 600.0
    agent_step_limit: int = 30
    reward_key: str = "reward"
    max_concurrent_rollouts: int = 4


@dataclass(frozen=True, slots=True)
class GrpoConfig:
    """GRPO optimisation settings (beta=0: no reference model, no KL term)."""

    num_generations: int = 4
    groups_per_step: int = 2
    max_steps: int = 3
    learning_rate: float = 1e-5
    temperature: float = 1.0
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28
    is_clip_min: float = 0.5
    is_clip_max: float = 5.0
    max_grad_norm: float = 1.0
    max_train_tokens: int = 12288
    logprob_chunk: int = 1024


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything one experiment run needs."""

    name: str
    output_dir: Path
    strategy: MemoryStrategy
    seed: int = 0
    model: ModelConfig = ModelConfig()
    vllm: VllmConfig = VllmConfig()
    harbor: HarborConfig = HarborConfig()
    grpo: GrpoConfig = GrpoConfig()


def _section(cls: type[Section], raw: Mapping[str, Any]) -> Section:  # noqa: UP047
    """Build a config section, turning TOML lists into tuples."""
    values = {key: tuple(v) if isinstance(v, list) else v for key, v in raw.items()}
    return cls(**values)


def parse_run_config(raw: Mapping[str, Any]) -> RunConfig:
    """Convert a parsed TOML mapping into a validated RunConfig.

    Raises:
        TypeError: If a section contains an unknown key.
        ValueError: If the strategy name is unknown.
    """
    run = raw["run"]
    return RunConfig(
        name=run["name"],
        output_dir=Path(run["output_dir"]),
        strategy=MemoryStrategy(run["strategy"]),
        seed=run.get("seed", 0),
        model=_section(ModelConfig, raw.get("model", {})),
        vllm=_section(VllmConfig, raw.get("vllm", {})),
        harbor=_section(HarborConfig, raw.get("harbor", {})),
        grpo=_section(GrpoConfig, raw.get("grpo", {})),
    )


def load_run_config(path: Path) -> RunConfig:
    """Read and parse a TOML run configuration."""
    return parse_run_config(tomllib.loads(path.read_text(encoding="utf-8")))
