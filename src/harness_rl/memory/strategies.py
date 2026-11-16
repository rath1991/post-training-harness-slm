"""Two ways to share one 16 GB GPU between vLLM rollouts and QLoRA training.

Both satisfy :class:`PolicyRuntime`, so the GRPO pipeline is identical and only
the GPU hand-over differs; that keeps the A/B comparison fair.

* **Sleep mode (A):** vLLM stays up and sleeps (level 1) during training; the
  QLoRA trainer stays resident on the GPU the whole time.
* **Sequential (B):** only one engine exists at a time; vLLM is stopped before
  training and restarted with the new adapter, the trainer is rebuilt from disk.
"""

import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from harness_rl.adapters import AdapterRef, adapter_ref
from harness_rl.config import MemoryStrategy, RunConfig
from harness_rl.inference import vllm_server
from harness_rl.training.qlora import (
    PolicyTrainer,
    free_gpu_memory,
    new_policy,
    restore_policy,
    save_policy,
)


class PolicyRuntime(Protocol):
    """GPU hand-over between the inference phase and the training phase."""

    def start(self) -> AdapterRef:
        """Create policy v0 (identical to the base model) and return it."""
        ...

    def to_inference(self, adapter: AdapterRef) -> None:
        """Make vLLM serve ``adapter`` with as much KV cache as allowed."""
        ...

    def to_training(self, adapter: AdapterRef) -> PolicyTrainer:
        """Free inference memory and return the trainer at ``adapter``."""
        ...

    def release_training(self) -> None:
        """Free training memory before the next inference phase."""
        ...

    def close(self) -> None:
        """Stop every process this runtime started."""
        ...


def log_path(config: RunConfig) -> Path:
    """Return where vLLM's output is appended."""
    return config.output_dir / "logs" / "vllm.log"


def launch_vllm(
    config: RunConfig, adapter: AdapterRef | None, *, sleep_mode: bool
) -> subprocess.Popen:
    """Start vLLM (optionally serving ``adapter``) and block until healthy."""
    command = vllm_server.serve_command(
        config.model, config.vllm, adapter, sleep_mode=sleep_mode
    )
    process = vllm_server.start_server(command, log_path(config))
    vllm_server.wait_until_ready(config.vllm.base_url, config.vllm.startup_timeout_sec)
    return process


class SleepModeRuntime:
    """Strategy A: resident trainer, vLLM sleeps (level 1) while training."""

    def __init__(self, config: RunConfig) -> None:
        """Hold the config; nothing touches the GPU until ``start``."""
        self._config = config
        self._trainer: PolicyTrainer | None = None
        self._server: subprocess.Popen | None = None
        self._served: AdapterRef | None = None

    def start(self) -> AdapterRef:
        """Load the trainer first so vLLM sizes its KV cache around it."""
        config = self._config
        self._trainer = new_policy(config.model, config.grpo.learning_rate)
        initial = save_policy(self._trainer, adapter_ref(config.output_dir, 0))
        self._server = launch_vllm(config, initial, sleep_mode=True)
        self._served = initial
        return initial

    def to_inference(self, adapter: AdapterRef) -> None:
        """Wake vLLM, load the new adapter, drop the previous one."""
        base_url = self._config.vllm.base_url
        if vllm_server.is_sleeping(base_url):
            vllm_server.wake_up(base_url)
        if adapter == self._served:
            return
        vllm_server.load_adapter(base_url, adapter)
        if self._served is not None:
            vllm_server.unload_adapter(base_url, self._served)
        self._served = adapter

    def to_training(self, adapter: AdapterRef) -> PolicyTrainer:  # noqa: ARG002
        """Sleep vLLM; the resident trainer is already at ``adapter``."""
        vllm_server.sleep(self._config.vllm.base_url, level=1)
        return self._require_trainer()

    def release_training(self) -> None:
        """Return cached training blocks so vLLM can wake into them."""
        free_gpu_memory()

    def close(self) -> None:
        """Stop vLLM."""
        if self._server is not None:
            vllm_server.stop_server(self._server)

    def _require_trainer(self) -> PolicyTrainer:
        if self._trainer is None:
            raise RuntimeError("start() must run before training")
        return self._trainer


class SequentialRuntime:
    """Strategy B: exactly one engine on the GPU at any time."""

    def __init__(self, config: RunConfig) -> None:
        """Hold the config; nothing touches the GPU until ``start``."""
        self._config = config
        self._trainer: PolicyTrainer | None = None
        self._server: subprocess.Popen | None = None
        self._served: AdapterRef | None = None

    def start(self) -> AdapterRef:
        """Create and save policy v0, then free the GPU completely."""
        config = self._config
        trainer = new_policy(config.model, config.grpo.learning_rate)
        initial = save_policy(trainer, adapter_ref(config.output_dir, 0))
        del trainer
        free_gpu_memory()
        return initial

    def to_inference(self, adapter: AdapterRef) -> None:
        """Start a fresh vLLM process with ``adapter`` preloaded.

        A step with no trainable groups leaves vLLM running on the same
        adapter, which is reused rather than started twice.
        """
        if self._server is not None and adapter == self._served:
            return
        self.close()
        self._server = launch_vllm(self._config, adapter, sleep_mode=False)
        self._served = adapter

    def to_training(self, adapter: AdapterRef) -> PolicyTrainer:
        """Stop vLLM, then rebuild the trainer and optimizer from disk."""
        self.close()
        config = self._config
        self._trainer = restore_policy(config.model, adapter, config.grpo.learning_rate)
        return self._trainer

    def release_training(self) -> None:
        """Drop the trainer so its GPU memory is actually returned."""
        self._trainer = None
        free_gpu_memory()

    def close(self) -> None:
        """Stop vLLM if it is running."""
        if self._server is not None:
            vllm_server.stop_server(self._server)
            self._server = None


RUNTIMES: Mapping[MemoryStrategy, Callable[[RunConfig], PolicyRuntime]] = {
    MemoryStrategy.SLEEP: SleepModeRuntime,
    MemoryStrategy.SEQUENTIAL: SequentialRuntime,
}


def make_runtime(config: RunConfig) -> PolicyRuntime:
    """Return the runtime for the configured memory strategy."""
    return RUNTIMES[config.strategy](config)
