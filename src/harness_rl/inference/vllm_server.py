"""Launch vLLM and drive its admin endpoints over plain HTTP.

Flags follow the TRL/OpenEnv Harbor example: token ids and *processed*
log-probs are required for the capture proxy to produce training traces.
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harness_rl.adapters import AdapterRef
from harness_rl.config import ModelConfig, VllmConfig

# Room for the current adapter plus the one being loaded during a swap.
MAX_LORAS = 2


def serve_command(
    model: ModelConfig,
    vllm: VllmConfig,
    adapter: AdapterRef | None,
    *,
    sleep_mode: bool,
) -> list[str]:
    """Build the ``vllm serve`` argument list.

    Args:
        model: Base model and LoRA shape.
        vllm: Server settings.
        adapter: Adapter to preload at startup, if any.
        sleep_mode: Enable ``/sleep`` and ``/wake_up`` (strategy A).

    Returns:
        The command, ready for ``subprocess.Popen``.
    """
    command = [
        "vllm", "serve", model.base_model,
        "--served-model-name", model.served_name,
        "--host", vllm.host,
        "--port", str(vllm.port),
        "--max-model-len", str(model.max_model_len),
        "--gpu-memory-utilization", str(vllm.gpu_memory_utilization),
        "--max-num-seqs", str(vllm.max_num_seqs),
        "--dtype", "bfloat16",
        "--enable-auto-tool-choice",
        "--tool-call-parser", vllm.tool_call_parser,
        "--logprobs-mode", "processed_logprobs",
        "--return-tokens-as-token-ids",
        "--enable-lora",
        "--max-lora-rank", str(model.lora_rank),
        "--max-loras", str(MAX_LORAS),
    ]  # fmt: skip
    if sleep_mode:
        command.append("--enable-sleep-mode")
    if vllm.enforce_eager:
        command.append("--enforce-eager")
    if adapter is not None:
        command += ["--lora-modules", f"{adapter.name}={adapter.path}"]
    return command


def serve_environment(base: Mapping[str, str]) -> dict[str, str]:
    """Return the process environment that unlocks the admin endpoints."""
    return {
        **base,
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
    }


def start_server(command: Sequence[str], log_path: Path) -> subprocess.Popen:
    """Start vLLM in the background, logging to a file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    return subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=serve_environment(os.environ)
    )


def stop_server(process: subprocess.Popen, timeout_sec: float = 60.0) -> None:
    """Stop vLLM and wait until its process (and GPU memory) is gone."""
    process.terminate()
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def request_json(
    url: str, payload: Mapping[str, Any] | None = None, timeout: float = 60.0
) -> Any:  # noqa: ANN401 - JSON
    """POST (with a payload) or GET a URL and decode a JSON reply, if any."""
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode()
    try:
        return json.loads(body) if body else None
    except json.JSONDecodeError:
        return body


def wait_until_ready(base_url: str, timeout_sec: float) -> float:
    """Poll ``/health`` until the server answers; return seconds waited.

    Raises:
        TimeoutError: If the server is not healthy in time.
    """
    started = time.monotonic()
    while time.monotonic() - started < timeout_sec:
        try:
            request_json(f"{base_url}/health", timeout=5.0)
            return time.monotonic() - started
        except (urllib.error.URLError, ConnectionError):
            time.sleep(2.0)
    raise TimeoutError(f"vLLM at {base_url} not healthy after {timeout_sec}s")


def sleep(base_url: str, level: int = 1) -> None:
    """Offload weights to CPU and drop the KV cache (level 1).

    Level 1 is enough here: LoRA training never changes base weights, so the
    restored copy is still correct and no ``reload_weights`` is needed.
    """
    request_json(f"{base_url}/sleep?level={level}", payload={})


def wake_up(base_url: str) -> None:
    """Restore weights and KV cache after ``sleep``."""
    request_json(f"{base_url}/wake_up", payload={})


def is_sleeping(base_url: str) -> bool:
    """Return True if the server reports being asleep."""
    return bool(request_json(f"{base_url}/is_sleeping")["is_sleeping"])


def load_adapter(base_url: str, adapter: AdapterRef) -> None:
    """Make a saved adapter servable under its versioned name."""
    payload = {"lora_name": adapter.name, "lora_path": str(adapter.path)}
    request_json(f"{base_url}/v1/load_lora_adapter", payload)


def unload_adapter(base_url: str, adapter: AdapterRef) -> None:
    """Free the slot of an adapter that is no longer the policy."""
    request_json(f"{base_url}/v1/unload_lora_adapter", {"lora_name": adapter.name})


def served_models(base_url: str) -> tuple[str, ...]:
    """Return every model and adapter name the server currently serves."""
    return tuple(m["id"] for m in request_json(f"{base_url}/v1/models")["data"])


def parse_prompt_logprobs(
    entries: Sequence[Mapping[str, Any] | None], token_ids: Sequence[int]
) -> tuple[float, ...]:
    """Extract log p(token_i | prefix) from vLLM ``prompt_logprobs`` entries.

    Entry 0 is always None (nothing predicts the first token). Keys may be
    ``"123"`` or ``"token_id:123"`` depending on ``--return-tokens-as-token-ids``.
    """
    values = []
    for entry, token in zip(entries[1:], token_ids[1:], strict=True):
        candidates = {str(token), f"token_id:{token}"}
        match = next(value for key, value in entry.items() if key in candidates)
        values.append(float(match["logprob"]))
    return tuple(values)


def prompt_logprobs(
    base_url: str, model_name: str, token_ids: Sequence[int]
) -> tuple[float, ...]:
    """Score a fixed token sequence with the served model (teacher-forced)."""
    reply = request_json(
        f"{base_url}/v1/completions",
        {
            "model": model_name,
            "prompt": list(token_ids),
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": 0,
        },
    )
    return parse_prompt_logprobs(reply["choices"][0]["prompt_logprobs"], token_ids)
