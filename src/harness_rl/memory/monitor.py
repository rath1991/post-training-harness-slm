"""Measure wall time, whole-GPU memory and system RAM for each pipeline phase.

``nvidia-smi`` is sampled rather than ``torch.cuda`` because vLLM runs in a
separate process: only the driver sees both processes' allocations.
"""

import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import psutil

SAMPLE_INTERVAL_SEC = 0.5
Result = TypeVar("Result")


@dataclass(frozen=True, slots=True)
class PhaseRecord:
    """Resource usage of one named phase."""

    phase: str
    seconds: float
    gpu_peak_mib: int
    gpu_end_mib: int
    ram_peak_mib: int


def gpu_used_mib() -> int:
    """Return memory in use on GPU 0 across all processes, in MiB."""
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(output.splitlines()[0])


def ram_used_mib() -> int:
    """Return system RAM in use, in MiB."""
    return psutil.virtual_memory().used // 2**20


class PeakSampler:
    """Background thread that tracks peak GPU and RAM use while active.

    A class because it owns a thread whose start and stop must be paired.
    """

    def __init__(self) -> None:
        """Prepare an idle sampler."""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.gpu_peak_mib = 0
        self.ram_peak_mib = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.gpu_peak_mib = max(self.gpu_peak_mib, gpu_used_mib())
            self.ram_peak_mib = max(self.ram_peak_mib, ram_used_mib())
            self._stop.wait(SAMPLE_INTERVAL_SEC)

    def __enter__(self) -> "PeakSampler":
        """Start sampling."""
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop sampling and wait for the thread to finish."""
        self._stop.set()
        self._thread.join()


def measured(  # noqa: UP047 - pydocstyle cannot parse PEP 695 generics
    phase: str, action: Callable[[], Result]
) -> tuple[Result, PhaseRecord]:
    """Run an action and return its result with the phase's resource record."""
    started = time.monotonic()
    with PeakSampler() as sampler:
        result = action()
    record = PhaseRecord(
        phase=phase,
        seconds=time.monotonic() - started,
        gpu_peak_mib=max(sampler.gpu_peak_mib, gpu_used_mib()),
        gpu_end_mib=gpu_used_mib(),
        ram_peak_mib=sampler.ram_peak_mib,
    )
    return result, record
