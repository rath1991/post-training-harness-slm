"""Versioned LoRA adapter references shared by the trainer and vLLM."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AdapterRef:
    """A saved LoRA adapter version, addressable by vLLM under ``name``."""

    version: int
    name: str
    path: Path


def adapter_ref(output_dir: Path, version: int) -> AdapterRef:
    """Return where adapter ``version`` lives and the name vLLM serves it as.

    Every update gets a new name (as TRL does) so a request can never be
    served by a half-swapped adapter.
    """
    name = f"policy-v{version}"
    return AdapterRef(version, name, output_dir.resolve() / "adapters" / name)
