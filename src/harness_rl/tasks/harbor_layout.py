"""Convert smol-software-harbor tasks into Harbor's single-step task layout.

The stock Harbor CLI does not stage ``environment/workdir/`` or run ``setup.sh``
(dataset README), so each task gets a Dockerfile that bakes the starting files
into the shared base image. Only ``environment/`` is the Docker build context,
which is what keeps ``solution/`` and the hidden tests out of the sandbox.
"""

import re
import shutil
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

BASE_IMAGE = "harbor-swe-base:2"
COPIED_ENTRIES = ("instruction.md", "tests", "solution")
FORBIDDEN_IN_ENVIRONMENT = frozenset({"solution", "tests"})


def environment_dockerfile(base_image: str) -> str:
    """Return a Dockerfile that stages the task's workdir and runs setup.sh."""
    return (
        f"FROM {base_image}\n"
        "COPY workdir/ /workdir/\n"
        "COPY setup.sh /tmp/task-setup.sh\n"
        "RUN bash /tmp/task-setup.sh && rm /tmp/task-setup.sh\n"
        "WORKDIR /workdir\n"
    )


def without_prebuilt_image(task_toml: str) -> str:
    """Drop ``docker_image`` so Harbor builds the per-task Dockerfile instead."""
    return re.sub(r"(?m)^docker_image\s*=.*\n", "", task_toml)


def leaked_paths(environment_files: Iterable[str]) -> list[str]:
    """Return environment paths that would expose answers or hidden tests."""
    return sorted(
        path
        for path in environment_files
        if FORBIDDEN_IN_ENVIRONMENT & set(PurePosixPath(path).parts[:1])
    )


def relative_files(root: Path) -> list[str]:
    """List files under root as POSIX paths relative to it."""
    return [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]


def convert_task(source: Path, target: Path, base_image: str = BASE_IMAGE) -> None:
    """Write one task in Harbor layout, refusing to leak answers into the sandbox.

    Raises:
        ValueError: If the generated environment would contain answer files.
    """
    shutil.rmtree(target, ignore_errors=True)
    environment = target / "environment"
    shutil.copytree(source / "environment", environment)
    (environment / "Dockerfile").write_text(environment_dockerfile(base_image))
    toml_text = (source / "task.toml").read_text(encoding="utf-8")
    (target / "task.toml").write_text(without_prebuilt_image(toml_text))
    for entry in COPIED_ENTRIES:
        copy_entry(source / entry, target / entry)
    leaks = leaked_paths(relative_files(environment))
    if leaks:
        raise ValueError(f"{source.name}: answer files in environment: {leaks}")


def copy_entry(source: Path, target: Path) -> None:
    """Copy a file or directory if it exists."""
    if source.is_dir():
        shutil.copytree(source, target)
    elif source.is_file():
        shutil.copy2(source, target)
