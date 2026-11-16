#!/usr/bin/env python3
"""PostToolUse hook: run the style gate on every Python file Claude edits.

Runs ``stylecheck.sh --fix`` (isort + black formatting, then pycodestyle, pydocstyle
and ruff checks). Exit 2 sends the failures back to Claude as feedback; exit 0 means
the file passed, or the hook could not run, which must never block work.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

STYLECHECK = Path(__file__).resolve().parent / "stylecheck.sh"
# Search PATH explicitly: on Windows a bare "bash" resolves to System32's WSL
# launcher before Git Bash, and that launcher cannot run this script.
BASH = shutil.which("bash") or "bash"
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
EXIT_CHECKS_FAILED = 1
SKIP_MARKER = "# stylecheck: skip"


def edited_file(payload: dict) -> Path | None:
    """Return the edited Python file from the hook payload, if there is one."""
    raw = payload.get("tool_input", {}).get("file_path")
    if not raw:
        return None
    path = Path(raw)
    return path if path.suffix in PYTHON_SUFFIXES and path.is_file() else None


def opted_out(file: Path) -> bool:
    """Return True if the file's first line is the skip marker (e.g. test fixtures)."""
    with file.open(errors="ignore") as handle:
        return handle.readline().strip() == SKIP_MARKER


def format_feedback(file: Path, report: str) -> str:
    """Build the message Claude sees when the style gate fails."""
    return (
        f"Style gate failed for {file} (isort/black were auto-applied; "
        "the file on disk may have changed - re-read before editing):\n"
        f"{report.strip()}\n"
        "Fix every finding in code you wrote or changed, then the gate reruns on "
        "your next edit. Run manually: stylecheck.sh --fix <path>"
    )


def main() -> int:
    """Run the style gate for the edited file and report failures."""
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    file = edited_file(payload)
    if file is None or opted_out(file):
        return 0
    result = subprocess.run(
        [BASH, str(STYLECHECK), "--fix", str(file)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    # 0 = passed; anything else = the gate itself could not run.
    if result.returncode != EXIT_CHECKS_FAILED:
        return 0
    print(format_feedback(file, result.stdout), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
