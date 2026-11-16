#!/usr/bin/env bash
# Install the functional-python harness into ~/.claude (idempotent, non-destructive).
#
# Usage: ./install.sh [--skip-tools]

set -euo pipefail

readonly REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SRC="$REPO_DIR/claude-home"
readonly CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
readonly BACKUP_DIR="$CLAUDE_DIR/backups/harness-$(date +%Y%m%d-%H%M%S)"
readonly TOOLS=(black isort pycodestyle pydocstyle ruff)
readonly MARK_START="<!-- functional-python-harness:start -->"
readonly MARK_END="<!-- functional-python-harness:end -->"

log() { printf '==> %s\n' "$*"; }

find_python() {
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null \
            && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
            echo "$candidate"
            return
        fi
    done
    echo "Python 3.10+ not found" >&2
    exit 69
}

backup() {
    local path
    mkdir -p "$BACKUP_DIR"
    for path in CLAUDE.md settings.json skills/functional-python hooks/python_style; do
        if [[ -e "$CLAUDE_DIR/$path" ]]; then
            mkdir -p "$BACKUP_DIR/$(dirname "$path")"
            cp -R "$CLAUDE_DIR/$path" "$BACKUP_DIR/$path"
        fi
    done
    log "Backed up existing files to $BACKUP_DIR"
}

copy_files() {
    mkdir -p "$CLAUDE_DIR/skills" "$CLAUDE_DIR/hooks"
    rm -rf "$CLAUDE_DIR/skills/functional-python" "$CLAUDE_DIR/hooks/python_style"
    cp -R "$SRC/skills/functional-python" "$CLAUDE_DIR/skills/"
    cp -R "$SRC/hooks/python_style" "$CLAUDE_DIR/hooks/"
    chmod +x "$CLAUDE_DIR/hooks/python_style/stylecheck.sh" \
        "$CLAUDE_DIR/hooks/python_style/check.py"
    log "Copied skill and hooks"
}

# Replace the marked harness block in CLAUDE.md, or append it; other content is kept.
merge_claude_md() {
    local python="$1"
    "$python" - "$CLAUDE_DIR/CLAUDE.md" "$SRC/CLAUDE.md" "$MARK_START" "$MARK_END" <<'EOF'
import re
import sys
from pathlib import Path

target, source, start, end = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
block = f"{start}\n{source.read_text().strip()}\n{end}"
current = target.read_text() if target.exists() else ""
pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
if pattern.search(current):
    merged = pattern.sub(lambda _: block, current)
elif current.strip():
    merged = f"{current.rstrip()}\n\n{block}\n"
else:
    merged = f"{block}\n"
target.write_text(merged)
EOF
    log "Merged instructions into $CLAUDE_DIR/CLAUDE.md"
}

# Add the PostToolUse hook unless an equivalent one already exists; other settings kept.
merge_settings() {
    local python="$1"
    "$python" - "$CLAUDE_DIR/settings.json" "$SRC/settings.hooks.json" "$python" <<'EOF'
import json
import sys
from pathlib import Path

target, source, python = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
settings = json.loads(target.read_text()) if target.exists() else {}
entry = json.loads(source.read_text())["hooks"]["PostToolUse"][0]
for hook in entry["hooks"]:
    hook["command"] = hook["command"].replace("python3", python, 1)
post = settings.setdefault("hooks", {}).setdefault("PostToolUse", [])
installed = any(
    "python_style/check.py" in hook.get("command", "")
    for group in post
    for hook in group.get("hooks", [])
)
if not installed:
    post.append(entry)
target.write_text(json.dumps(settings, indent=2) + "\n")
print("hook added" if not installed else "hook already present")
EOF
    log "Merged hook into $CLAUDE_DIR/settings.json"
}

install_tools() {
    local tool
    for tool in "${TOOLS[@]}"; do
        if command -v uv >/dev/null; then
            uv tool install --quiet "$tool" 2>/dev/null || uv tool upgrade --quiet "$tool"
        elif command -v pipx >/dev/null; then
            pipx install "$tool" >/dev/null 2>&1 || true
        else
            "$1" -m pip install --user --quiet "$tool"
        fi
    done
    log "Tools installed: ${TOOLS[*]}"
}

link_stylecheck() {
    mkdir -p "$HOME/.local/bin"
    ln -sf "$CLAUDE_DIR/hooks/python_style/stylecheck.sh" "$HOME/.local/bin/stylecheck.sh"
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) ;;
        *) log "Add ~/.local/bin to PATH to run stylecheck.sh directly" ;;
    esac
}

expect_exit() {
    local expected="$1" label="$2"; shift 2
    local actual=0
    "$@" >/dev/null 2>&1 || actual=$?
    if [[ "$actual" == "$expected" ]]; then
        log "PASS  $label (exit $actual)"
    else
        log "FAIL  $label (expected exit $expected, got $actual)"
        return 1
    fi
}

verify() {
    local python="$1" gate="$CLAUDE_DIR/hooks/python_style/stylecheck.sh"
    local scratch
    scratch="$(mktemp -d)"
    # Drop the "# stylecheck: skip" marker so the hook actually checks the copy.
    tail -n +2 "$REPO_DIR/examples/bad.py" >"$scratch/bad.py"
    expect_exit 0 "good.py passes the gate" bash "$gate" "$REPO_DIR/examples/good.py"
    expect_exit 1 "bad.py fails the gate" bash "$gate" "$REPO_DIR/examples/bad.py"
    expect_exit 2 "hook reports bad.py" bash -c \
        "printf '{\"tool_input\":{\"file_path\":\"%s\"}}' '$scratch/bad.py' \
        | '$python' '$CLAUDE_DIR/hooks/python_style/check.py'"
    rm -rf "$scratch"
}

main() {
    local python
    python="$(find_python)"
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    backup
    copy_files
    merge_claude_md "$python"
    merge_settings "$python"
    [[ "${1:-}" == "--skip-tools" ]] || install_tools "$python"
    link_stylecheck
    verify "$python"
    log "Done. Hook is live now; skill and CLAUDE.md load in the next Claude Code session."
}

main "$@"
