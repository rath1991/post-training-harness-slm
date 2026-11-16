#!/usr/bin/env bash
# Python style gate: isort, black, pycodestyle, pydocstyle, ruff (functional-style rules).
#
# Usage: stylecheck.sh [--fix] PATH...
#   --fix   apply isort and black formatting first, then run every check
#
# Exits 0 only when every check passes. A tool's own project config (pyproject.toml,
# setup.cfg, tox.ini, ...) wins; otherwise these black-compatible defaults apply.

set -uo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
readonly RUFF_CONFIG="$SCRIPT_DIR/ruff.toml"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

usage() {
    echo "Usage: $(basename "$0") [--fix] PATH..." >&2
    exit 64
}

# Nearest ancestor that looks like a project root; falls back to the path's own directory.
project_root() {
    local dir
    dir="$(cd "$(dirname "$1")" && pwd)"
    [[ -d "$1" ]] && dir="$(cd "$1" && pwd)"
    local start="$dir"
    while [[ "$dir" != "/" ]]; do
        for marker in pyproject.toml setup.cfg tox.ini .git; do
            [[ -e "$dir/$marker" ]] && { echo "$dir"; return; }
        done
        dir="$(dirname "$dir")"
    done
    echo "$start"
}

# has_config ROOT FILE:PATTERN... -> true if any FILE under ROOT contains PATTERN (or exists, for bare names).
has_config() {
    local root="$1"; shift
    local spec file pattern
    for spec in "$@"; do
        file="${spec%%:*}"
        pattern="${spec#*:}"
        if [[ "$spec" == "$file" ]]; then
            [[ -f "$root/$file" ]] && return 0
        else
            grep -qsF "$pattern" "$root/$file" && return 0
        fi
    done
    return 1
}

failures=()

# run_check NAME CMD... -> run quietly; print output only on failure.
run_check() {
    local name="$1"; shift
    local output
    if output="$("$@" 2>&1)"; then
        echo "PASS  $name"
    else
        echo "FAIL  $name"
        [[ -n "$output" ]] && sed 's/^/      /' <<<"$output"
        failures+=("$name")
    fi
}

require_tools() {
    local missing=()
    for tool in black isort pycodestyle pydocstyle ruff; do
        command -v "$tool" >/dev/null || missing+=("$tool")
    done
    if ((${#missing[@]})); then
        echo "Missing tools: ${missing[*]}  (install: uv tool install <name>)" >&2
        exit 69
    fi
}

main() {
    local fix=false
    [[ "${1:-}" == "--fix" ]] && { fix=true; shift; }
    (($#)) || usage
    require_tools

    local root
    root="$(project_root "$1")"

    local isort_opts=() pycodestyle_opts=() pydocstyle_opts=() ruff_opts=()
    has_config "$root" "pyproject.toml:[tool.isort]" ".isort.cfg" "setup.cfg:[isort]" "tox.ini:[isort]" \
        || isort_opts=(--profile black)
    has_config "$root" "setup.cfg:[pycodestyle]" "tox.ini:[pycodestyle]" \
        || pycodestyle_opts=(--max-line-length=88 --ignore=E203,E701,E704,W503)
    has_config "$root" ".pydocstyle" ".pydocstylerc" "pyproject.toml:[tool.pydocstyle]" \
        "setup.cfg:[pydocstyle]" "tox.ini:[pydocstyle]" \
        || pydocstyle_opts=(--convention=google --add-ignore=D105,D107)
    has_config "$root" "ruff.toml" ".ruff.toml" "pyproject.toml:[tool.ruff" \
        || ruff_opts=(--config "$RUFF_CONFIG")

    if $fix; then
        isort --quiet ${isort_opts[@]+"${isort_opts[@]}"} "$@" >/dev/null 2>&1
        black --quiet "$@" >/dev/null 2>&1
    fi

    run_check isort       isort --check-only --diff ${isort_opts[@]+"${isort_opts[@]}"} "$@"
    run_check black       black --check --quiet "$@"
    run_check pycodestyle pycodestyle ${pycodestyle_opts[@]+"${pycodestyle_opts[@]}"} "$@"
    run_check pydocstyle  pydocstyle ${pydocstyle_opts[@]+"${pydocstyle_opts[@]}"} "$@"
    run_check ruff        ruff check --no-cache --output-format concise ${ruff_opts[@]+"${ruff_opts[@]}"} "$@"

    if ((${#failures[@]})); then
        echo "Style check failed: ${failures[*]}"
        return 1
    fi
    echo "All style checks passed."
}

main "$@"
