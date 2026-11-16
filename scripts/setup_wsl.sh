#!/usr/bin/env bash
# One-time environment setup INSIDE WSL2 Ubuntu (run from the repo root).
# Prerequisites done on Windows first (see docs/runbook.md):
#   wsl --install -d Ubuntu-24.04      and   Docker Desktop -> WSL integration ON
#
# Idempotent: re-running skips finished steps. Records exact versions to env/.

set -euo pipefail

readonly OPENENV_TAG="v0.7.0"
readonly DATASET="efederici/smol-software-harbor"

log() { printf '==> %s\n' "$*"; }

check_prerequisites() {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        || { echo "GPU not visible in WSL" >&2; exit 1; }
    docker info >/dev/null 2>&1 \
        || { echo "Docker not reachable: enable Docker Desktop WSL integration" >&2; exit 1; }
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
}

install_python_env() {
    # The OpenEnv wheel does not package harbor_env.harness; it needs the source.
    [[ -d vendor/OpenEnv ]] || git clone --depth 1 --branch "$OPENENV_TAG" \
        https://github.com/openenv/OpenEnv.git vendor/OpenEnv
    [[ -d .venv ]] || uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -e ".[inference,dev]"
    mkdir -p env
    uv pip freeze --python .venv/bin/python > env/requirements.lock
    log "Pinned versions written to env/requirements.lock"
}

fetch_dataset_and_image() {
    [[ -f data/smol-software-harbor/manifest.jsonl ]] || .venv/bin/hf download \
        "$DATASET" --repo-type dataset --local-dir data/smol-software-harbor
    docker image inspect harbor-swe-base:2 >/dev/null 2>&1 || {
        docker build -t harbor-swe-base:2 data/smol-software-harbor/base_image
        docker tag harbor-swe-base:2 harbor-swe-base:1
    }
}

main() {
    check_prerequisites
    install_python_env
    fetch_dataset_and_image
    .venv/bin/harness-rl prepare-data --dataset-dir data/smol-software-harbor --out-dir data/harbor
    log "Done. Next: scripts/serve_harbor.sh, then scripts/run_day1.sh"
}

main "$@"
