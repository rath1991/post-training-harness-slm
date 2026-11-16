#!/usr/bin/env bash
# Start the OpenEnv Harbor server (task API + capture proxy) with Docker sandboxes.
# Each --dataset entry becomes a split named by the same string, which is what
# [harbor].split in configs/*.toml refers to.

set -euo pipefail

export PYTHONPATH="$PWD/vendor/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"
# Must be at least max_concurrent_rollouts + 1.
export MAX_CONCURRENT_ENVS="${MAX_CONCURRENT_ENVS:-5}"

exec .venv/bin/openenv harbor serve \
    --dataset data/harbor/smoke,data/harbor/train,data/harbor/eval,data/harbor/ood \
    --port 8200 --capture-port 8300
