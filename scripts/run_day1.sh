#!/usr/bin/env bash
# Day 1 in order. Each stage must pass before the next is worth running.
#   0  one Pi task end-to-end: rollout + verifier + valid token trace (no training)
#   1  Strategy A smoke (sleep mode)
#   2  Strategy B smoke (sequential switching), same tasks/hyper-parameters
# Harbor must already be running (scripts/serve_harbor.sh in another terminal).

set -euo pipefail
export PYTHONPATH="$PWD/vendor/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"

stage0_single_task() {
    # Scope s.10: prove one task runs in Pi and yields a trace before training.
    VLLM_SERVER_DEV_MODE=1 .venv/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 \
        --served-model-name student-base --max-model-len 12288 \
        --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes \
        --logprobs-mode processed_logprobs --return-tokens-as-token-ids \
        > runs/stage0-vllm.log 2>&1 &
    local vllm_pid=$!
    trap 'kill $vllm_pid 2>/dev/null || true' RETURN
    until curl -sf localhost:8000/health >/dev/null; do sleep 5; done
    .venv/bin/openenv harbor rollout --llm-url http://localhost:8000/v1 \
        --dataset data/harbor/smoke --task-index 0 --harness pi --sandbox docker \
        --out runs/stage0-rollout.json
    # The rollout command writes the TrainingTrace next to --out.
    .venv/bin/harness-rl inspect-trace runs/stage0-rollout.training_trace.json
}

mkdir -p runs
stage0_single_task
.venv/bin/harness-rl grpo --config configs/smoke_sleep.toml
.venv/bin/harness-rl grpo --config configs/smoke_sequential.toml
.venv/bin/harness-rl compare-strategies runs/smoke-sleep runs/smoke-sequential
