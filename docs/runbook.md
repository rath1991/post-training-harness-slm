# Runbook: deploy steps (each needs approval)

Nothing below has been run yet. Each step changes the machine or uses the GPU.

| # | Where | Command | Changes |
|---|---|---|---|
| 1 | Windows PowerShell (admin) | `wsl --install -d Ubuntu-24.04` | Installs an Ubuntu WSL distro. Today only `docker-desktop` exists. |
| 2 | Docker Desktop | Settings → Resources → WSL integration → enable Ubuntu-24.04 | Lets the Harbor sandboxes run from WSL |
| 3 | WSL, repo root | `bash scripts/setup_wsl.sh` | uv, `.venv` (Python 3.12), OpenEnv v0.7.0 source, dataset download (~25 MB), `harbor-swe-base:2` image (~1.8 GB), `data/harbor/*` splits, and the version lock in `env/requirements.lock` |
| 4 | WSL, terminal 1 | `bash scripts/serve_harbor.sh` | Harbor server on :8200, capture proxy on :8300 |
| 5 | WSL, terminal 2 | `bash scripts/run_day1.sh` | Stage 0 single task, then Strategy A, then Strategy B, then the comparison table |
| 6 | optional | `scripts/probe_stock_trainer.py` (see its docstring) | Documents whether stock TRL runs on one GPU |

Day 2: set `strategy` in `configs/grpo_full.toml` to the Day 1 winner. Then:

```bash
harness-rl eval --config configs/eval.toml --out runs/eval/base.json
harness-rl grpo --config configs/grpo_full.toml
harness-rl eval --config configs/eval.toml --adapter runs/grpo-full/adapters/policy-v50 --out runs/eval/trained.json
harness-rl compare runs/eval/base.json runs/eval/trained.json
```

## Assumptions that Stage 0 checks first

These came from the docs and source but haven't been run. If one fails, I'll document it as a blocker rather than work around it silently.

1. **Pi inside an offline sandbox.** Tasks set `allow_internet = false`, and the image ships Node 18. If Harbor installs Pi at sandbox start via npm, or Pi needs Node 20 or newer, the agent can't start. Fallback: bake Pi into a derived base image.
2. The Harbor split name equals the `--dataset` string (`data/harbor/smoke`).
3. Harbor's docker sandbox builds the per-task `environment/Dockerfile`.
4. `openenv harbor rollout --out X.json` writes the trace as `X.training_trace.json`.
5. vLLM accepts `prompt_logprobs: 0` and lists runtime-loaded adapters in `/v1/models`.
6. Strategy A's KV budget (about 1.5 GB at `gpu_memory_utilization = 0.62`) fits one 12k-token agent trace. If it doesn't, lower `max_model_len`, or use the 2B fallback the scope allows.
