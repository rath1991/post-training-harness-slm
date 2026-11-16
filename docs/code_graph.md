# Code graph: where to start reading

`src/harness_rl/` is split into a **pure core**, which you can test without a GPU, and a **thin I/O shell** (processes, HTTP, CUDA). Read it in this order:

| # | File | What it answers | Pure? |
|---|---|---|---|
| 1 | `rollout/traces.py` | How a captured Pi model call becomes trainable tokens. Which tokens are masked. | yes |
| 2 | `rollout/rewards.py`, `rollout/batching.py` | Verifier reward → group advantage → training rows; the sync probe | yes |
| 3 | `training/grpo_loss.py`, `training/logprobs.py` | The GRPO surrogate (beta=0) and chunked log-probs over the 151k vocab | torch only |
| 4 | `training/step.py`, `training/qlora.py` | One optimizer step on NF4 + LoRA, and the evidence that weights moved | GPU |
| 5 | `memory/strategies.py` | **Strategy A (sleep) vs. B (sequential)**: the Day 1 comparison | GPU + processes |
| 6 | `pipeline.py` | The loop that ties 1–5 together | orchestration |
| 7 | `cli.py` | Entry points | shell |

Then `tasks/` (dataset → splits → Harbor layout), `evaluation.py`, `reports.py`, `distill/` (Experiment B).

## Module dependencies

```mermaid
flowchart TD
    cli[cli.py] --> prep[tasks/*]
    cli --> pipeline[pipeline.py]
    cli --> evaluation[evaluation.py]
    cli --> reports[reports.py]

    pipeline --> strategies[memory/strategies.py]
    pipeline --> collect[rollout/collect.py]
    pipeline --> batching[rollout/batching.py]
    pipeline --> step[training/step.py]
    pipeline --> monitor[memory/monitor.py]

    strategies --> vllm[inference/vllm_server.py]
    strategies --> qlora[training/qlora.py]

    batching --> rewards[rollout/rewards.py]
    batching --> traces[rollout/traces.py]
    collect --> openenv[(OpenEnv HarborSessionFactory)]

    step --> loss[training/grpo_loss.py]
    step --> logprobs[training/logprobs.py]
    step --> qlora

    distill[distill/step.py] --> dlosses[distill/losses.py]
    distill --> context[distill/teacher_context.py]
    distill --> logprobs
```

## One GRPO iteration (`pipeline.iteration`)

```mermaid
sequenceDiagram
    participant P as pipeline
    participant R as runtime (A or B)
    participant V as vLLM
    participant H as Harbor + Pi (black box)
    participant T as QLoRA trainer
    P->>R: to_inference(policy-vN)
    R->>V: A: wake + load adapter / B: start process
    P->>H: N attempts per task (collect_groups)
    H->>V: model calls via capture proxy
    H-->>P: TrainingTrace + verifier reward
    P->>V: prompt_logprobs(probe) → sync check
    P->>R: to_training(vN)
    R->>V: A: sleep(level 1) / B: stop process
    P->>T: train_step(rows) → save policy-vN+1
    P->>R: release_training()
```

## What is reused and what is custom

| Reused unchanged | Custom (allowed by scope s.10) |
|---|---|
| Pi harness, Harbor tasks and verifier, Docker sandbox | GPU hand-over: `memory/strategies.py` |
| OpenEnv capture proxy, `TrainingTrace`, `HarborSessionFactory` | Phase-synchronous scheduler: `pipeline.py` |
| vLLM runtime LoRA loading (`policy-vN` naming, as in TRL) | Privileged teacher scoring: `distill/` |
| TRL's loss rule (`score_centering=False` branch) and advantage rule | Evaluation and reports |
