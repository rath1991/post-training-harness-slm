# Project: Harness-Based Post-Training of Small Coding Models

**Phase 1: RTX 5060 Ti pilot | Phase 2: RTX PRO 6000 scaling**

### Objective

Build and validate an end-to-end system for improving coding language models through post-training inside real agent harnesses.

The eventual target is 27–30B parameter models trained with QLoRA on RTX PRO 6000 Blackwell hardware. Before scaling, we will validate the architecture, training algorithms, memory management and evaluation methodology using a 4B model on an RTX 5060 Ti with 16 GB VRAM and 64 GB system RAM.

The immediate goal is **a reproducible training system, not a state-of-the-art benchmark score**.

### 1. Technical stack

- **Model:** Qwen3-4B-Instruct-2507 initially; use a 2B model if memory constraints block progress.
- **Training:** Hugging Face TRL, PyTorch, PEFT/QLoRA.
- **Inference:** vLLM, initially BF16, with SGLang as a later alternative.
- **Agent harness:** Pi, representing the kind of tool-driven workflow we eventually want inside VS Code.
- **Task orchestration:** Harbor.
- **Trace capture:** OpenEnv Harbor integration and capture proxy.
- **Sandbox:** Local Docker, subject to compatibility validation.
- **Development:** Windows 11 machine using WSL2 Ubuntu and GPU-enabled Linux tooling.

Use pinned package versions and record the working environment. Do not assume that examples from different TRL versions have identical APIs.

### 2. Experiment A — Black-box GRPO

The first experiment trains a model based on how it performs within Pi.

**Execution architecture**

```text
TRL HarnessAsyncGRPOTrainer
        |
        v
HarborSessionFactory / OpenEnv server
        |
        v
Harbor task + isolated sandbox
        |
        v
Pi harness
        |
        |---- Prompt + tool schemas ------|
        |                                 v
        |                         OpenEnv capture proxy
        |                                 |
        |                                 v
        |                            vLLM model
        |                                 |
        |<------ Generated response ------|
        |
        |---- Execute tools in sandbox
        |---- Observe file/test results
        |---- Update context
        |---- Repeat model calls
        |
        v
Harness stops / task times out
        |
        v
Harbor verifier → task reward
        |
        v
OpenEnv TrainingTrace
        |
        v
TRL: advantages + GRPO loss + gradients
        |
        v
Updated QLoRA adapter
        |
        v
Synchronize policy → next rollouts
```

**Implementation requirements**

1. Run Pi against the locally served student model.
2. Start a Harbor task in a fresh sandbox.
3. Verify that Pi can inspect code, edit files and run tests.
4. Capture the actual model prompt tokens, generated token IDs, log-probabilities and loss masks.
5. Confirm that non-model-generated context is excluded from the policy loss.
6. Execute multiple independent attempts at the same task.
7. Compute group-relative rewards and perform a GRPO optimizer update.
8. Verify that the updated adapter is used during subsequent inference.

Start with correctness-based verifier rewards. Introduce tool-efficiency rewards only after the basic training pipeline works.

Start with `beta=0` to avoid unnecessary reference-model overhead. Test KL regularization later if policy drift becomes significant.

### 3. Experiment B — Privileged On-Policy Self-Distillation

The second experiment tests whether privileged token-level supervision can improve coding-agent behaviour more efficiently than outcome-level GRPO.

Use the same student model, tasks and evaluation process.

**Teacher–student approach**

- Student and teacher initially share the same model checkpoint and tokenizer.
- Student generates a trajectory through Pi.
- OpenEnv captures the actual model-call contexts and sampled tokens.
- A privileged teacher receives the corresponding context plus additional information, such as the verified solution, reference patch or an oracle trajectory.
- The teacher scores the student's sampled-token prefixes.
- A distillation objective updates the student.

```text
Coding task
    |
    v
Student → Pi harness → student trajectory
    |
    v
Captured contexts + generated tokens
    |
    +---------------------------+
    |                           |
    v                           v
Student scoring          Privileged teacher
Original context         Context + oracle
    |                           |
    v                           v
Student probabilities    Teacher probabilities
    |                           |
    +-------------+-------------+
                  |
                  v
        Distillation loss
                  |
                  v
        Student QLoRA update
```

**Loss experiments**

Start with one clear implementation:

- Full-distribution KL/JSD when practical, using chunked teacher/student scoring.
- Sampled-token log-probability objective as a separate alternative if full distribution matching is too expensive.

Do not treat a sampled-token log-probability difference as automatically equivalent to a full KL divergence.

**Implementation considerations**

The standard TRL `DistillationTrainer` is a useful reference, but privileged conditioning over externally captured Pi trajectories will require a custom integration.

The initial privileged teacher should use frozen weights during each scoring/update operation. Experiment with teacher synchronization to the current student separately.

A frontier model may generate reference patches or verified trajectories beforehand. That is distinct from the privileged teacher used to score student tokens.

The reference solutions must never be exposed to the student harness or sandbox during task execution.

### 4. GPU memory-management experiments

Compare two approaches on the 16 GB GPU.

**Strategy A — vLLM Sleep Mode**

1. Generate model rollouts using vLLM.
2. Put vLLM into sleep mode and release unnecessary inference allocations.
3. Run QLoRA optimization.
4. Synchronize updated adapter weights.
5. Wake vLLM and continue.

Important: This does not automatically remove the training model from GPU memory during generation.

**Strategy B — Full sequential model switching**

1. Load inference model and generate a batch of rollouts.
2. Capture and retain the necessary training data.
3. Release/offload inference-model GPU allocations.
4. Load or restore the QLoRA training model and optimizer state.
5. Compute the loss and update model weights.
6. Release/offload training allocations.
7. Restore the inference model with the updated adapter.
8. Begin the next batch.

The two strategies must use equivalent tasks, hyperparameters and model checkpoints.

**Measure:**

- Peak and allocated GPU memory
- CPU RAM utilization
- Rollout generation time
- Model unloading/reloading time
- Optimizer-step time
- End-to-end training throughput
- Stability, model synchronization and inference/training probability mismatch

This is primarily a comparison of **resource efficiency and correctness**, not two different learning algorithms.

The existing black-box asynchronous trainer may need custom scheduling to implement complete sequential switching. Do not assume standard GRPO sleep-mode configuration alone provides it.

### 5. Dataset

**Primary dataset:** [smol-software-harbor](https://huggingface.co/datasets/efederici/smol-software-harbor)

It contains 1,002 small coding tasks across 576 mini-codebases, with verifiers and repository-level holdout splits.

The tasks include bug fixing, finding relevant files, implementing missing functions, adding validation, writing tests and fixing edge cases.

Use these targets once infrastructure is stable:

| Partition | Planned tasks |
|---|---:|
| Training | 300–400 |
| Held-out evaluation | 100 |
| Additional out-of-distribution evaluation | 50 |

Choose training and evaluation tasks from different codebases. Stratify by language, task type, difficulty and number of expected tool calls.

The dataset's native OOD split contains only 27 tasks. To reach 50 OOD tasks, supplement it with a separately verified external task collection; do not relabel ordinary held-out tasks as OOD.

Validate that the dataset works with the selected OpenEnv/Harbor backend. Its published palingenesis workflow does not by itself guarantee drop-in compatibility with every Harbor training configuration.

### 6. Four-day pilot plan

### 6. GRPO Execution Plan — Day 1 and Day 2

**Immediate objective:** Validate black-box GRPO training with Pi + Harbor + OpenEnv + TRL on the RTX 5060 Ti (16 GB VRAM), compare two GPU memory-management strategies, and select one for the full experiment.

#### Day 1 — GRPO Smoke Tests (November 16)

**Strategy A: vLLM Sleep Mode**

- Configure TRL, Harbor, OpenEnv, Pi and a 4B student model with QLoRA.
- Generate coding-agent rollouts through Pi using vLLM.
- Use vLLM sleep mode during GRPO optimization.
- Confirm verifier rewards, valid captured tokens, loss masks and successful LoRA weight updates.
- Verify that the updated adapter is loaded for subsequent rollouts.

**Strategy B: Full Sequential Model Switching**

- Generate rollouts using vLLM.
- Fully offload or unload the inference model from GPU memory.
- Load the QLoRA training model and perform GRPO optimization.
- Preserve optimizer state, unload training allocations and reload updated weights into vLLM.
- Confirm that the next rollout uses the updated policy.

**Compare both strategies on the same 5–10 smoke-test tasks**, with identical model initialization and GRPO configuration.

Evaluate peak VRAM, system RAM, rollout latency, model-switching overhead, optimizer-step time, total throughput and stability.

**Day 1 deliverable:** Two tested memory-management approaches, documented errors and limitations, at least one successful end-to-end GRPO update, and a decision on which strategy to use.

If either approach proves incompatible with the current TRL black-box API, document the blocker rather than silently replacing black-box GRPO with a white-box training loop.

#### Day 2 — Full GRPO Training and Evaluation (November 17)

Use the selected strategy to run the main experiment on `smol-software-harbor`.

| Partition | Target |
|---|---:|
| Training | 400 tasks |
| Held-out evaluation | 100 tasks |
| Additional OOD evaluation | 50 tasks, if a verified external set is available |

- Run the Base model evaluation before training.
- Generate multiple student rollouts per training task and use Harbor verifier rewards for GRPO.
- Train the 4B student using QLoRA with the selected memory-management strategy.
- Save checkpoints and log rewards, KL (if enabled), policy loss, gradient norms, GPU memory and throughput.
- Evaluate the trained model using the same harness, evaluation tasks and execution limits as Base.
- Compare task success, tool-call efficiency, failed/repeated actions and recovery from execution errors.

**Day 2 deliverable:** A reproducible GRPO training run, updated LoRA adapter, Base-versus-trained evaluation and a summary of observed improvements and failures.

**Execution constraint:** The target is 400 training tasks, not necessarily 400 successful trajectories. Verify effective rollout throughput and the proportion of nonzero reward-variance groups during the smoke test. If a full 400-task run cannot finish within the available compute window, complete the largest feasible subset, preserve checkpoints and report the exact coverage rather than treating an incomplete run as a full evaluation.

**Success criteria:** A stable black-box GRPO training loop operating within 16 GB VRAM, verified model-weight updates, successful post-update inference and measurable evaluation results suitable for deciding how to scale the architecture to the RTX PRO 6000.

### 7. Success criteria

The pilot succeeds if we demonstrate:

1. Pi can operate using the local 4B model within a Harbor-managed sandbox.
2. OpenEnv captures accurate, usable model-token trajectories.
3. Harbor's verifier produces reliable task rewards.
4. TRL successfully executes a QLoRA GRPO weight update.
5. The updated model can be synchronized into the inference engine and changes subsequent outputs.
6. A privileged teacher can score student-generated token prefixes and produce a usable distillation gradient.
7. At least one memory-management strategy runs within the 16 GB GPU limit.
8. Base and trained models can be evaluated on the same held-out tasks.

Performance improvements are desirable but are not required to establish infrastructure feasibility.

### 8. Risks and decisions

- **GPU memory:** vLLM and the training process may retain independent CUDA allocations.
- **Training mismatch:** BF16 inference and NF4 QLoRA training are not numerically identical; captured behaviour log-probabilities and importance corrections must be validated.
- **Package compatibility:** TRL's black-box training and distillation APIs are evolving.
- **Long trajectories:** Context growth, KV cache usage and repeated tool calls can exhaust memory or slow experiments.
- **Reward sparsity:** A 4B model may fail every attempt at difficult tasks, producing no useful group-relative advantage.
- **Distillation:** Privileged teacher scoring over real harness traces is not a turnkey feature of the standard distillation trainer.
- **Dataset portability:** Verify toolchains, Docker images and task execution before large-scale training.

### 9. Expected project outputs

- Working black-box GRPO training pipeline
- Working privileged on-policy distillation prototype
- QLoRA adapters and baseline checkpoints
- Local GPU memory/switching benchmark
- Reproducible dataset splits and evaluation harness
- Task-level performance report
- Documented failures and fixes
- Scaling recommendations for a 30B model on RTX PRO 6000 Blackwell

### 10. Implementation instructions

Start from the latest version-compatible Hugging Face TRL and OpenEnv examples. Reuse their Harbor integration and token-capture machinery rather than rebuilding it.

Implement the work incrementally. Before attempting model training, prove that a single coding task executes inside Pi, produces a verifier result, and yields a valid token-level training trace.

Do not spend the pilot rewriting TRL or creating a custom agent harness. Custom development should focus on memory orchestration, privileged teacher scoring and reproducible evaluation.

**References**

- [TRL OpenEnv integration](https://huggingface.co/docs/trl/openenv)
- [OpenEnv black-box Harbor tutorial](https://huggingface.co/docs/openenv/main/en/tutorials/harbor-harness)
- [TRL asynchronous GRPO](https://huggingface.co/docs/trl/async_grpo_trainer)
- [TRL DistillationTrainer](https://huggingface.co/docs/trl/distillation_trainer)
- [TRL asynchronous distillation](https://huggingface.co/docs/trl/main/async_distillation_trainer)
- [TRL QLoRA integration](https://huggingface.co/docs/trl/peft_integration)
- [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
- [Small software engineering tasks](https://huggingface.co/datasets/efederici/smol-software-harbor)
- [TRL agent-training examples](https://github.com/huggingface/trl/tree/main/examples)