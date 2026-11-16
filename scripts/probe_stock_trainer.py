r"""Probe: does TRL's stock HarnessAsyncGRPOTrainer run on ONE 16 GB GPU?

TRL documents this trainer for 2 GPUs (vLLM on one, trainer on the other) and
has no sleep mode or QLoRA path. This probe runs it unmodified, both processes
on GPU 0, with LoRA sync (adapter files, no NCCL) and a small bf16 model, so
the blocker (if any) is documented rather than assumed (scope s.6).

Start vLLM first, sharing the GPU:

    VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \\
    vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.45 \\
        --enable-auto-tool-choice --tool-call-parser hermes \\
        --logprobs-mode processed_logprobs --return-tokens-as-token-ids \\
        --enable-lora --max-lora-rank 16 --max-loras 6 \\
        --weight-transfer-config '{"backend":"nccl"}'

Then: ``python scripts/probe_stock_trainer.py --split data/harbor/smoke``
"""

import argparse
from functools import partial

from datasets import Dataset
from harbor_env.harness import HarborSessionFactory
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import AsyncGRPOConfig
from trl.experimental.async_grpo.openenv_harness import HarnessAsyncGRPOTrainer


def parse_args() -> argparse.Namespace:
    """Read the probe's few knobs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--server", default="http://127.0.0.1:8200")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    return parser.parse_args()


def task_dataset(factory: partial) -> Dataset:
    """Build the prompt dataset from the Harbor server's task list."""
    metadata_factory = factory()
    try:
        return Dataset.from_list(metadata_factory.prompt_rows())
    finally:
        # The pinned OpenEnv factory has no public close method (TRL example).
        if metadata_factory._env is not None:
            metadata_factory._env.close()


def main() -> None:
    """Run two optimizer steps with the stock trainer and report the outcome."""
    args = parse_args()
    factory = partial(
        HarborSessionFactory,
        args.server,
        split=args.split,
        harness="pi",
        sandbox="docker",
        llm_url=args.vllm_url,
        model=args.model,
        agent_step_limit=30,
    )
    config = AsyncGRPOConfig(
        output_dir="runs/stock-probe",
        num_generations=4,
        max_concurrent_rollouts=4,
        max_steps=2,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        vllm_server_base_url=args.vllm_url,
        chat_template_kwargs={"enable_thinking": False},
        activation_checkpointing=True,
        activation_checkpointing_kwargs={"use_reentrant": False},
        dtype="bfloat16",
        bf16=True,
        report_to="none",
    )
    trainer = HarnessAsyncGRPOTrainer(
        model=args.model,
        harness_session_factory=factory,
        args=config,
        train_dataset=task_dataset(factory),
        processing_class=AutoTokenizer.from_pretrained(args.model),
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),
    )
    trainer.train()
    print("stock trainer completed 2 steps on a single GPU")


if __name__ == "__main__":
    main()
