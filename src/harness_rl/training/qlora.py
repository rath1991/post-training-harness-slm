"""Load, save and release the NF4 QLoRA policy and its optimizer state."""

import gc
from dataclasses import dataclass

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from harness_rl.adapters import AdapterRef
from harness_rl.config import ModelConfig

OPTIMIZER_FILE = "optimizer.pt"


@dataclass(frozen=True, slots=True)
class PolicyTrainer:
    """The trainable policy and its optimizer, always handled together."""

    model: PeftModel
    optimizer: torch.optim.Optimizer


def nf4_config() -> BitsAndBytesConfig:
    """Return the standard QLoRA 4-bit quantisation settings."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_quantized_base(model: ModelConfig) -> torch.nn.Module:
    """Load the frozen NF4 base model on GPU 0 with gradient checkpointing.

    ``prepare_model_for_kbit_training`` is skipped on purpose: it upcasts every
    non-quantised weight (including the 151k x 2560 embedding) to fp32, which
    costs ~1.5 GB we do not have on a 16 GB card.
    """
    base = AutoModelForCausalLM.from_pretrained(
        model.base_model,
        quantization_config=nf4_config(),
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    base.gradient_checkpointing_enable({"use_reentrant": False})
    base.enable_input_require_grads()
    base.config.use_cache = False
    return base


def new_policy(model: ModelConfig, learning_rate: float) -> PolicyTrainer:
    """Create a fresh LoRA policy; zero-initialised B means it equals the base."""
    lora = LoraConfig(
        r=model.lora_rank,
        lora_alpha=model.lora_alpha,
        target_modules=list(model.lora_targets),
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(load_quantized_base(model), lora)
    return PolicyTrainer(peft_model, make_optimizer(peft_model, learning_rate))


def restore_policy(
    model: ModelConfig, adapter: AdapterRef, learning_rate: float
) -> PolicyTrainer:
    """Reload a saved adapter and its optimizer moments for further training."""
    peft_model = PeftModel.from_pretrained(
        load_quantized_base(model), adapter.path, is_trainable=True
    )
    optimizer = make_optimizer(peft_model, learning_rate)
    state_file = adapter.path / OPTIMIZER_FILE
    if state_file.exists():
        optimizer.load_state_dict(torch.load(state_file, weights_only=True))
    return PolicyTrainer(peft_model, optimizer)


def make_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.AdamW:
    """Return AdamW over the LoRA parameters only."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)


def save_policy(trainer: PolicyTrainer, adapter: AdapterRef) -> AdapterRef:
    """Write the adapter (for vLLM) and optimizer state (for resuming)."""
    trainer.model.save_pretrained(adapter.path)
    torch.save(trainer.optimizer.state_dict(), adapter.path / OPTIMIZER_FILE)
    return adapter


def lora_norm(model: torch.nn.Module) -> float:
    """Return the L2 norm of all LoRA weights; it changes iff the policy did."""
    squares = [
        p.detach().float().pow(2).sum()
        for name, p in model.named_parameters()
        if "lora_" in name
    ]
    return float(torch.stack(squares).sum().sqrt())


def free_gpu_memory() -> None:
    """Return cached CUDA blocks to the driver after dropping model references."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
