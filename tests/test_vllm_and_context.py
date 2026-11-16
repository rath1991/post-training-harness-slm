from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_rl.adapters import adapter_ref
from harness_rl.config import ModelConfig, VllmConfig, load_run_config
from harness_rl.distill.teacher_context import distill_example
from harness_rl.inference.vllm_server import parse_prompt_logprobs, serve_command
from harness_rl.rollout.collect import count_tool_calls, turn_tokens
from harness_rl.rollout.traces import TurnTokens

ROOT = Path(__file__).resolve().parents[1]


def test_sleep_flag_only_for_sleep_strategy(tmp_path: Path) -> None:
    adapter = adapter_ref(tmp_path, 0)
    sleeping = serve_command(ModelConfig(), VllmConfig(), adapter, sleep_mode=True)
    sequential = serve_command(ModelConfig(), VllmConfig(), None, sleep_mode=False)
    assert "--enable-sleep-mode" in sleeping
    assert "--enable-sleep-mode" not in sequential
    assert f"policy-v0={adapter.path}" in sleeping
    for flag in ("--return-tokens-as-token-ids", "--enable-lora"):
        assert flag in sequential


@pytest.mark.parametrize("key", ["{t}", "token_id:{t}"])
def test_prompt_logprobs_parse_both_key_styles(key: str) -> None:
    entries = [None, {key.format(t=7): {"logprob": -1.5}, "3": {"logprob": -0.1}}]
    assert parse_prompt_logprobs(entries, [5, 7]) == (-1.5,)


def test_turn_mask_is_split_at_the_prompt_boundary() -> None:
    captured = SimpleNamespace(
        prompt_token_ids=[10, 11],
        completion_token_ids=[20, 21, 22],
        per_token_logps=[-0.1, -0.2, -0.3],
        loss_mask=[0, 0, 1, 0, 1],
    )
    assert turn_tokens(captured).completion_mask == (1, 0, 1)


def test_tool_calls_are_counted_from_the_response() -> None:
    response = {"choices": [{"message": {"tool_calls": [{}, {}]}}]}
    assert count_tool_calls(response) == 2
    assert count_tool_calls({}) == 0


def test_teacher_sees_hint_after_first_message_and_same_completion() -> None:
    end = (90, 91)  # stands in for <|im_end|>\n
    turn = TurnTokens((1, 2, 90, 91, 3, 4), (5, 6), (-0.1, -0.2), (1, 1))
    example = distill_example(turn, privileged_block=(77, 78), message_end=end)
    assert example.teacher_ids == (1, 2, 90, 91, 77, 78, 3, 4, 5, 6)
    assert example.student_ids[-2:] == example.teacher_ids[-2:] == (5, 6)


@pytest.mark.parametrize(
    "name", sorted(p.name for p in (ROOT / "configs").glob("*.toml"))
)
def test_shipped_configs_parse(name: str) -> None:
    config = load_run_config(ROOT / "configs" / name)
    assert config.model.base_model.startswith("Qwen/")
