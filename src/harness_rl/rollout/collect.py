"""Run black-box Pi rollouts through OpenEnv Harbor and keep what training needs.

This is an I/O edge: it talks to the OpenEnv Harbor server, which starts a
Docker sandbox per rollout, runs Pi inside it, routes every model call through
the capture proxy to vLLM, and scores the result with the task's verifier.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from harness_rl.config import HarborConfig
from harness_rl.rollout.rewards import correctness_reward
from harness_rl.rollout.traces import TurnTokens


@dataclass(frozen=True, slots=True)
class PromptRow:
    """One task prompt as served by the Harbor server (``prompt_rows()``)."""

    prompt: tuple[Mapping[str, str], ...]
    task_name: str
    task_index: int


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Outcome of one attempt at one task."""

    rollout_id: str
    task_name: str
    reward: float | None
    turns: tuple[TurnTokens, ...]
    tool_calls: int
    seconds: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RolloutGroup:
    """All attempts at the same task under the same policy version."""

    task_name: str
    results: tuple[RolloutResult, ...]

    @property
    def rewards(self) -> tuple[float | None, ...]:
        """Return each attempt's reward in order."""
        return tuple(result.reward for result in self.results)


type SessionFactory = Any  # harbor_env.harness.HarborSessionFactory (lazy import)
type FactoryBuilder = Callable[[str], SessionFactory]


def make_factory_builder(
    harbor: HarborConfig, llm_url: str, temperature: float
) -> FactoryBuilder:
    """Return a function that builds a session factory for a served model name.

    OpenEnv only accepts ``temperature`` for training rollouts: truncating the
    distribution (top-p/top-k) would make captured log-probs unusable as the
    behaviour policy in the importance ratio.
    """
    from harbor_env.harness import HarborSessionFactory
    from openenv.core.harness.capture.upstream import training_sampling

    def build(model_name: str) -> SessionFactory:
        return HarborSessionFactory(
            harbor.server_url,
            split=harbor.split,
            llm_url=llm_url,
            model=model_name,
            harness=harbor.harness,
            sandbox=harbor.sandbox,
            reward_key=harbor.reward_key,
            agent_timeout_sec=harbor.agent_timeout_sec,
            agent_step_limit=harbor.agent_step_limit,
            sampling=training_sampling({"temperature": temperature}),
        )

    return build


def prompt_rows(factory: SessionFactory) -> tuple[PromptRow, ...]:
    """Fetch the split's task prompts from the Harbor server."""
    return tuple(
        PromptRow(tuple(row["prompt"]), row["task_name"], row["task_index"])
        for row in factory.prompt_rows()
    )


def turn_tokens(turn: Any) -> TurnTokens:  # noqa: ANN401 - openenv TrainingTurn
    """Convert an OpenEnv TrainingTurn; its loss mask spans prompt + completion."""
    n_prompt = len(turn.prompt_token_ids)
    return TurnTokens(
        prompt_ids=tuple(turn.prompt_token_ids),
        completion_ids=tuple(turn.completion_token_ids),
        completion_logps=tuple(turn.per_token_logps),
        completion_mask=tuple(turn.loss_mask[n_prompt:]),
    )


def count_tool_calls(response: Mapping[str, Any]) -> int:
    """Count tool calls in a captured OpenAI-style chat completion response."""
    choices = response.get("choices") or [{}]
    message = choices[0].get("message") or {}
    return len(message.get("tool_calls") or [])


def run_rollout(
    factory: SessionFactory, row: PromptRow, rollout_id: str, seed: int
) -> RolloutResult:
    """Run one Pi session to completion and collect its trace and reward."""
    started = time.monotonic()
    session = factory.create(list(row.prompt), seed=seed, episode_id=rollout_id)
    try:
        status = session.wait_for_completion()
        result = session.result
        if status != 0 or result is None:
            raise RuntimeError(f"rollout failed (status={status})")
        trace = session.fetch_training_trace()
    except Exception as error:  # noqa: BLE001 - one bad sandbox must not stop a run
        elapsed = time.monotonic() - started
        return RolloutResult(
            rollout_id, row.task_name, None, (), 0, elapsed, repr(error)
        )
    finally:
        session.close()
    return RolloutResult(
        rollout_id=rollout_id,
        task_name=row.task_name,
        reward=correctness_reward(result.reward),
        turns=tuple(turn_tokens(turn) for turn in trace.turns),
        tool_calls=sum(count_tool_calls(turn.response) for turn in trace.turns),
        seconds=time.monotonic() - started,
    )


def collect_groups(
    factory: SessionFactory,
    rows: Sequence[PromptRow],
    num_generations: int,
    max_workers: int,
) -> tuple[RolloutGroup, ...]:
    """Run ``num_generations`` independent attempts per task, concurrently."""
    jobs = [
        (row, f"{row.task_name}#{attempt}", attempt)
        for row in rows
        for attempt in range(num_generations)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda job: run_rollout(factory, *job), jobs))
    return tuple(
        RolloutGroup(
            row.task_name, tuple(r for r in results if r.task_name == row.task_name)
        )
        for row in rows
    )
