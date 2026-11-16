from harness_rl.reports import check_trace, markdown_table, strategy_report
from harness_rl.rollout.traces import TurnTokens


def test_good_trace_has_no_problems() -> None:
    check = check_trace([TurnTokens((1, 2), (3, 4), (-0.2, -1.0), (1, 1))], 0)
    assert check.problems == ()
    assert check.completion_tokens_trained == 2


def test_trace_flags_trained_prompt_and_bad_logprobs() -> None:
    check = check_trace([TurnTokens((1,), (3,), (0.5,), (1,))], prompt_mask_sum=1)
    assert len(check.problems) == 2


def phase(name: str, seconds: float, gpu: int = 1000) -> dict[str, object]:
    return {"phase": name, "seconds": seconds, "gpu_peak_mib": gpu, "ram_peak_mib": 10}


RECORDS = [
    {"step": -1, "phases": [phase("start", 5.0)]},
    {
        "step": 0,
        "rollouts": {"rollouts": 8},
        "train": {"mean_abs_logp_diff": 0.02},
        "phases": [
            phase("to_inference", 1.0),
            phase("rollouts", 40.0, gpu=15000),
            phase("train_step", 6.0),
        ],
    },
    {
        "step": 1,
        "rollouts": {"rollouts": 8},
        "sync": {"served_vs_trainer": 0.03, "served_change": 0.01},
        "phases": [phase("rollouts", 30.0)],
    },
]


def test_strategy_report_aggregates_phases() -> None:
    report = strategy_report("sleep", RECORDS)
    assert (report.rollouts, report.steps_trained) == (16, 1)
    assert report.gpu_peak_mib == 15000
    assert report.rollout_seconds == 70.0
    assert report.switch_seconds == 1.0
    assert report.mean_abs_logp_diff_first_step == 0.02
    assert report.min_served_change == 0.01


def test_markdown_table_has_a_row_per_metric() -> None:
    table = markdown_table([strategy_report("a", RECORDS)])
    assert table.splitlines()[0] == "| metric | a |"
    assert "| gpu_peak_mib | 15000 |" in table
