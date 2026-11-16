import pytest

torch = pytest.importorskip("torch")

from harness_rl.distill.losses import (  # noqa: E402
    chunked_jsd_sum,
    generalized_jsd,
    kl_divergence,
)
from harness_rl.training.grpo_loss import ClipConfig, grpo_loss_sum  # noqa: E402
from harness_rl.training.logprobs import selected_token_logps  # noqa: E402

CLIP = ClipConfig(0.2, 0.28, 0.5, 5.0)


def test_on_policy_ratio_is_one_and_gradient_follows_advantage() -> None:
    logps = torch.tensor([-1.0, -2.0], requires_grad=True)
    terms = grpo_loss_sum(logps, logps.detach().clone(), advantage=1.0, clip=CLIP)
    terms.loss_sum.backward()
    assert terms.ratio_sum == pytest.approx(2.0)
    assert torch.all(logps.grad < 0)  # minimising loss raises good tokens' prob


def test_tokens_outside_importance_window_are_dropped() -> None:
    new = torch.tensor([0.0, -0.1], requires_grad=True)
    old = torch.tensor([-3.0, -0.1])  # ratio e^3 > 5 for the first token
    terms = grpo_loss_sum(new, old, advantage=1.0, clip=CLIP)
    assert terms.is_masked == 1


def test_chunked_logps_match_full_computation() -> None:
    torch.manual_seed(0)
    hidden, weight = torch.randn(7, 4), torch.randn(11, 4)
    targets = torch.randint(0, 11, (7,))
    full = (hidden @ weight.T).log_softmax(-1).gather(-1, targets[:, None])[:, 0]
    chunked = selected_token_logps(hidden, weight, targets, 1.0, chunk_size=3)
    assert torch.allclose(full, chunked, atol=1e-6)


def test_jsd_endpoints_are_forward_and_reverse_kl() -> None:
    torch.manual_seed(0)
    s = torch.randn(3, 5).log_softmax(-1)
    t = torch.randn(3, 5).log_softmax(-1)
    assert torch.allclose(generalized_jsd(s, t, 0.0), kl_divergence(t, s))
    assert torch.allclose(generalized_jsd(s, t, 1.0), kl_divergence(s, t))
    assert torch.all(generalized_jsd(s, t, 0.5) >= 0)


def test_chunked_jsd_matches_unchunked_and_has_gradient() -> None:
    torch.manual_seed(0)
    student = torch.randn(6, 4, requires_grad=True)
    teacher, weight = torch.randn(6, 4), torch.randn(9, 4)
    chunked = chunked_jsd_sum(student, teacher, weight, beta=0.5, chunk_size=4)
    whole = generalized_jsd(
        (student @ weight.T).log_softmax(-1), (teacher @ weight.T).log_softmax(-1), 0.5
    ).sum()
    assert torch.allclose(chunked, whole, atol=1e-5)
    chunked.backward()
    assert student.grad.abs().sum() > 0
