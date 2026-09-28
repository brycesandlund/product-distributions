import pytest
import torch

from product_distributions.losses import (
    full_vocab_opd_loss,
    imitation_loss,
    opd_loss,
    sampled_logprobs,
    trajectory_weights,
)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("cap", [1e-6, 0.05])
def test_pointwise_clip_matches_opsd_elementwise_reference(alpha, cap):
    torch.manual_seed(11)
    s = torch.randn(137, 13, requires_grad=True)
    q = torch.randn_like(s, requires_grad=True)
    logp = ((1-alpha)*s + alpha*q.detach()).log_softmax(-1)
    logq = q.detach().log_softmax(-1)
    entries = torch.nn.functional.kl_div(logq, logp, reduction="none", log_target=True)
    expected = entries.clamp(max=cap).sum()/2048
    expected_grad = torch.autograd.grad(expected, s)[0]
    actual = full_vocab_opd_loss(s,q,alpha,2048,pointwise_clip=cap)
    actual.backward()
    assert torch.allclose(actual,expected,atol=1e-7)
    assert torch.allclose(s.grad,expected_grad,atol=1e-7)
    assert q.grad is None


def test_large_pointwise_cap_preserves_unclipped_loss_and_gradient():
    s=torch.tensor([[0.2,-0.4,0.8]],requires_grad=True)
    q=torch.tensor([[0.7,-0.8,0.3]])
    a=full_vocab_opd_loss(s,q,0,1)
    b=full_vocab_opd_loss(s,q,0,1,pointwise_clip=100)
    assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a,s)[0],torch.autograd.grad(b,s)[0])


def test_weight_endpoints_and_independent_scale():
    rewards = torch.tensor([0.0, 1.0, 1.0, 1.0])
    assert torch.equal(
        trajectory_weights(rewards, 2, 0), torch.tensor([-0.5, 0.5, 0.0, 0.0])
    )
    assert torch.equal(trajectory_weights(rewards, 2, 1), rewards)
    assert torch.allclose(
        trajectory_weights(rewards, 2, 0.3).reshape(2, 2).mean(1),
        torch.tensor([0.15, 0.3]),
    )
    assert torch.equal(
        trajectory_weights(rewards, 2, 1, 2), torch.tensor([-1.0, 2.0, 1.0, 1.0])
    )


@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0])
def test_binary_weight_interpolation_and_uniform_groups(beta):
    rewards = torch.tensor([0.0, 1.0, 1.0, 1.0], requires_grad=True)
    weights = trajectory_weights(rewards, 4, beta)
    expected = torch.tensor([-0.75 * (1 - beta)] + [0.25 + 0.75 * beta] * 3)
    assert torch.allclose(weights, expected)
    assert not weights.requires_grad
    assert torch.equal(trajectory_weights(torch.zeros(4), 4, beta), torch.zeros(4))
    assert torch.allclose(
        trajectory_weights(torch.ones(4), 4, beta), torch.full((4,), beta)
    )


@pytest.mark.parametrize("alpha", [0.0, 0.4, 1.0])
def test_sampled_opd_expectation_matches_exact_fixed_state_kl_gradient(alpha):
    student = torch.tensor([0.2, -0.8, 1.1], requires_grad=True)
    teacher = torch.tensor([-0.1, 1.2, 0.3], requires_grad=True)
    product_logp = ((1 - alpha) * student + alpha * teacher.detach()).log_softmax(-1)
    teacher_logp = teacher.detach().log_softmax(-1)
    probabilities = product_logp.detach().exp()
    exact = (product_logp.exp() * (product_logp - teacher_logp)).sum()
    expected = torch.autograd.grad(exact, student, retain_graph=True)[0]
    full = full_vocab_opd_loss(student[None], teacher[None], alpha, 1)
    assert torch.allclose(full, exact, atol=1e-6)
    assert torch.allclose(torch.autograd.grad(full, student)[0], expected, atol=1e-6)
    # Enumerate sampled actions, weighted by collection probabilities.
    loss = sum(
        probabilities[a]
        * opd_loss(
            student[None],
            teacher[None],
            torch.tensor([a]),
            teacher_logp[a : a + 1],
            product_logp[a : a + 1],
            alpha,
            1,
        )
        for a in range(3)
    )
    loss.backward()
    assert torch.allclose(student.grad, expected, atol=1e-6)
    assert teacher.grad is None
    if alpha == 1:
        assert torch.equal(student.grad, torch.zeros_like(student))


def test_full_vocab_opd_fixed_length_normalization_and_teacher_detach():
    student = torch.randn(7, 11, requires_grad=True)
    teacher = torch.randn(7, 11, requires_grad=True)
    loss = full_vocab_opd_loss(student, teacher, 0.0, 32)
    expected = (
        student.softmax(-1)
        * (student.log_softmax(-1) - teacher.detach().log_softmax(-1))
    ).sum() / 32
    assert torch.allclose(loss, expected)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()


def test_imitation_endpoints_and_reward_gradient():
    logits = torch.tensor([0.1, -0.7, 0.8], requires_grad=True)
    logp = logits.log_softmax(-1)
    prob = logp.detach().exp()
    unweighted = sum(
        prob[a] * imitation_loss(logp[a : a + 1], 1.0, 1) for a in range(3)
    )
    assert torch.allclose(
        torch.autograd.grad(unweighted, logits, retain_graph=True)[0],
        torch.zeros(3),
        atol=1e-6,
    )
    reward = torch.tensor([0.0, 1.0, 0.0])
    loss = sum(
        prob[a] * imitation_loss(logp[a : a + 1], reward[a], 1) for a in range(3)
    )
    expected = torch.autograd.grad(
        -(logp.exp() * reward).sum(), logits, retain_graph=True
    )[0]
    assert torch.allclose(torch.autograd.grad(loss, logits)[0], expected)
    fresh = torch.randn(2, 5, requires_grad=True)
    ids = torch.tensor([1, 3])
    assert torch.allclose(
        imitation_loss(sampled_logprobs(fresh, ids), 1, 2),
        torch.nn.functional.cross_entropy(fresh, ids),
    )


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_chunked_full_opd_matches_unchunked_value_and_gradient(alpha):
    torch.manual_seed(0)
    student = torch.randn(2, 137, 17, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    product = ((1-alpha)*student + alpha*teacher.detach()).log_softmax(-1)
    expected = (product.exp()*(product-teacher.detach().log_softmax(-1))).sum()/4096
    gradient = torch.autograd.grad(expected, student)[0]
    actual = full_vocab_opd_loss(student, teacher, alpha, 4096)
    actual.backward()
    assert torch.allclose(actual, expected, atol=1e-7)
    assert torch.allclose(student.grad, gradient, atol=1e-7)
    assert teacher.grad is None
