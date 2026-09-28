"""Fixed-prefix objectives. No rollout or teacher gradients are propagated."""

import torch
from torch.utils.checkpoint import checkpoint


def trajectory_weights(rewards, group_size: int, beta: float, reward_scale=None):
    """Interpolate centered advantages and raw rewards; stop verifier gradients."""
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("Rewards must contain complete groups of at least two")
    grouped = rewards.detach().reshape(-1, group_size)
    advantages = (grouped - grouped.mean(dim=1, keepdim=True)).flatten()
    scale = 1 - beta if reward_scale is None else reward_scale
    return beta * rewards.detach() + scale * advantages


def sampled_logprobs(logits, token_ids):
    logits = logits.float()
    return logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)


def imitation_loss(student_logp, weight, denominator: int):
    return (
        -(
            torch.as_tensor(weight, device=student_logp.device).detach() * student_logp
        ).sum()
        / denominator
    )


def opd_loss(
    student_logits,
    teacher_logits,
    token_ids,
    old_teacher_logp,
    old_behavior_logp,
    alpha: float,
    denominator: int,
):
    product_logits = (
        1 - alpha
    ) * student_logits.float() + alpha * teacher_logits.detach().float()
    current_logp = sampled_logprobs(product_logits, token_ids)
    advantage = (old_teacher_logp - old_behavior_logp).detach()
    return -(advantage * current_logp).sum() / denominator


def full_vocab_opd_loss(student_logits, teacher_logits, alpha: float, denominator: int,
                       pointwise_clip: float | None = None):
    """Exact action KL(product || teacher) at the sampled, fixed prefixes.

    Differentiate both product probabilities and their log probabilities; only
    the teacher is detached. This does not differentiate the prefix distribution.
    """
    # Chunk positions, not vocabulary: each softmax still sees every action.
    # Checkpointing avoids retaining sequence-wide float32 KL intermediates.
    def chunk_kl(student, teacher):
        teacher = teacher.detach().float()
        product_logp = ((1 - alpha) * student.float() + alpha * teacher).log_softmax(-1)
        teacher_logp = teacher.log_softmax(-1)
        contributions = product_logp.exp() * (product_logp - teacher_logp)
        # OPSD released implementation caps each position/vocabulary entry,
        # before vocabulary reduction. Signed negative entries are unchanged.
        # The resulting clipped surrogate need not be a nonnegative divergence.
        if pointwise_clip is not None:
            contributions = contributions.clamp(max=pointwise_clip)
        return contributions.sum()

    student = student_logits.reshape(-1, student_logits.shape[-1])
    teacher = teacher_logits.detach().reshape(-1, teacher_logits.shape[-1])
    total = student.new_zeros((), dtype=torch.float32)
    for start in range(0, student.shape[0], 128):
        args = (student[start:start + 128], teacher[start:start + 128])
        if torch.is_grad_enabled() and student.requires_grad:
            total = total + checkpoint(chunk_kl, *args, use_reentrant=False)
        else:
            total = total + chunk_kl(*args)
    return total / denominator
