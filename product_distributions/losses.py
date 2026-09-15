"""Sampled-token objectives. No rollout or teacher gradients are propagated."""

import torch


def trajectory_weights(rewards, group_size: int, beta: float, reward_scale=None):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("Rewards must contain complete groups of at least two")
    grouped = rewards.detach().reshape(-1, group_size)
    advantages = (grouped - grouped.mean(dim=1, keepdim=True)).flatten()
    scale = 1 - beta if reward_scale is None else reward_scale
    return beta + scale * advantages


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
