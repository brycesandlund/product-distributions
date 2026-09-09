"""High-throughput token sampling from two contextual views of one model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ProductGenerationResult:
    """Serializable output from a two-context product rollout."""

    text: str
    token_ids: list[int]
    num_generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    mean_behavior_logprob: float
    mean_teacher_student_kl: float
    teacher_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Apply top-k and nucleus filtering to a single logit row."""
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        cutoff = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        logits = torch.full_like(logits, -torch.inf).scatter(
            dim=-1, index=sorted_indices, src=sorted_logits
        )

    return logits


def _as_eos_set(eos_token_id: int | list[int] | tuple[int, ...] | None) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return set(eos_token_id)


@torch.inference_mode()
def product_generate_two_contexts(
    model: torch.nn.Module,
    tokenizer: Any,
    student_prompt: str,
    teacher_prompt: str,
    *,
    teacher_weight: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = 20,
    top_p: float | None = 0.95,
    max_new_tokens: int = 128,
    seed: int | None = None,
) -> ProductGenerationResult:
    """Sample a shared continuation from two contextual views of one model.

    The behavior distribution is

        mu(a | h) proportional to
            pi_student(a | h) ** (1 - teacher_weight)
            * pi_teacher(a | h) ** teacher_weight.

    Both contexts are evaluated as a batch. The generated token is then appended
    to both views, so the method performs one batched model call per decode step.
    The teacher view is purely contextual here; training code is responsible for
    applying stop-gradient semantics to teacher logits.
    """
    if not 0.0 <= teacher_weight <= 1.0:
        raise ValueError("teacher_weight must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        inputs = tokenizer(
            [student_prompt, teacher_prompt],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = original_padding_side

    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    generated: list[int] = []
    behavior_logprobs: list[float] = []
    teacher_student_kls: list[float] = []
    eos_ids = _as_eos_set(tokenizer.eos_token_id)
    started_at = perf_counter()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :].float()

    for step in range(max_new_tokens):
        student_logits = next_logits[0]
        teacher_logits = next_logits[1]

        # Combining logits is exactly the normalized weighted geometric mean.
        behavior_logits = (
            (1.0 - teacher_weight) * student_logits
            + teacher_weight * teacher_logits
        ) / temperature
        behavior_logits = _filter_logits(
            behavior_logits, top_k=top_k, top_p=top_p
        )
        behavior_probs = F.softmax(behavior_logits, dim=-1)
        next_token = torch.multinomial(
            behavior_probs, num_samples=1, generator=generator
        )
        token_id = int(next_token.item())
        generated.append(token_id)

        behavior_logprobs.append(
            float(F.log_softmax(behavior_logits, dim=-1)[token_id].item())
        )
        student_logprobs = F.log_softmax(student_logits, dim=-1)
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
        teacher_probs = teacher_logprobs.exp()
        teacher_student_kls.append(
            float(
                torch.sum(teacher_probs * (teacher_logprobs - student_logprobs)).item()
            )
        )

        if token_id in eos_ids or step + 1 == max_new_tokens:
            break

        shared_token = next_token.view(1, 1).expand(2, 1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (2, 1), dtype=attention_mask.dtype, device=attention_mask.device
                ),
            ],
            dim=1,
        )
        outputs = model(
            input_ids=shared_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :].float()

    elapsed = perf_counter() - started_at
    count = len(generated)
    return ProductGenerationResult(
        text=tokenizer.decode(generated, skip_special_tokens=True),
        token_ids=generated,
        num_generated_tokens=count,
        elapsed_seconds=elapsed,
        tokens_per_second=count / elapsed if elapsed else 0.0,
        mean_behavior_logprob=sum(behavior_logprobs) / count,
        mean_teacher_student_kl=sum(teacher_student_kls) / count,
        teacher_weight=teacher_weight,
    )
