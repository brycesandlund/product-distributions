"""Batched product-distribution sampling for one or two causal language models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameters shared by all rollouts in a batch."""

    teacher_weight: float = 0.5
    temperature: float = 1.0
    top_k: int | None = 20
    top_p: float | None = 0.95
    max_new_tokens: int = 128
    seed: int | None = None
    compute_diagnostics: bool = False

    def validate(self) -> None:
        if not 0.0 <= self.teacher_weight <= 1.0:
            raise ValueError("teacher_weight must be in [0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1] or None")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass
class ProductGenerationResult:
    """Serializable result for one rollout in a generated batch."""

    text: str
    token_ids: list[int]
    num_generated_tokens: int
    elapsed_seconds: float
    aggregate_tokens_per_second: float
    mean_behavior_logprob: float
    mean_teacher_student_kl: float | None
    teacher_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _DecodeState:
    attention_mask: torch.Tensor
    past_key_values: Any
    next_logits: torch.Tensor


def _as_list(value: str | Sequence[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _model_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def _sync_devices(*devices: torch.device) -> None:
    seen: set[torch.device] = set()
    for device in devices:
        if device.type == "cuda" and device not in seen:
            torch.cuda.synchronize(device)
            seen.add(device)


def _eos_ids(*tokenizers: Any) -> set[int]:
    result: set[int] = set()
    for tokenizer in tokenizers:
        value = tokenizer.eos_token_id
        if value is None:
            continue
        result.update([value] if isinstance(value, int) else value)
    return result


def _tokenize(
    tokenizer: Any, prompts: list[str], device: torch.device
) -> dict[str, torch.Tensor]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = original_padding_side
    return {name: tensor.to(device) for name, tensor in encoded.items()}


def _forward(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any = None,
) -> _DecodeState:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    return _DecodeState(
        attention_mask=attention_mask,
        past_key_values=outputs.past_key_values,
        next_logits=outputs.logits[:, -1, :],
    )


def _advance(
    model: torch.nn.Module,
    state: _DecodeState,
    token_ids: torch.Tensor,
    active: torch.Tensor,
) -> _DecodeState:
    device = state.next_logits.device
    token_ids = token_ids.to(device=device, non_blocking=True).unsqueeze(1)
    next_mask = active.to(
        device=state.attention_mask.device,
        dtype=state.attention_mask.dtype,
        non_blocking=True,
    ).unsqueeze(1)
    attention_mask = torch.cat((state.attention_mask, next_mask), dim=1)
    return _forward(
        model,
        input_ids=token_ids,
        attention_mask=attention_mask,
        past_key_values=state.past_key_values,
    )


def _sample_from_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample tokens and log-probabilities without sorting the full vocabulary."""
    if top_k is not None:
        candidate_logits, candidate_ids = torch.topk(
            logits, min(top_k, logits.shape[-1]), dim=-1, sorted=True
        )
    else:
        candidate_logits = logits
        candidate_ids = torch.arange(logits.shape[-1], device=logits.device).expand_as(
            logits
        )
        if top_p is not None and top_p < 1.0:
            candidate_logits, order = torch.sort(
                candidate_logits, descending=True, dim=-1
            )
            candidate_ids = torch.gather(candidate_ids, dim=-1, index=order)

    if top_p is not None and top_p < 1.0:
        cumulative = torch.cumsum(F.softmax(candidate_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        candidate_logits = candidate_logits.masked_fill(remove, -torch.inf)

    candidate_logprobs = F.log_softmax(candidate_logits, dim=-1)
    sampled_offsets = torch.multinomial(
        candidate_logprobs.exp(), num_samples=1, generator=generator
    )
    sampled_ids = torch.gather(candidate_ids, dim=-1, index=sampled_offsets).squeeze(1)
    sampled_logprobs = torch.gather(
        candidate_logprobs, dim=-1, index=sampled_offsets
    ).squeeze(1)
    return sampled_ids, sampled_logprobs


class ProductSampler:
    """Generate shared rollouts from two model distributions.

    If both views use the same model object, their prompts are concatenated into
    one batch and the model is called once per step. If they use different model
    objects, each model owns its tokenizer, device, and cache. Tokenizer
    vocabularies must map every token string to the same integer ID because the
    sampled continuation is shared by both views.
    """

    def __init__(
        self,
        student_model: torch.nn.Module,
        student_tokenizer: Any,
        teacher_model: torch.nn.Module | None = None,
        teacher_tokenizer: Any | None = None,
        *,
        verify_vocabularies: bool = True,
    ) -> None:
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        self.teacher_model = student_model if teacher_model is None else teacher_model
        self.teacher_tokenizer = (
            student_tokenizer if teacher_tokenizer is None else teacher_tokenizer
        )
        self.shared_model = self.student_model is self.teacher_model

        if (
            verify_vocabularies
            and (
                self.student_tokenizer is not self.teacher_tokenizer
                or not self.shared_model
            )
            and self.student_tokenizer.get_vocab() != self.teacher_tokenizer.get_vocab()
        ):
            raise ValueError(
                "Product sampling requires identical token-to-ID vocabularies"
            )

    @torch.inference_mode()
    def generate(
        self,
        student_prompts: str | Sequence[str],
        teacher_prompts: str | Sequence[str],
        *,
        config: SamplingConfig | None = None,
    ) -> list[ProductGenerationResult]:
        config = config or SamplingConfig()
        config.validate()
        student_prompts = _as_list(student_prompts)
        teacher_prompts = _as_list(teacher_prompts)
        if not student_prompts or len(student_prompts) != len(teacher_prompts):
            raise ValueError(
                "student_prompts and teacher_prompts need equal nonzero lengths"
            )

        student_device = _model_device(self.student_model)
        teacher_device = _model_device(self.teacher_model)
        batch_size = len(student_prompts)
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=student_device).manual_seed(config.seed)

        _sync_devices(student_device, teacher_device)
        started_at = perf_counter()

        if self.shared_model:
            encoded = _tokenize(
                self.student_tokenizer,
                student_prompts + teacher_prompts,
                student_device,
            )
            shared_state = _forward(self.student_model, **encoded)
            student_state = _DecodeState(
                attention_mask=shared_state.attention_mask[:batch_size],
                past_key_values=shared_state.past_key_values,
                next_logits=shared_state.next_logits[:batch_size],
            )
            teacher_state = _DecodeState(
                attention_mask=shared_state.attention_mask[batch_size:],
                past_key_values=shared_state.past_key_values,
                next_logits=shared_state.next_logits[batch_size:],
            )
        else:
            student_state = _forward(
                self.student_model,
                **_tokenize(self.student_tokenizer, student_prompts, student_device),
            )
            teacher_state = _forward(
                self.teacher_model,
                **_tokenize(self.teacher_tokenizer, teacher_prompts, teacher_device),
            )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=student_device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=student_device)
        behavior_logprob_sums = torch.zeros(batch_size, device=student_device)
        kl_sums = (
            torch.zeros(batch_size, device=student_device)
            if config.compute_diagnostics
            else None
        )
        generated_steps: list[torch.Tensor] = []
        eos_ids = _eos_ids(self.student_tokenizer, self.teacher_tokenizer)
        eos_tensor = torch.tensor(sorted(eos_ids), device=student_device)
        fallback_token_id = self.student_tokenizer.pad_token_id
        if fallback_token_id is None:
            fallback_token_id = self.student_tokenizer.eos_token_id or 0

        for step in range(config.max_new_tokens):
            was_active = ~finished
            student_logits = student_state.next_logits.to(student_device).float()
            teacher_logits = teacher_state.next_logits.to(
                student_device, non_blocking=True
            ).float()
            behavior_logits = (
                (1.0 - config.teacher_weight) * student_logits
                + config.teacher_weight * teacher_logits
            ) / config.temperature
            sampled_ids, sampled_logprobs = _sample_from_logits(
                behavior_logits,
                top_k=config.top_k,
                top_p=config.top_p,
                generator=generator,
            )
            sampled_ids = torch.where(
                was_active,
                sampled_ids,
                torch.full_like(sampled_ids, fallback_token_id),
            )
            generated_steps.append(sampled_ids)
            behavior_logprob_sums += sampled_logprobs * was_active
            lengths += was_active

            if kl_sums is not None:
                student_logprobs = F.log_softmax(student_logits, dim=-1)
                teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
                kl_sums += (
                    teacher_logprobs.exp() * (teacher_logprobs - student_logprobs)
                ).sum(dim=-1) * was_active

            if eos_tensor.numel():
                finished |= torch.isin(sampled_ids, eos_tensor)
            if bool(finished.all().item()) or step + 1 == config.max_new_tokens:
                break

            if self.shared_model:
                active_pair = torch.cat((was_active, was_active))
                token_pair = torch.cat((sampled_ids, sampled_ids))
                shared_state = _advance(
                    self.student_model,
                    shared_state,
                    token_pair,
                    active_pair,
                )
                student_state = _DecodeState(
                    attention_mask=shared_state.attention_mask[:batch_size],
                    past_key_values=shared_state.past_key_values,
                    next_logits=shared_state.next_logits[:batch_size],
                )
                teacher_state = _DecodeState(
                    attention_mask=shared_state.attention_mask[batch_size:],
                    past_key_values=shared_state.past_key_values,
                    next_logits=shared_state.next_logits[batch_size:],
                )
            else:
                student_state = _advance(
                    self.student_model, student_state, sampled_ids, was_active
                )
                teacher_state = _advance(
                    self.teacher_model, teacher_state, sampled_ids, was_active
                )

        _sync_devices(student_device, teacher_device)
        elapsed = perf_counter() - started_at
        generated = torch.stack(generated_steps, dim=0).transpose(0, 1).cpu()
        lengths_cpu = lengths.cpu().tolist()
        behavior_means = (behavior_logprob_sums / lengths.clamp_min(1)).cpu().tolist()
        kl_means = (
            (kl_sums / lengths.clamp_min(1)).cpu().tolist()
            if kl_sums is not None
            else [None] * batch_size
        )
        aggregate_tps = int(lengths.sum().item()) / elapsed if elapsed else 0.0

        results = []
        for index, length in enumerate(lengths_cpu):
            token_ids = generated[index, :length].tolist()
            results.append(
                ProductGenerationResult(
                    text=self.student_tokenizer.decode(
                        token_ids, skip_special_tokens=True
                    ),
                    token_ids=token_ids,
                    num_generated_tokens=length,
                    elapsed_seconds=elapsed,
                    aggregate_tokens_per_second=aggregate_tps,
                    mean_behavior_logprob=behavior_means[index],
                    mean_teacher_student_kl=kl_means[index],
                    teacher_weight=config.teacher_weight,
                )
            )
        return results


def product_generate_two_contexts(
    model: torch.nn.Module,
    tokenizer: Any,
    student_prompt: str,
    teacher_prompt: str,
    **sampling_kwargs: Any,
) -> ProductGenerationResult:
    """Convenience wrapper for a single shared-model contextual rollout."""
    config = SamplingConfig(**sampling_kwargs)
    return ProductSampler(model, tokenizer).generate(
        student_prompt, teacher_prompt, config=config
    )[0]
