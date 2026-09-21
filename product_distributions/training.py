"""Single-GPU, synchronous LoRA training with one update per fresh rollout batch."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch

from .data import prompts, read_examples, verify_answer
from .losses import (
    full_vocab_opd_loss,
    imitation_loss,
    opd_loss,
    sampled_logprobs,
    trajectory_weights,
)
from .sampling import ProductSampler, SamplingConfig


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen3.5-9B"
    teacher_model_id: str | None = None
    model_revision: str | None = None
    teacher_revision: str | None = None
    teacher_privileged: bool = True
    loss: str = "imitation"
    alpha: float = 0.5
    beta: float = 0.5
    reward_scale: float | None = None
    steps: int = 10
    prompts_per_step: int = 1
    group_size: int = 4
    max_new_tokens: int = 512
    max_prompt_tokens: int = 2048
    learning_rate: float = 1e-4
    lora_rank: int = 16
    seed: int = 42
    enable_thinking: bool = True
    eval_every: int = 5
    checkpoint_every: int = 5
    instruction: str | None = None
    extra_eos_token_ids: tuple[int, ...] = ()
    eval_batch_size: int = 1

    def validate(self):
        if self.loss not in {"imitation", "opd", "opd_full"}:
            raise ValueError("loss must be imitation, opd, or opd_full")
        if not 0 <= self.alpha <= 1 or not 0 <= self.beta <= 1:
            raise ValueError("alpha and beta must be in [0, 1]")
        for name in (
            "steps",
            "prompts_per_step",
            "group_size",
            "max_new_tokens",
            "max_prompt_tokens",
            "lora_rank",
            "eval_every",
            "checkpoint_every",
            "eval_batch_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.group_size < 2:
            raise ValueError("group_size must be at least two")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.reward_scale is not None and not math.isfinite(self.reward_scale):
            raise ValueError("reward_scale must be finite")


def enable_qwen_kernels():
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )

    modeling.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    modeling.torch_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule


def completion_logits(model, tokenizer, prompt, token_ids):
    """Unpadded microbatch, predicting exactly the continuation, including EOS."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([prompt_ids + token_ids[:-1]], device=device)
    return model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=False,
        logits_to_keep=len(token_ids),
        return_dict=True,
    ).logits[0]


def write_json(path, value):
    from uuid import uuid4
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class Trainer:
    def __init__(self, config: TrainConfig, cache_dir: str):
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from huggingface_hub import HfApi
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        config.validate()
        self.config = config
        self.accelerator = Accelerator(mixed_precision="bf16")
        if self.accelerator.num_processes != 1:
            raise ValueError("This initial trainer supports one process/GPU only")
        set_seed(config.seed)
        config.model_revision = (
            config.model_revision or HfApi().model_info(config.model_id).sha
        )
        if config.teacher_model_id:
            config.teacher_revision = (
                config.teacher_revision
                or HfApi().model_info(config.teacher_model_id).sha
            )
        enable_qwen_kernels()
        torch.set_float32_matmul_precision("high")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, cache_dir=cache_dir, revision=config.model_revision
        )
        kwargs = {
            "cache_dir": cache_dir,
            "dtype": torch.bfloat16,
            "device_map": {"": self.accelerator.device},
            "low_cpu_mem_usage": True,
        }
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            config.model_id, revision=config.model_revision, **kwargs
        )
        # Match decoder linear layers only, excluding the vision encoder and LM head.
        targets = [
            name
            for name, module in base.named_modules()
            if isinstance(module, torch.nn.Linear) and ".language_model.layers." in name
        ]
        if not targets:
            raise ValueError("No decoder linear modules found for LoRA")
        self.model = get_peft_model(
            base,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=2 * config.lora_rank,
                lora_dropout=0.0,
                target_modules=targets,
                task_type="CAUSAL_LM",
            ),
        )
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.model.enable_input_require_grads()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=0.0,
        )
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.teacher = self.model
        self.teacher_tokenizer = self.tokenizer
        if config.teacher_model_id:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                config.teacher_model_id,
                cache_dir=cache_dir,
                revision=config.teacher_revision,
            )
            self.teacher = Qwen3_5ForConditionalGeneration.from_pretrained(
                config.teacher_model_id, revision=config.teacher_revision, **kwargs
            )
            self.teacher.requires_grad_(False).eval()
        self.sampler = ProductSampler(
            self.model, self.tokenizer, self.teacher, self.teacher_tokenizer
        )
        self.step = 0
        self.data_fingerprint = None
        self._register_checkpoint_hooks()
        print(
            json.dumps(
                {
                    "event": "model_ready",
                    "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "trainable_parameters": sum(
                        p.numel() for p in self.model.parameters() if p.requires_grad
                    ),
                }
            ),
            flush=True,
        )

    def _register_checkpoint_hooks(self):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        def save_hook(models, weights, output_dir):
            self.accelerator.unwrap_model(models[0]).save_pretrained(
                Path(output_dir) / "adapter"
            )
            weights.clear()  # Avoid saving a redundant 18-GB frozen base checkpoint.

        def load_hook(models, input_dir):
            set_peft_model_state_dict(
                self.accelerator.unwrap_model(models[0]),
                load_file(str(Path(input_dir) / "adapter/adapter_model.safetensors")),
            )
            models.clear()

        self.accelerator.register_save_state_pre_hook(save_hook)
        self.accelerator.register_load_state_pre_hook(load_hook)

    def checkpoint(self, directory: Path):
        from uuid import uuid4
        destination = directory
        if destination.exists():
            raise FileExistsError(destination)
        directory = destination.with_name(f".{destination.name}.incomplete-{uuid4().hex}")
        directory.mkdir(parents=True, exist_ok=False)
        self.accelerator.save_state(str(directory))
        self.tokenizer.save_pretrained(directory / "tokenizer")
        write_json(
            directory / "state.json",
            {
                "step": self.step,
                "config": asdict(self.config),
                "imitation_weighting": "reward_interpolation_v1",
                "data_fingerprint": self.data_fingerprint,
            },
        )
        directory.rename(destination)
        return str(destination)

    def resume(self, directory: str):
        state = json.loads((Path(directory) / "state.json").read_text())
        if (
            self.config.loss == "imitation"
            and state.get("imitation_weighting") != "reward_interpolation_v1"
        ):
            raise ValueError(
                "Cannot resume imitation checkpoint with old weight semantics"
            )
        old, new = asdict(TrainConfig(**state["config"])), asdict(self.config)
        for values in (old, new):
            values["extra_eos_token_ids"] = list(values["extra_eos_token_ids"])
        old.pop("steps")
        new.pop("steps")
        if old != new:
            raise ValueError("Resume configuration must match (except total steps)")
        if self.data_fingerprint != state.get("data_fingerprint"):
            raise ValueError("Resume dataset fingerprint does not match")
        self.accelerator.load_state(directory)
        self.step = state["step"]

    def _prompts(self, example):
        student, _ = prompts(self.tokenizer, example, self.config.enable_thinking, instruction=self.config.instruction)
        teacher_pair = prompts(
            self.teacher_tokenizer, example, self.config.enable_thinking, instruction=self.config.instruction
        )
        teacher = teacher_pair[1 if self.config.teacher_privileged else 0]
        for tokenizer, prompt in (
            (self.tokenizer, student),
            (self.teacher_tokenizer, teacher),
        ):
            if (
                len(tokenizer.encode(prompt, add_special_tokens=False))
                > self.config.max_prompt_tokens
            ):
                raise ValueError(
                    f"Prompt {example.id} exceeds max_prompt_tokens; no silent truncation"
                )
        return student, teacher

    def update(self, examples):
        config = self.config
        pairs = [self._prompts(e) for e in examples for _ in range(config.group_size)]
        student_prompts, teacher_prompts = map(list, zip(*pairs, strict=True))
        self.model.eval()
        self.teacher.eval()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        results = self.sampler.generate(
            student_prompts,
            teacher_prompts,
            config=SamplingConfig(
                teacher_weight=config.alpha,
                temperature=1.0,
                top_k=None,
                top_p=None,
                max_new_tokens=config.max_new_tokens,
                seed=config.seed + self.step,
                record_logprobs=True,
                require_teacher_logprobs=config.loss == "opd",
                extra_eos_token_ids=tuple(config.extra_eos_token_ids),
            ),
        )
        rollout_seconds = perf_counter() - start
        rollout_peak = torch.cuda.max_memory_allocated() / 2**30
        print(
            json.dumps(
                {
                    "event": "rollouts_ready",
                    "step": self.step + 1,
                    "tokens": sum(r.num_generated_tokens for r in results),
                }
            ),
            flush=True,
        )
        expanded = [e for e in examples for _ in range(config.group_size)]
        grades = [
            verify_answer(r.text, e.answer)
            for r, e in zip(results, expanded, strict=True)
        ]
        rewards = torch.tensor(
            [g["reward"] for g in grades], device=self.accelerator.device
        )
        weights = trajectory_weights(
            rewards, config.group_size, config.beta, config.reward_scale
        )
        denominator = len(results) * config.max_new_tokens
        self.optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        max_logp_error = 0.0
        start = perf_counter()
        for index, (result, (student_prompt, teacher_prompt)) in enumerate(
            zip(results, pairs, strict=True)
        ):
            ids = torch.tensor(result.token_ids, device=self.accelerator.device)
            teacher_logits = None
            if config.loss in {"opd", "opd_full"}:
                self.teacher.eval()
                with torch.no_grad():
                    teacher_logits = completion_logits(
                        self.teacher,
                        self.teacher_tokenizer,
                        teacher_prompt,
                        result.token_ids,
                    )
            self.model.train()
            logits = completion_logits(
                self.model, self.tokenizer, student_prompt, result.token_ids
            )
            logp = sampled_logprobs(logits, ids)
            with torch.no_grad():
                old = torch.tensor(result.student_logprobs, device=ids.device)
                max_logp_error = max(max_logp_error, (logp - old).abs().max().item())
            if config.loss == "imitation":
                loss = imitation_loss(logp, weights[index], denominator)
            elif config.loss == "opd_full":
                loss = full_vocab_opd_loss(
                    logits, teacher_logits, config.alpha, denominator
                )
            else:
                loss = opd_loss(
                    logits,
                    teacher_logits,
                    ids,
                    torch.tensor(result.teacher_logprobs, device=ids.device),
                    torch.tensor(result.behavior_logprobs, device=ids.device),
                    config.alpha,
                    denominator,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss_total += loss.detach().item()
            self.accelerator.backward(loss)
            del logits, logp, loss, teacher_logits
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        grad_sq = sum(
            (p.grad.float().square().sum() for p in parameters if p.grad is not None),
            torch.zeros((), device=self.accelerator.device),
        )
        grad_norm = grad_sq.sqrt().item()
        if not math.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradients; optimizer not stepped")
        # Store adapters only, to measure whether the optimizer actually changes them.
        before = [p.detach().clone() for p in parameters]
        self.optimizer.step()
        with torch.no_grad():
            change = sum(
                (p.float() - b.float()).square().sum()
                for p, b in zip(parameters, before, strict=True)
            )
        update_norm = change.sqrt().item()
        del before
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        self.step += 1
        metrics = {
            "step": self.step,
            "loss_type": config.loss,
            "loss": loss_total,
            "reward_mean": rewards.mean().item(),
            "mean_weight": weights.mean().item() if config.loss == "imitation" else 1.0,
            "group_reward_variance": rewards.reshape(-1, config.group_size)
            .var(dim=1, unbiased=False)
            .mean()
            .item(),
            "grad_norm": grad_norm,
            "adapter_update_norm": update_norm,
            "max_student_logp_recompute_error": max_logp_error,
            "rollout_seconds": rollout_seconds,
            "update_seconds": perf_counter() - start,
            "rollout_peak_gib": rollout_peak,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "generated_tokens": sum(r.num_generated_tokens for r in results),
            "truncated_fraction": sum(
                r.token_ids[-1] not in {self.tokenizer.eos_token_id, *config.extra_eos_token_ids} for r in results
            )
            / len(results),
        }
        records = [
            {
                "example_id": e.id,
                "grade": g,
                "weight": float(weights[i]) if config.loss == "imitation" else 1.0,
                "rollout": r.to_dict(),
            }
            for i, (e, g, r) in enumerate(zip(expanded, grades, results, strict=True))
        ]
        return metrics, records

    def evaluate(self, examples):
        print(
            json.dumps(
                {"event": "evaluation", "step": self.step, "examples": len(examples)}
            ),
            flush=True,
        )
        self.model.eval()
        sampler = ProductSampler(self.model, self.tokenizer)
        records = []
        for index in range(0, len(examples), self.config.eval_batch_size):
            batch = examples[index:index + self.config.eval_batch_size]
            batch_prompts = [self._prompts(e)[0] for e in batch]
            results = sampler.generate(
                batch_prompts,
                batch_prompts,
                config=SamplingConfig(
                    teacher_weight=0,
                    top_k=None,
                    top_p=None,
                    temperature=1,
                    max_new_tokens=self.config.max_new_tokens,
                    seed=self.config.seed + 100000 + index,
                    extra_eos_token_ids=tuple(self.config.extra_eos_token_ids),
                ),
            )
            records.extend(
                {
                    "example_id": e.id,
                    "text": result.text,
                    "num_generated_tokens": result.num_generated_tokens,
                    "ended_with_eos": result.token_ids[-1] in {self.tokenizer.eos_token_id, *self.config.extra_eos_token_ids},
                    **verify_answer(result.text, e.answer),
                }
                for e, result in zip(batch, results, strict=True)
            )
        return {
            "step": self.step,
            "accuracy": sum(r["reward"] for r in records) / len(records),
            "records": records,
        }

    def verify_checkpoint(self, directory, example):
        prompt, _ = self._prompts(example)
        tokens = self.tokenizer.encode("Test", add_special_tokens=False)
        self.model.eval()
        with torch.no_grad():
            expected = completion_logits(
                self.model, self.tokenizer, prompt, tokens
            ).clone()
            # Deliberately corrupt trainable weights, then restore model + optimizer + RNG.
            for p in self.model.parameters():
                if p.requires_grad:
                    p.add_(0.01)
        self.resume(directory)
        self.model.eval()
        with torch.no_grad():
            actual = completion_logits(self.model, self.tokenizer, prompt, tokens)
        error = (actual - expected).abs().max().item()
        if error != 0:
            raise AssertionError(f"Checkpoint reload changed logits: {error}")
        return {"max_logit_error": error, "step": self.step}


def run_training(
    config, output_dir, data_dir, cache_dir, resume=None, retention_path=None, commit=None,
    evaluate_before=True,
):
    config = TrainConfig(**config)
    config.validate()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    train = read_examples(str(Path(data_dir) / "train.jsonl"))
    evaluation = read_examples(str(Path(data_dir) / "eval.jsonl"))
    if {e.id for e in train} & {e.id for e in evaluation} or {
        e.question for e in train
    } & {e.question for e in evaluation}:
        raise ValueError("Training and evaluation overlap")
    manifest = Path(data_dir) / "manifest.json"
    if manifest.exists():
        write_json(root / "data_manifest.json", json.loads(manifest.read_text()))
    if resume:
        previous = json.loads((Path(resume) / "state.json").read_text())
        for field in ("model_revision", "teacher_revision"):
            if getattr(config, field) is None:
                setattr(config, field, previous["config"].get(field))
        if previous["step"] > config.steps:
            raise ValueError("Total steps precedes the resumed checkpoint")
    trainer = Trainer(config, cache_dir)
    trainer.data_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "train": [asdict(e) for e in train],
                "eval": [asdict(e) for e in evaluation],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    write_json(root / "config.json", asdict(config))
    from importlib.metadata import version

    write_json(
        root / "runtime.json",
        {
            "packages": {
                name: version(name)
                for name in (
                    "torch",
                    "transformers",
                    "accelerate",
                    "peft",
                    "math-verify",
                )
            },
            "gpu": torch.cuda.get_device_name(),
            "data_fingerprint": trainer.data_fingerprint,
            "trainable_parameters": sum(
                p.numel() for p in trainer.model.parameters() if p.requires_grad
            ),
        },
    )
    if resume:
        trainer.resume(resume)
    retention = read_examples(retention_path) if retention_path else None
    if evaluate_before:
        write_json(root / "eval-before.json", trainer.evaluate(evaluation))
    if commit:
        commit()
    if retention:
        write_json(root / "retention-before.json", trainer.evaluate(retention))
    while trainer.step < config.steps:
        start = trainer.step * config.prompts_per_step
        batch = [
            train[(start + i) % len(train)] for i in range(config.prompts_per_step)
        ]
        metrics, records = trainer.update(batch)
        print(json.dumps(metrics), flush=True)
        write_json(root / f"rollouts-{trainer.step:06d}.json", records)
        with (root / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        if trainer.step % config.eval_every == 0 or trainer.step == config.steps:
            write_json(
                root / f"eval-{trainer.step:06d}.json", trainer.evaluate(evaluation)
            )
        if trainer.step % config.checkpoint_every == 0 or trainer.step == config.steps:
            checkpoint = trainer.checkpoint(root / f"checkpoint-{trainer.step:06d}")
        if commit:
            commit()
    if retention:
        write_json(root / "retention-after.json", trainer.evaluate(retention))
    if trainer.step == config.steps and "checkpoint" not in locals():
        checkpoint = trainer.checkpoint(root / f"checkpoint-{trainer.step:06d}")
    verification = trainer.verify_checkpoint(checkpoint, train[0])
    write_json(root / "checkpoint-verification.json", verification)
    return {"output_dir": str(root), "checkpoint": checkpoint, "reload": verification}
