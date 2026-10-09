"""Pinned 256-update 0.8B experiments I/J/K, matching the hardware pilots."""
from dataclasses import asdict
from .training import TrainConfig
from .short_reasoning import CONCISE

DATA_NAME = "bigmath-opd-8192-v1"


def arm_config(arm):
    if arm not in {"I", "J", "K"}:
        raise ValueError("Expected arm I, J or K")
    config = TrainConfig(
        model_id="Qwen/Qwen3.5-0.8B", model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        teacher_model_id=None if arm == "I" else "Qwen/Qwen3.5-9B",
        teacher_revision=None if arm == "I" else "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        teacher_privileged=False, loss="opd_full" if arm == "K" else "imitation",
        alpha=0.5 if arm == "J" else 0, beta=0,
        steps=256, prompts_per_step=16 if arm == "K" else 4,
        group_size=1 if arm == "K" else 4, max_new_tokens=2048,
        learning_rate=2e-5, enable_thinking=False, instruction=CONCISE,
        extra_eos_token_ids=(248044,), eval_batch_size=8,
        eval_every=32, checkpoint_every=32,
    )
    config.validate()
    return asdict(config)
