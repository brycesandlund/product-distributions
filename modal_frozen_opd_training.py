"""Isolated fixed-base teacher OPD ablation and disposable preflight."""
import modal
from modal_training import training_image
from modal_app import volume_mounts

app = modal.App("product-distributions-frozen-opd")
image = training_image.add_local_python_source("modal_training")


def validate(config):
    from product_distributions.training import TrainConfig
    TrainConfig(**config).validate()
    if not (config.get("fixed_self_teacher") is True
            and config.get("teacher_privileged") is True
            and config.get("opd_pointwise_clip") is None
            and config.get("group_size") == 1
            and config.get("prompts_per_step") == 16):
        raise ValueError("Expected fixed privileged teacher, no clipping, 16 independent questions")


@app.function(image=image, gpu="A10", cpu=4, memory=32768,
              timeout=12 * 3600, volumes=volume_mounts)
def train_segment(config: dict, run_name: str, data_name: str, resume: str | None = None):
    from modal_training import run_segment
    validate(config)
    if data_name != "bigmath-opd-8192-v1":
        raise ValueError("Expected matched OPD dataset")
    return run_segment(config, run_name, data_name, resume, train_segment)


@app.function(image=image, gpu="A10", cpu=4, memory=32768,
              timeout=1800, volumes=volume_mounts)
def check(config: dict):
    import torch
    from product_distributions.training import Trainer, TrainConfig, completion_logits
    from product_distributions.data import Example
    from product_distributions.losses import full_vocab_opd_loss
    from modal_app import HF_CACHE_PATH
    validate(config)
    t = Trainer(TrainConfig(**config), HF_CACHE_PATH)
    student, teacher = t._prompts(Example("check", "Joan has 8 balloons and loses 2. How many remain?", "6"))
    tokens = t.tokenizer.encode("8 - 2 = 6. \\boxed{6}", add_special_tokens=False)
    target = t.teacher_completion_logits(teacher, tokens).clone()
    with torch.no_grad():
        initial = completion_logits(t.model, t.tokenizer, teacher, tokens).clone()
    assert torch.equal(target, initial), "Initial adapter must reproduce base teacher"
    losses = []
    for _ in range(3):
        t.model.train()
        t.optimizer.zero_grad(set_to_none=True)
        logits = completion_logits(t.model, t.tokenizer, student, tokens)
        loss = full_vocab_opd_loss(logits, t.teacher_completion_logits(teacher, tokens), 0, len(tokens))
        t.model.train()
        t.accelerator.backward(loss)
        t.optimizer.step()
        losses.append(loss.item())
        del logits, loss
    final_teacher = t.teacher_completion_logits(teacher, tokens)
    with torch.no_grad():
        final_student = completion_logits(t.model, t.tokenizer, teacher, tokens)
    teacher_delta = (final_teacher-target).abs().max().item()
    student_delta = (final_student-initial).abs().max().item()
    assert teacher_delta == 0, f"Frozen teacher changed: {teacher_delta}"
    assert student_delta > 0, "Student failed to update or adapters not restored"
    assert all("lora_" in name for name,p in t.model.named_parameters() if p.requires_grad)
    return {"passed": True, "teacher_max_logit_change": teacher_delta,
            "student_max_logit_change": student_delta, "losses": losses}
