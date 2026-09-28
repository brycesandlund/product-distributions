"""Dedicated deployment for the bounded 512-step privileged self-OPD run."""

import modal
from modal_training import training_image
from modal_app import volume_mounts

app = modal.App("product-distributions-opd-training")
image = training_image.add_local_python_source("modal_training")


@app.function(image=image, gpu="A10", cpu=4, memory=32768,
              timeout=12 * 3600, volumes=volume_mounts)
def train_segment(config: dict, run_name: str, data_name: str, resume: str | None = None):
    import inspect
    from product_distributions.losses import full_vocab_opd_loss
    from modal_training import run_segment

    if not (config.get("loss") == "opd_full" and config.get("alpha") == 0
            and config.get("teacher_privileged") is True
            and config.get("teacher_model_id") is None
            and config.get("group_size") == 1
            and config.get("prompts_per_step") == 16
            and data_name == "bigmath-opd-8192-v1"):
        raise ValueError("Expected the validated privileged-self OPD configuration")
    if "checkpoint(chunk_kl" not in inspect.getsource(full_vocab_opd_loss):
        raise RuntimeError("Expected memory-bounded full-vocabulary KL implementation")
    return run_segment(config, run_name, data_name, resume, train_segment)
