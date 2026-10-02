"""Dedicated, bounded arm F training deployment."""
import modal
from modal_training import training_image
from modal_app import volume_mounts

app = modal.App("product-distributions-arm-f-training")
image = training_image.add_local_python_source("modal_training")


@app.function(image=image, gpu="L40S", cpu=4, memory=65536,
              timeout=12 * 3600, volumes=volume_mounts)
def train_segment(config: dict, run_name: str, data_name: str, resume: str | None = None):
    from product_distributions.training import TrainConfig
    from modal_training import run_segment
    TrainConfig(**config).validate()
    if not (1 <= config["steps"] <= 192
            and config.get("model_id") == "Qwen/Qwen3.5-4B"
            and config.get("teacher_model_id") == "Qwen/Qwen3.5-9B"
            and config.get("teacher_privileged") is False
            and config.get("loss") == "opd_full" and config.get("alpha") == 0
            and config.get("opd_pointwise_clip") is None
            and config.get("prompts_per_step") == 16 and config.get("group_size") == 1
            and data_name == "bigmath-opd-8192-v1"):
        raise ValueError("Expected bounded arm F configuration")
    return run_segment(config, run_name, data_name, resume, train_segment)
