"""Arm H: annealed moving privileged self-teacher, then ordinary RL."""
import modal
from modal_training import training_image
from modal_app import volume_mounts

app = modal.App("product-distributions-arm-h-training")
image = training_image.add_local_python_source("modal_training")


@app.function(image=image, gpu="A10", cpu=4, memory=32768,
              timeout=12 * 3600, volumes=volume_mounts)
def train_segment(config: dict, run_name: str, data_name: str, resume: str | None = None):
    from product_distributions.training import TrainConfig
    from modal_training import run_segment
    TrainConfig(**config).validate()
    assert config["steps"] == 256 and not config["fixed_self_teacher"]
    assert config["teacher_privileged"] and config.get("teacher_model_id") is None
    assert config["loss"] == "imitation" and config["alpha"] == 0.5 and config["beta"] == 0
    assert config["alpha_anneal_steps"] == 128
    assert config["prompts_per_step"] == 4 and config["group_size"] == 4
    assert config["eval_every"] == 32 and config["checkpoint_every"] == 32
    assert data_name == "bigmath-ab-4b-512-v1"
    return run_segment(config, run_name, data_name, resume, train_segment)
