"""Bounded production arms I/J/K; each starts fresh and hands off every 64 updates."""
import modal
from modal_training import training_image
from modal_app import volume_mounts

app = modal.App("product-distributions-08b-training")
image = training_image.add_local_python_source("modal_training")


def run(arm, run_name, resume, successor):
    from product_distributions.arms_08b import arm_config, DATA_NAME
    from modal_training import run_segment
    if run_name != f"bigmath-08b-{arm}-256-20261008":
        raise ValueError("Expected the authorized arm's unique run directory")
    return run_segment(arm_config(arm), run_name, DATA_NAME, resume, successor)


@app.function(image=image, gpu="A10", cpu=4, memory=32768,
              timeout=12*3600, retries=0, volumes=volume_mounts)
def arm_i(config: dict, run_name: str, data_name: str, resume: str | None = None):
    validate("I", config, data_name)
    return run("I", run_name, resume, arm_i)


@app.function(image=image, gpu="L40S", cpu=4, memory=32768,
              timeout=12*3600, retries=0, volumes=volume_mounts)
def arm_j(config: dict, run_name: str, data_name: str, resume: str | None = None):
    validate("J", config, data_name)
    return run("J", run_name, resume, arm_j)


@app.function(image=image, gpu="L40S", cpu=4, memory=32768,
              timeout=12*3600, retries=0, volumes=volume_mounts)
def arm_k(config: dict, run_name: str, data_name: str, resume: str | None = None):
    validate("K", config, data_name)
    return run("K", run_name, resume, arm_k)


def validate(arm, config, data_name):
    import json
    from product_distributions.arms_08b import arm_config, DATA_NAME
    if json.dumps(config, sort_keys=True) != json.dumps(arm_config(arm), sort_keys=True) or data_name != DATA_NAME:
        raise ValueError("Configuration differs from authorized arm")
