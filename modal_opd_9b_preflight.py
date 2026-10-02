"""Arm F: isolated three-update L40S preflight; never spawns training."""
import modal
from modal_training import training_image
from modal_app import HF_CACHE_PATH, RESULTS_PATH, volume_mounts, results_volume, model_cache, kernel_cache

app = modal.App("product-distributions-opd-9b-preflight")
image = training_image.add_local_python_source("modal_training")


@app.function(image=image, gpu="L40S", cpu=4, memory=65536,
              timeout=3600, volumes=volume_mounts)
def pilot(config: dict, run_name: str):
    from pathlib import Path
    from product_distributions.opd_preflight import benchmark
    if Path(run_name).name != run_name:
        raise ValueError("Run name must be one path component")
    if not (config.get("teacher_model_id") == "Qwen/Qwen3.5-9B"
            and config.get("teacher_privileged") is False
            and config.get("opd_pointwise_clip") is None):
        raise ValueError("Expected frozen question-only 9B teacher, no clipping")
    results_volume.reload()
    try:
        return benchmark(config, f"{RESULTS_PATH}/hardware-benchmarks/{run_name}",
                         f"{RESULTS_PATH}/data/bigmath-opd-8192-v1",
                         HF_CACHE_PATH, results_volume.commit)
    finally:
        results_volume.commit()
        model_cache.commit()
        kernel_cache.commit()
