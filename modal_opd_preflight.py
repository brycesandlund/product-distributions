"""Isolated, bounded CPU preparation and A10 OPD pilot."""

import modal
from modal_training import training_image
from modal_app import HF_CACHE_PATH, RESULTS_PATH, volume_mounts, results_volume, model_cache, kernel_cache

app = modal.App("product-distributions-opd-preflight")
image = training_image.add_local_python_source("modal_training")
DATA_NAME = "bigmath-opd-8192-v1"


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes=volume_mounts)
def prepare():
    from product_distributions.opd_preflight import expand_data
    results_volume.reload()
    result = expand_data(f"{RESULTS_PATH}/data/bigmath-raw/train.parquet",
                         f"{RESULTS_PATH}/data/bigmath-ab-4b-512-v1",
                         f"{RESULTS_PATH}/data/{DATA_NAME}")
    results_volume.commit()
    return result


@app.function(image=image, gpu="A10", cpu=4, memory=32768, timeout=3600, volumes=volume_mounts)
def pilot(config: dict, run_name: str):
    from pathlib import Path
    from product_distributions.opd_preflight import benchmark
    if Path(run_name).name != run_name:
        raise ValueError("Run name must be one path component")
    results_volume.reload()
    try:
        return benchmark(config, f"{RESULTS_PATH}/hardware-benchmarks/{run_name}",
                         f"{RESULTS_PATH}/data/{DATA_NAME}", HF_CACHE_PATH, results_volume.commit)
    finally:
        results_volume.commit()
        model_cache.commit()
        kernel_cache.commit()
