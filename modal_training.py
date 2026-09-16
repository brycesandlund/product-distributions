"""Modal entry points for bounded training runs and CPU-only data preparation."""

from datetime import UTC

import modal

from modal_app import (
    HF_CACHE_PATH,
    RESULTS_PATH,
    kernel_cache,
    model_cache,
    results_volume,
    runtime_base_image,
    volume_mounts,
)

app = modal.App("product-distributions-training")
training_image = runtime_base_image.uv_pip_install(
    "peft==0.21.0",
    "datasets==5.0.1",
    "math-verify[antlr4_13_2]==0.9.0",
).add_local_python_source("product_distributions", "modal_app")


@app.function(
    image=training_image, cpu=4, memory=16384, timeout=3600, volumes=volume_mounts
)
def prepare_data(name: str = "deepmath-v1", train_size: int = 128, eval_size: int = 16):
    from pathlib import Path

    from product_distributions.data import prepare_deepmath

    if Path(name).name != name:
        raise ValueError("name must be a single path component")
    result = prepare_deepmath(f"{RESULTS_PATH}/data/{name}", train_size, eval_size)
    results_volume.commit()
    model_cache.commit()
    return result


@app.function(
    image=training_image, cpu=8, memory=131072, timeout=4 * 3600, volumes=volume_mounts
)
def train(
    config: dict,
    run_name: str,
    data_name: str = "deepmath-v1",
    resume: str | None = None,
    retention_path: str | None = None,
):
    from pathlib import Path

    from product_distributions.training import run_training

    for name in (run_name, data_name):
        if Path(name).name != name:
            raise ValueError("Run and data names must be single path components")
    try:
        return run_training(
            config,
            f"{RESULTS_PATH}/runs/{run_name}",
            f"{RESULTS_PATH}/data/{data_name}",
            HF_CACHE_PATH,
            resume=resume,
            retention_path=retention_path,
        )
    finally:
        results_volume.commit()
        model_cache.commit()
        kernel_cache.commit()


@app.function(
    image=training_image, cpu=8, memory=131072, timeout=3600, volumes=volume_mounts
)
def calibrate_lengths(
    run_name: str,
    data_name: str = "deepmath-v1",
    lengths: list[int] | None = None,
    questions: int = 4,
    group_size: int = 4,
    alpha: float = 0.5,
):
    from pathlib import Path

    from product_distributions.calibration import run_calibration

    for name in (run_name, data_name):
        if Path(name).name != name:
            raise ValueError("Run and data names must be single path components")
    try:
        return run_calibration(
            f"{RESULTS_PATH}/calibration/{run_name}",
            f"{RESULTS_PATH}/data/{data_name}",
            HF_CACHE_PATH,
            lengths=[2048, 4096] if lengths is None else lengths,
            questions=questions,
            group_size=group_size,
            alpha=alpha,
        )
    finally:
        results_volume.commit()
        model_cache.commit()
        kernel_cache.commit()


@app.function(
    image=training_image, cpu=8, memory=131072, timeout=1800, volumes=volume_mounts
)
def inspect_native_generation(
    run_name: str, source_name: str = "lengths-9b-self-20260916"
):
    from pathlib import Path

    from product_distributions.native_inspection import run_native_inspection

    for name in (run_name, source_name):
        if Path(name).name != name:
            raise ValueError("Run and source names must be single path components")
    try:
        return run_native_inspection(
            f"{RESULTS_PATH}/calibration/{source_name}",
            f"{RESULTS_PATH}/native-inspection/{run_name}",
            HF_CACHE_PATH,
        )
    finally:
        results_volume.commit()
        model_cache.commit()
        kernel_cache.commit()


@app.local_entrypoint()
def main(
    loss: str = "imitation",
    steps: int = 1,
    max_new_tokens: int = 64,
    group_size: int = 2,
    alpha: float = 0.5,
    beta: float = 0.5,
    gpu: str = "H100",
    data_name: str = "deepmath-smoke",
    thinking: bool = False,
):
    from datetime import datetime

    run_name = f"{loss}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    print(
        train.with_options(gpu=gpu).remote(
            {
                "loss": loss,
                "steps": steps,
                "max_new_tokens": max_new_tokens,
                "group_size": group_size,
                "alpha": alpha,
                "beta": beta,
                "enable_thinking": thinking,
            },
            run_name,
            data_name,
        )
    )
