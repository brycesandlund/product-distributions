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
            commit=results_volume.commit,
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


@app.function(
    image=training_image, cpu=8, memory=131072, timeout=3600, volumes=volume_mounts
)
def calibrate_short_reasoning(run_name: str):
    from pathlib import Path

    from product_distributions.short_reasoning import run_short_reasoning

    if Path(run_name).name != run_name:
        raise ValueError("Run name must be a single path component")
    try:
        return run_short_reasoning(
            f"{RESULTS_PATH}/short-reasoning/{run_name}",
            f"{RESULTS_PATH}/data/deepmath-v1",
            HF_CACHE_PATH,
        )
    finally:
        results_volume.commit()
        model_cache.commit()
        kernel_cache.commit()


@app.function(
    image=training_image, gpu="H100", cpu=8, memory=131072,
    timeout=4 * 3600, volumes=volume_mounts,
)
def calibrate_bigmath(run_name: str):
    from pathlib import Path
    from product_distributions.bigmath_calibration import run_bigmath_calibration

    if Path(run_name).name != run_name:
        raise ValueError("Run name must be a single path component")
    try:
        return run_bigmath_calibration(
            f"{RESULTS_PATH}/bigmath-calibration/{run_name}",
            f"{RESULTS_PATH}/data/bigmath-raw/train.parquet",
            HF_CACHE_PATH, results_volume.commit,
        )
    finally:
        results_volume.commit()
        model_cache.commit()
        kernel_cache.commit()


@app.function(
    image=training_image, cpu=4, memory=32768, timeout=3600, volumes=volume_mounts,
)
def benchmark_small_model(run_name: str, model_revision: str):
    """Four real updates to test 24GB GPU fit/cost, not a learning comparison."""
    import json
    import resource
    from pathlib import Path
    from time import perf_counter
    import torch
    from product_distributions.training import Trainer, TrainConfig, write_json
    from product_distributions.short_reasoning import CONCISE
    from product_distributions.data import read_examples

    if Path(run_name).name != run_name:
        raise ValueError("Run name must be a single path component")
    output = Path(RESULTS_PATH) / "hardware-benchmarks" / run_name
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    try:
        config = TrainConfig(model_id="Qwen/Qwen3.5-4B", model_revision=model_revision,
                             enable_thinking=False, instruction=CONCISE,
                             max_new_tokens=2048, group_size=4, prompts_per_step=4,
                             beta=0, learning_rate=2e-5, extra_eos_token_ids=(248044,))
        trainer = Trainer(config, HF_CACHE_PATH)
        examples = read_examples(f"{RESULTS_PATH}/data/bigmath-ab-v1/train.jsonl")[:4]
        report = {"model_revision": model_revision, "gpu": torch.cuda.get_device_name(),
                  "setup_seconds": perf_counter() - started, "updates": [],
                  "note": "Throughput/memory test only; arms run sequentially on one disposable adapter."}
        for alpha in (0.0, 0.5):
            config.alpha = alpha
            for repeat in range(2):
                metrics, records = trainer.update(examples)
                metrics.update(alpha=alpha, repeat=repeat,
                               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                               host_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
                report["updates"].append(metrics)
                report["elapsed_seconds"] = perf_counter() - started
                write_json(output / "summary.json", report)
                write_json(output / f"rollouts-{trainer.step}.json", records)
                results_volume.commit()
                print(json.dumps(metrics), flush=True)
        return report
    finally:
        results_volume.commit()
        model_cache.commit()
        kernel_cache.commit()


@app.function(
    image=training_image, gpu="A10", cpu=4, memory=32768,
    timeout=12 * 3600, volumes=volume_mounts,
)
def train_small_ab_segment(config: dict, run_name: str, data_name: str, resume: str | None = None):
    """Bounded 64-step segments, handing off up to the requested total only."""
    import json
    from pathlib import Path
    from product_distributions.training import run_training, write_json

    for name in (run_name, data_name):
        if Path(name).name != name:
            raise ValueError("Run and data names must be single path components")
    total = config["steps"]
    if not 1 <= total <= 512 or config["model_id"] != "Qwen/Qwen3.5-4B":
        raise ValueError("This runner is bounded to 512 steps of the 4B model")
    previous = json.loads((Path(resume) / "state.json").read_text())["step"] if resume else 0
    if previous >= total:
        raise ValueError("Requested training is already complete")
    target = min(previous + 64, total)
    root = Path(RESULTS_PATH) / "runs" / run_name
    try:
        result = run_training(
            {**config, "steps": target}, str(root / f"segment-{target:06d}"),
            f"{RESULTS_PATH}/data/{data_name}", HF_CACHE_PATH,
            resume=resume, commit=results_volume.commit, evaluate_before=resume is None,
        )
        write_json(root / "progress.json", {"completed_steps": target, "total_steps": total, **result})
        results_volume.commit()
        model_cache.commit()
        kernel_cache.commit()
        if target < total:
            child = train_small_ab_segment.spawn(config, run_name, data_name, result["checkpoint"])
            result["next_call_id"] = child.object_id
            write_json(root / "progress.json", {"completed_steps": target, "total_steps": total, **result})
        return result
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
