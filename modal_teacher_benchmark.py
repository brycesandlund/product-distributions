"""Isolated hardware benchmark deployment; does not redeploy active training."""

import modal
from modal_training import training_image
from modal_app import HF_CACHE_PATH, RESULTS_PATH, volume_mounts, results_volume, model_cache, kernel_cache

app = modal.App("product-distributions-teacher-benchmark")

# The entry point imports this module at container startup, not just locally.
benchmark_image = training_image.add_local_python_source("modal_training")


@app.function(image=benchmark_image, cpu=1, memory=4096, timeout=120)
def check_imports():
    import modal_training
    from product_distributions.training import TrainConfig, Trainer
    return {"status": "ok", "modal_training": modal_training.__file__,
            "question_only_supported": not TrainConfig(teacher_privileged=False).teacher_privileged}


@app.function(image=benchmark_image, gpu="L40S", cpu=4, memory=65536,
              timeout=3600, volumes=volume_mounts)
def benchmark(run_name: str):
    import json
    import resource
    from pathlib import Path
    from time import perf_counter
    from dataclasses import asdict
    import torch
    from product_distributions.training import TrainConfig, Trainer, write_json
    from product_distributions.short_reasoning import CONCISE
    from product_distributions.data import read_examples

    if Path(run_name).name != run_name:
        raise ValueError("Run name must be one path component")
    output = Path(RESULTS_PATH) / "hardware-benchmarks" / run_name
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    config = TrainConfig(
        model_id="Qwen/Qwen3.5-4B", model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        teacher_model_id="Qwen/Qwen3.5-9B", teacher_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        teacher_privileged=False, enable_thinking=False, instruction=CONCISE,
        alpha=0.5, beta=0, loss="imitation", max_new_tokens=2048,
        prompts_per_step=4, group_size=4, learning_rate=2e-5, extra_eos_token_ids=(248044,),
    )
    report = {"config": asdict(config), "updates": []}
    try:
        write_json(output / "summary.json", {**report, "status": "loading_models"})
        results_volume.commit()
        trainer = Trainer(config, HF_CACHE_PATH)
        assert not any(p.requires_grad for p in trainer.teacher.parameters())
        teacher_versions = [p._version for p in trainer.teacher.parameters()]
        examples = read_examples(f"{RESULTS_PATH}/data/bigmath-ab-v1/train.jsonl")[:4]
        report.update(gpu=torch.cuda.get_device_name(), setup_seconds=perf_counter()-started,
                      prompts=[{"student": trainer._prompts(e)[0], "teacher": trainer._prompts(e)[1]} for e in examples])
        write_json(output / "summary.json", report)
        results_volume.commit()
        for repeat in range(3):
            metrics, records = trainer.update(examples)
            assert teacher_versions == [p._version for p in trainer.teacher.parameters()]
            assert all(p.grad is None for p in trainer.teacher.parameters())
            metrics.update(repeat=repeat, teacher_unchanged=True,
                           peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                           host_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
            report["updates"].append(metrics)
            report["elapsed_seconds"] = perf_counter()-started
            write_json(output / f"rollouts-{repeat}.json", records)
            write_json(output / "summary.json", report)
            results_volume.commit()
            print(json.dumps(metrics), flush=True)
        return report
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(output / "summary.json", report)
        raise
    finally:
        results_volume.commit()
        model_cache.commit()
        kernel_cache.commit()
