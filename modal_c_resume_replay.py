"""One bounded replay of C updates 161–192; never spawns a successor."""
import modal

from modal_training import training_image
from modal_app import HF_CACHE_PATH, RESULTS_PATH, results_volume, volume_mounts

app = modal.App("product-distributions-c-resume-replay")
image = training_image.add_local_python_source("modal_training")
SOURCE = "runs/bigmath-4b-teacher9b-192-20260921/segment-000192-attempt-b008b86f396f"


@app.function(image=image, gpu="L40S", cpu=4, memory=65536,
              timeout=6 * 3600, retries=0, volumes=volume_mounts)
def replay():
    import json
    from pathlib import Path
    from product_distributions.training import run_training, write_json
    from product_distributions.recovery import checkpoint_state

    results_volume.reload()
    source = Path(RESULTS_PATH) / SOURCE
    checkpoint = source / "checkpoint-000160"
    config = json.loads((source / "config.json").read_text())
    config["steps"] = 192
    state = checkpoint_state(checkpoint, config)
    if state is None or state["step"] != 160:
        raise ValueError("Expected complete original C checkpoint 160")
    output = Path(RESULTS_PATH) / "diagnostics/c-resume160-replay-20261002"
    if output.exists():
        raise FileExistsError("Refusing to overwrite or automatically replay a diagnostic")
    originals = {r["step"]: r for r in map(json.loads, (source / "metrics.jsonl").read_text().splitlines())}
    comparisons = []

    def commit():
        metrics = output / "metrics.jsonl"
        if metrics.exists():
            for line in metrics.read_text().splitlines()[len(comparisons):]:
                current = json.loads(line)
                step = current["step"]
                filename = f"rollouts-{step:06d}.json"
                old = json.loads((source / filename).read_text())
                new = json.loads((output / filename).read_text())
                if [r["example_id"] for r in old] != [r["example_id"] for r in new]:
                    raise AssertionError("Replay question order differs")
                prefixes = []
                for a, b in zip(old, new, strict=True):
                    x, y = a["rollout"]["token_ids"], b["rollout"]["token_ids"]
                    prefixes.append(next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y))))
                comparisons.append({
                    "step": step, "question_order_matches": True,
                    "identical_responses": sum(a["rollout"]["token_ids"] == b["rollout"]["token_ids"] for a, b in zip(old, new)),
                    "common_prefix_tokens": prefixes,
                    "original": originals[step], "replay": current,
                })
        write_json(output / "comparison.json", {
            "source": str(source), "resume": str(checkpoint),
            "target_step": 192, "updates": comparisons,
        })
        results_volume.commit()

    try:
        result = run_training(config, str(output),
                              f"{RESULTS_PATH}/data/bigmath-ab-4b-512-v1",
                              HF_CACHE_PATH, resume=str(checkpoint),
                              commit=commit, evaluate_before=False)
        write_json(output / "result.json", result)
        return result
    finally:
        results_volume.commit()
