"""Four bounded hardware pilots; disposable adapters, no training successors."""
import modal
from modal_training import training_image
from modal_app import HF_CACHE_PATH, RESULTS_PATH, volume_mounts, results_volume, model_cache, kernel_cache

app = modal.App("product-distributions-08b-preflight")
image = training_image.add_local_python_source("modal_training")


def pilot(arm, label, run_name):
    import json
    import math
    from dataclasses import asdict
    from pathlib import Path
    from time import perf_counter
    from unittest.mock import patch
    import torch
    from product_distributions.training import Trainer, TrainConfig, completion_logits, write_json
    from product_distributions.losses import full_vocab_opd_loss, sampled_logprobs, imitation_loss
    from product_distributions.sampling import SamplingConfig
    from product_distributions.data import read_examples
    from product_distributions.short_reasoning import CONCISE

    if arm not in {"rl", "product", "opd"} or Path(run_name).name != run_name:
        raise ValueError("Invalid bounded pilot request")
    results_volume.reload()
    output = Path(RESULTS_PATH) / "hardware-benchmarks" / run_name / label
    output.mkdir(parents=True, exist_ok=False)
    config = TrainConfig(
        model_id="Qwen/Qwen3.5-0.8B", model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        teacher_model_id=None if arm == "rl" else "Qwen/Qwen3.5-9B",
        teacher_revision=None if arm == "rl" else "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        teacher_privileged=False, loss="opd_full" if arm == "opd" else "imitation",
        alpha=0.5 if arm == "product" else 0, beta=0,
        steps=3, prompts_per_step=16 if arm == "opd" else 4,
        group_size=1 if arm == "opd" else 4, max_new_tokens=2048,
        learning_rate=2e-5, enable_thinking=False, instruction=CONCISE,
        extra_eos_token_ids=(248044,), eval_batch_size=8,
    )
    report = {"config": asdict(config), "status": "loading", "updates": [],
              "note": "Three disposable updates; synthetic stress is not an accuracy measurement."}
    started = perf_counter()

    def save(status):
        report.update(status=status, elapsed_seconds=perf_counter()-started)
        write_json(output / "summary.json", report)
        results_volume.commit()
        print(json.dumps({"status": status, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)

    try:
        save("loading")
        trainer = Trainer(config, HF_CACHE_PATH)
        examples = read_examples(Path(RESULTS_PATH) / "data/bigmath-opd-8192-v1/train.jsonl")
        report.update(gpu=torch.cuda.get_device_name(), setup_seconds=perf_counter()-started)
        teacher_versions = [p._version for p in trainer.teacher.parameters()] if arm != "rl" else None
        if arm == "rl":
            save("baseline_evaluation")
            heldout = read_examples(Path(RESULTS_PATH) / "data/bigmath-opd-8192-v1/eval.jsonl")[:16]
            baseline = trainer.evaluate(heldout)
            write_json(output / "eval-before.json", baseline)
            report["baseline_accuracy_16"] = baseline["accuracy"]
        # Full-length backward with a real prompt and synthetic continuation.
        save("backward_stress")
        example = max(examples[:48], key=lambda e: len(trainer.tokenizer.encode(trainer._prompts(e)[0])))
        student, teacher = trainer._prompts(example)
        pattern = trainer.tokenizer.encode(" Let us compute this carefully.", add_special_tokens=False)
        tokens = (pattern * (2048 // len(pattern) + 1))[:2048]
        torch.cuda.reset_peak_memory_stats()
        teacher_logits = trainer.teacher_completion_logits(teacher, tokens) if arm == "opd" else None
        trainer.model.train()
        logits = completion_logits(trainer.model, trainer.tokenizer, student, tokens)
        loss = (full_vocab_opd_loss(logits, teacher_logits, 0, 16 * 2048) if arm == "opd"
                else imitation_loss(sampled_logprobs(logits, torch.tensor(tokens, device=logits.device)), 1, 16 * 2048))
        trainer.accelerator.backward(loss)
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)
        report["backward_stress"] = {"tokens": 2048, "optimizer_step": False,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
        trainer.optimizer.zero_grad(set_to_none=True)
        del logits, teacher_logits, loss
        torch.cuda.empty_cache()
        # Disable EOS stopping only for this synthetic cache-capacity stress.
        # Production updates below retain normal EOS behavior and exact sampling.
        save("decode_stress")
        trainer.model.eval()
        trainer.teacher.eval()
        torch.cuda.reset_peak_memory_stats()
        tick = perf_counter()
        with patch("product_distributions.sampling._eos_ids", return_value=set()):
            stress = trainer.sampler.generate([student] * 16, [teacher] * 16,
                config=SamplingConfig(teacher_weight=config.alpha, temperature=1,
                    top_k=None, top_p=None, max_new_tokens=2048, seed=42))
        assert all(r.num_generated_tokens == 2048 for r in stress)
        report["decode_stress"] = {"sequences": 16, "tokens_each": 2048,
            "seconds": perf_counter()-tick, "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
        del stress
        torch.cuda.empty_cache()
        save("updating")
        for step in range(3):
            batch = examples[step*config.prompts_per_step:(step+1)*config.prompts_per_step]
            metrics, records = trainer.update(batch)
            assert all(math.isfinite(metrics[k]) for k in ("loss", "grad_norm", "adapter_update_norm"))
            if teacher_versions is not None:
                assert teacher_versions == [p._version for p in trainer.teacher.parameters()]
                assert all(p.grad is None for p in trainer.teacher.parameters())
                metrics["teacher_unchanged"] = True
            if arm != "opd":
                rewards = [r["grade"]["reward"] for r in records]
                groups = [rewards[i:i+4] for i in range(0, len(rewards), 4)]
                metrics["groups_with_signal"] = sum(min(g) != max(g) for g in groups)
                metrics["groups_total"] = len(groups)
            report["updates"].append(metrics)
            write_json(output / f"rollouts-{step+1}.json", records)
            save("updating")
        checkpoint = trainer.checkpoint(output / "checkpoint-000003")
        report["checkpoint_verification"] = trainer.verify_checkpoint(checkpoint, examples[0])
        save("complete")
        return report
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        save("failed")
        raise
    finally:
        results_volume.commit()
        model_cache.commit()
        kernel_cache.commit()


@app.function(image=image, gpu="L4", cpu=4, memory=32768, timeout=7200, retries=0, volumes=volume_mounts)
def rl_l4(run_name: str):
    return pilot("rl", "rl-l4", run_name)


@app.function(image=image, gpu="A10", cpu=4, memory=32768, timeout=7200, retries=0, volumes=volume_mounts)
def rl_a10(run_name: str):
    return pilot("rl", "rl-a10", run_name)


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=7200, retries=0, volumes=volume_mounts)
def product_l40s(run_name: str):
    return pilot("product", "product-l40s", run_name)


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=7200, retries=0, volumes=volume_mounts)
def opd_l40s(run_name: str):
    return pilot("opd", "opd-l40s", run_name)
