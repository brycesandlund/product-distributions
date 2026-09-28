"""Data expansion and bounded OPD feasibility checks; never launches training successors."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .bigmath_calibration import REVISION, select_examples
from .data import example_id, read_examples


def expand_data(parquet, previous, output, count=8192):
    import pyarrow.parquet as pq

    previous, output = Path(previous), Path(output)
    old = json.loads((previous / "manifest.json").read_text())
    assert old["revision"] == REVISION
    excluded = set(old["calibration_excluded_ids"])
    evaluation = read_examples(previous / "eval.jsonl")
    eval_ids = {e.id for e in evaluation}
    rows = []
    for row in pq.read_table(parquet).to_pylist():
        key = example_id(str(row["problem"]))
        is_eval = int(hashlib.sha256(f"ab-v1:{key}".encode()).hexdigest()[:8], 16) % 10 == 0
        if key not in excluded and not is_eval:
            rows.append(row)
    selected, rejected = select_examples(rows, count=count, seed=43)
    ids = [r["id"] for r in selected]
    assert len(set(ids)) == count
    assert not set(ids) & (excluded | eval_ids)
    baseline = read_examples(previous / "train.jsonl")
    assert selected[:len(baseline)] == old["splits"]["train"]["examples"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "train.jsonl").write_text("".join(
        json.dumps({k: r[k] for k in ("id", "question", "answer")}) + "\n" for r in selected
    ))
    (output / "eval.jsonl").write_bytes((previous / "eval.jsonl").read_bytes())
    manifest = {**old, "expanded_from": previous.name,
                "splits": {"train": {"examples": selected, "rejected": rejected},
                           "eval": old["splits"]["eval"]}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report = {"train": count, "eval": len(evaluation), "unique_train": len(set(ids)),
              "calibration_excluded": len(excluded), "holdout_overlap": 0,
              "baseline_train_prefix_preserved": True,
              "eval_sha256": hashlib.sha256((output / "eval.jsonl").read_bytes()).hexdigest()}
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    return report


def benchmark(config, output, data, cache, commit):
    import resource
    from time import perf_counter
    import torch
    from .training import Trainer, TrainConfig, completion_logits, write_json
    from .losses import full_vocab_opd_loss
    import inspect

    config = TrainConfig(**config)
    config.validate()
    assert config.loss == "opd_full" and config.alpha == 0 and config.group_size == 1
    assert config.prompts_per_step == 16 and config.max_new_tokens == 2048
    assert config.model_id == "Qwen/Qwen3.5-4B" and config.teacher_model_id is None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"config": asdict(config), "status": "loading", "updates": [],
              "loss_source": inspect.getsource(full_vocab_opd_loss),
              "note": "Disposable adapter; three updates only, no successor job."}
    started = perf_counter()
    try:
        write_json(output / "summary.json", report)
        commit()
        trainer = Trainer(config, cache)
        examples = read_examples(Path(data) / "train.jsonl")
        report.update(gpu=torch.cuda.get_device_name(), setup_seconds=perf_counter()-started)
        # Force a full-cap backward pass even if all natural pilot rollouts end early.
        # Use the longest tokenized prompt among the first 48 pilot questions.
        example = max(examples[:48], key=lambda e: len(trainer.tokenizer.encode(trainer._prompts(e)[1])))
        student, teacher = trainer._prompts(example)
        pattern = trainer.tokenizer.encode(" Let us compute this carefully.", add_special_tokens=False)
        tokens = (pattern * (2048 // len(pattern) + 1))[:2048]
        torch.cuda.reset_peak_memory_stats()
        trainer.model.eval()
        with torch.no_grad():
            teacher_logits = completion_logits(trainer.teacher, trainer.teacher_tokenizer, teacher, tokens)
        trainer.model.train()
        logits = completion_logits(trainer.model, trainer.tokenizer, student, tokens)
        loss = full_vocab_opd_loss(logits, teacher_logits, 0.0, 16 * 2048)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite stress loss")
        trainer.accelerator.backward(loss)
        assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)
        report["full_length_backward"] = {
            "tokens": len(tokens), "loss": loss.item(), "optimizer_step": False,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
        trainer.optimizer.zero_grad(set_to_none=True)
        del logits, teacher_logits, loss
        torch.cuda.empty_cache()
        write_json(output / "summary.json", {**report, "status": "updating"})
        commit()
        for step in range(3):
            metrics, records = trainer.update(examples[step*16:(step+1)*16])
            metrics.update(peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                           host_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
            report["updates"].append(metrics)
            report["elapsed_seconds"] = perf_counter()-started
            write_json(output / f"rollouts-{step+1}.json", records)
            write_json(output / "summary.json", {**report, "status": "updating"})
            commit()
            print(json.dumps(metrics), flush=True)
        checkpoint = trainer.checkpoint(output / "checkpoint-000003")
        report["checkpoint_verification"] = trainer.verify_checkpoint(checkpoint, examples[0])
        report.update(status="complete", elapsed_seconds=perf_counter()-started)
        write_json(output / "summary.json", report)
        commit()
        return report
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        write_json(output / "summary.json", report)
        commit()
        raise
