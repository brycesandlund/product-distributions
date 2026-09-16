"""Inference-only length calibration using the training model and prompt path."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from .data import last_boxed, read_examples, verify_answer
from .sampling import SamplingConfig
from .training import TrainConfig, Trainer, write_json


def summarize(rows):
    """Keep completion and verifier success separate, including capped answers."""
    if not rows:
        raise ValueError("No calibration rows")
    groups = {}
    for row in rows:
        groups.setdefault(row["example_id"], []).append(row["reward"])
    return {
        "rollouts": len(rows),
        "questions": len(groups),
        "eos_fraction": sum(r["ended_with_eos"] for r in rows) / len(rows),
        "boxed_fraction": sum(r["has_boxed_answer"] for r in rows) / len(rows),
        "reward_mean": sum(r["reward"] for r in rows) / len(rows),
        "completed_correct_fraction": sum(
            r["reward"] * r["ended_with_eos"] for r in rows
        )
        / len(rows),
        "mixed_success_groups": sum(0 < sum(rs) < len(rs) for rs in groups.values()),
        "all_failed_groups": sum(sum(rs) == 0 for rs in groups.values()),
        "all_successful_groups": sum(sum(rs) == len(rs) for rs in groups.values()),
        "mean_tokens": sum(r["num_tokens"] for r in rows) / len(rows),
    }


def run_calibration(
    output_dir,
    data_dir,
    cache_dir,
    lengths=(2048, 4096),
    questions=4,
    group_size=4,
    alpha=0.5,
    seed=42,
):
    if questions < 1 or group_size < 2 or not lengths or any(n < 1 for n in lengths):
        raise ValueError("Positive lengths/questions and group_size >= 2 required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    examples = read_examples(Path(data_dir) / "train.jsonl")[:questions]
    if len(examples) != questions:
        raise ValueError("Not enough training examples for calibration")
    config = TrainConfig(
        alpha=alpha,
        group_size=group_size,
        seed=seed,
        max_new_tokens=max(lengths),
        enable_thinking=True,
    )
    trainer = Trainer(config, cache_dir)
    trainer.model.eval()
    trainer.teacher.eval()
    write_json(
        output / "config.json",
        {
            "training_config_for_model_setup_only": asdict(config),
            "lengths": list(lengths),
            "question_ids": [e.id for e in examples],
            "optimizer_steps": 0,
            "gpu": torch.cuda.get_device_name(0),
            "dataset_manifest": json.loads(
                (Path(data_dir) / "manifest.json").read_text()
            ),
        },
    )
    eos = trainer.tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    summaries = []
    for length in lengths:
        rows = []
        batches = []
        for index, example in enumerate(examples):
            student, teacher = trainer._prompts(example)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = perf_counter()
            results = trainer.sampler.generate(
                [student] * group_size,
                [teacher] * group_size,
                config=SamplingConfig(
                    teacher_weight=alpha,
                    temperature=1.0,
                    top_k=None,
                    top_p=None,
                    max_new_tokens=length,
                    seed=seed + index,
                ),
            )
            torch.cuda.synchronize()
            seconds = perf_counter() - start
            batch = {
                "max_new_tokens": length,
                "example_id": example.id,
                "seconds": seconds,
                "tokens": sum(r.num_generated_tokens for r in results),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
            batches.append(batch)
            for sample, result in enumerate(results):
                grade = verify_answer(result.text, example.answer)
                row = {
                    "max_new_tokens": length,
                    "example_id": example.id,
                    "sample": sample,
                    "question": example.question,
                    "gold_answer": example.answer,
                    "text": result.text,
                    "token_ids": result.token_ids,
                    "num_tokens": result.num_generated_tokens,
                    "ended_with_eos": bool(
                        result.token_ids and result.token_ids[-1] in eos_ids
                    ),
                    "has_boxed_answer": last_boxed(result.text) is not None,
                    **grade,
                }
                rows.append(row)
                with (output / "rollouts.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {
                        "event": "calibration_batch",
                        **batch,
                        **summarize(rows[-group_size:]),
                    }
                ),
                flush=True,
            )
        summary = {
            "max_new_tokens": length,
            **summarize(rows),
            "generation_seconds": sum(b["seconds"] for b in batches),
            "aggregate_tokens_per_second": sum(b["tokens"] for b in batches)
            / sum(b["seconds"] for b in batches),
            "peak_allocated_gib": max(b["peak_allocated_gib"] for b in batches),
            "batches": batches,
        }
        summaries.append(summary)
        write_json(output / "summary.json", summaries)
        print(json.dumps({"event": "calibration_summary", **summary}), flush=True)
    assert trainer.step == 0
    return {"output_dir": str(output), "optimizer_steps": 0, "summaries": summaries}
