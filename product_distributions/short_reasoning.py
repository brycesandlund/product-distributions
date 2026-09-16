"""Bounded, inference-only prompt/mode calibration; no training updates."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from .data import Example, example_id, parse_gold, prompts, read_examples, verify_answer
from .sampling import SamplingConfig
from .training import TrainConfig, Trainer, write_json

CONCISE = (
    "Solve the problem using one concise derivation. Avoid alternative methods "
    "and repeated checks once the result is established. Give only the requested "
    "quantity, with the final answer in \\boxed{}."
)


def run_short_reasoning(output_dir, data_dir, cache_dir):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    revision = HfApi().dataset_info("openai/gsm8k").sha
    gsm = load_dataset(
        "openai/gsm8k", "main", split="train", revision=revision, streaming=True
    )
    examples = [
        ("deepmath", e) for e in read_examples(Path(data_dir) / "train.jsonl")[:2]
    ]
    for row in gsm.take(2):
        answer = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        parse_gold(answer)
        examples.append(
            ("gsm8k", Example(example_id(row["question"]), row["question"], answer))
        )
    setup = TrainConfig(model_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    trainer = Trainer(setup, cache_dir)
    trainer.model.eval()
    eos = trainer.model.generation_config.eos_token_id
    extra_eos = tuple([eos] if isinstance(eos, int) else eos or [])
    eos_ids = set(extra_eos) | {trainer.tokenizer.eos_token_id}
    arms = [
        ("concise-off-student", False, 0.0, 2048),
        ("concise-off-product", False, 0.5, 2048),
        ("concise-on-product", True, 0.5, 8192),
    ]
    write_json(
        output / "config.json",
        {
            "model_setup": asdict(setup),
            "gsm8k_revision": revision,
            "deepmath_manifest": json.loads(
                (Path(data_dir) / "manifest.json").read_text()
            ),
            "examples": [{"source": source, **asdict(e)} for source, e in examples],
            "instruction": CONCISE,
            "arms": arms,
            "eos_ids": sorted(eos_ids),
            "optimizer_steps": 0,
            "seed": 42,
        },
    )
    summaries = []
    for name, thinking, alpha, cap in arms:
        pairs = [
            prompts(trainer.tokenizer, e, thinking, instruction=CONCISE)
            for _, e in examples
        ]
        student, teacher = map(list, zip(*pairs, strict=True))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print(
            json.dumps({"event": "short_reasoning_start", "arm": name, "cap": cap}),
            flush=True,
        )
        start = perf_counter()
        results = trainer.sampler.generate(
            student,
            teacher,
            config=SamplingConfig(
                teacher_weight=alpha,
                temperature=1,
                top_k=None,
                top_p=None,
                max_new_tokens=cap,
                seed=42,
                extra_eos_token_ids=extra_eos,
            ),
        )
        torch.cuda.synchronize()
        seconds = perf_counter() - start
        rows = []
        for index, ((source, example), pair, result) in enumerate(
            zip(examples, pairs, results, strict=True)
        ):
            raw = trainer.tokenizer.decode(result.token_ids, skip_special_tokens=False)
            row = {
                "arm": name,
                "source": source,
                "example_id": example.id,
                "question": example.question,
                "answer": example.answer,
                "student_prompt": pair[0],
                "teacher_prompt": pair[1],
                "raw_text": raw,
                **result.to_dict(),
                "ended_with_eos": bool(
                    result.token_ids and result.token_ids[-1] in eos_ids
                ),
                **verify_answer(result.text, example.answer),
            }
            row["prefix_scores"] = []
            for prefix_cap in (2048, 4096, 8192):
                if prefix_cap > cap:
                    continue
                ids = result.token_ids[:prefix_cap]
                row["prefix_scores"].append(
                    {
                        "cap": prefix_cap,
                        "ended_with_eos": bool(ids and ids[-1] in eos_ids),
                        **verify_answer(
                            trainer.tokenizer.decode(ids, skip_special_tokens=True),
                            example.answer,
                        ),
                    }
                )
            rows.append(row)
            write_json(output / f"{name}-{index + 1}.json", row)
            (output / f"{name}-{index + 1}.md").write_text(
                f"# {name}: {source}, sample {index + 1}\n\n"
                f"Tokens: {result.num_generated_tokens}; EOS: {row['ended_with_eos']}; verifier: {row['status']}.\n\n"
                "## Student prompt\n\n```text\n" + pair[0] + "\n```\n\n"
                "## Teacher prompt\n\n```text\n" + pair[1] + "\n```\n\n"
                "## Complete response (special tokens retained)\n\n```text\n"
                + raw
                + "\n```\n"
            )
        summary = {
            "arm": name,
            "cap": cap,
            "seconds": seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "results": [
                {
                    k: r[k]
                    for k in (
                        "source",
                        "example_id",
                        "num_generated_tokens",
                        "ended_with_eos",
                        "reward",
                        "status",
                        "prefix_scores",
                    )
                }
                for r in rows
            ],
        }
        summaries.append(summary)
        write_json(output / "summary.json", summaries)
        print(json.dumps({"event": "short_reasoning_complete", **summary}), flush=True)
    assert trainer.step == 0
    return {"output_dir": str(output), "optimizer_steps": 0, "summaries": summaries}
