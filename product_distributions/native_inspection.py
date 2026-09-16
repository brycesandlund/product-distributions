"""Inference-only comparison of saved calibration prompts with native generate."""

import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import GenerationConfig

from .data import Example, prompts, verify_answer
from .training import TrainConfig, Trainer, write_json


def run_native_inspection(source_dir, output_dir, cache_dir):
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((source / "config.json").read_text())
    setup = TrainConfig(**config["training_config_for_model_setup_only"])
    rows = [
        json.loads(line)
        for line in (source / "rollouts.jsonl").read_text().splitlines()
    ]
    selected = [r for r in rows if r["max_new_tokens"] == 4096 and r["sample"] == 0][:2]
    if len(selected) != 2:
        raise ValueError("Expected two saved 4K samples")
    trainer = Trainer(setup, cache_dir)
    model = trainer.model.eval()
    tokenizer = trainer.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    records = []
    for index, row in enumerate(selected, 1):
        example = Example(row["example_id"], row["question"], row["gold_answer"])
        pair = prompts(tokenizer, example, enable_thinking=True)
        for context, prompt in zip(("student", "teacher"), pair, strict=True):
            records.append(
                {
                    "sample": index,
                    "context": context,
                    "example_id": example.id,
                    "gold_answer": example.answer,
                    "prompt": prompt,
                }
            )
    native_eos = model.generation_config.eos_token_id
    generation = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_new_tokens=4096,
        use_cache=True,
        eos_token_id=native_eos,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=model.generation_config.bos_token_id,
    )
    write_json(
        output / "config.json",
        {
            "model_id": setup.model_id,
            "model_revision": setup.model_revision,
            "seed": setup.seed,
            "generation_config": generation.to_dict(),
            "custom_sampler_eos_token_id": tokenizer.eos_token_id,
            "native_default_eos_token_id": native_eos,
            "optimizer_steps": 0,
            "note": "Same initial model, adapters and FLA kernels; independent native "
            "generation on each context, not a product-distribution sampler.",
            "prompts": records,
        },
    )
    encoded = tokenizer(
        [r["prompt"] for r in records],
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(trainer.accelerator.device)
    torch.manual_seed(setup.seed)
    torch.cuda.manual_seed_all(setup.seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = perf_counter()
    print(
        json.dumps(
            {
                "event": "native_generate_start",
                "sequences": len(records),
                "eos_token_id": native_eos,
                "max_new_tokens": 4096,
            }
        ),
        flush=True,
    )
    with torch.inference_mode():
        generated = model.generate(**encoded, generation_config=generation)
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    eos_ids = {native_eos} if isinstance(native_eos, int) else set(native_eos or [])
    summaries = []
    for record, full_ids in zip(records, generated, strict=True):
        ids = full_ids[encoded.input_ids.shape[1] :].tolist()
        for index, token in enumerate(ids):
            if token in eos_ids:
                ids = ids[: index + 1]
                break
        text = tokenizer.decode(ids, skip_special_tokens=True)
        raw_text = tokenizer.decode(ids, skip_special_tokens=False)
        result = {
            **record,
            "token_ids": ids,
            "text": text,
            "raw_text": raw_text,
            "num_tokens": len(ids),
            "ended_with_eos": bool(ids and ids[-1] in eos_ids),
            **verify_answer(text, record["gold_answer"]),
        }
        write_json(
            output / f"sample-{record['sample']}-{record['context']}.json", result
        )
        transcript = (
            f"# Native generate: sample {record['sample']}, {record['context']} context\n\n"
            "Independent Transformers model.generate() output; no product mixing.\n\n"
            f"Tokens: {len(ids)}; EOS: {result['ended_with_eos']}; verifier: {result['status']}.\n\n"
            "## Full rendered prompt\n\n```text\n" + record["prompt"] + "\n```\n\n"
            "## Full response (including special tokens)\n\n```text\n"
            + raw_text
            + "\n```\n"
        )
        (output / f"sample-{record['sample']}-{record['context']}.md").write_text(
            transcript
        )
        summaries.append(
            {
                k: result[k]
                for k in (
                    "sample",
                    "context",
                    "num_tokens",
                    "ended_with_eos",
                    "reward",
                    "status",
                )
            }
        )
    summary = {
        "generation_seconds": seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "optimizer_steps": trainer.step,
        "results": summaries,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"event": "native_generate_complete", **summary}), flush=True)
    return {"output_dir": str(output), **summary}
