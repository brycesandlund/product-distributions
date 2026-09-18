"""Source-balanced Big-Math calibration, without optimizer updates."""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .data import Example, example_id, parse_gold, prompts, verify_answer

REVISION = "c75d2f117cddfecb6bd08756e61e508e59732b21"


def select_examples(rows, count=128, seed=42):
    """Round-robin sources, hash-shuffled within source; no outcome filtering."""
    pools = defaultdict(list)
    for row in rows:
        question, answer = str(row["problem"]), str(row["answer"])
        key = example_id(question)
        pools[str(row["source"])].append((key, question, answer))
    for source in pools:
        pools[source].sort(
            key=lambda r: hashlib.sha256(f"{seed}:{r[0]}".encode()).hexdigest(),
            reverse=True,
        )
    selected, seen, rejected = [], set(), []
    while len(selected) < count:
        progress = False
        for source in sorted(pools):
            while pools[source]:
                key, question, answer = pools[source].pop()
                if key in seen:
                    continue
                seen.add(key)
                try:
                    parse_gold(answer)
                except Exception as error:
                    rejected.append({"id": key, "reason": str(error)})
                    continue
                selected.append({"source": source, **asdict(Example(key, question, answer))})
                progress = True
                break
            if len(selected) == count:
                break
        if not progress:
            raise ValueError("Not enough unique parseable examples")
    return selected, rejected


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["example_id"]].append(row)
    n = len(rows)
    return {
        "rollouts": n,
        "questions": len(groups),
        "accuracy": sum(r["reward"] for r in rows) / n,
        "completed_correct": sum(r["reward"] * r["ended_with_eos"] for r in rows) / n,
        "truncated": sum(not r["ended_with_eos"] for r in rows),
        "mean_tokens": sum(r["num_generated_tokens"] for r in rows) / n,
        "all_correct_groups": sum(all(r["reward"] for r in g) for g in groups.values()),
        "all_failed_groups": sum(not any(r["reward"] for r in g) for g in groups.values()),
        "mixed_groups": sum(0 < sum(r["reward"] for r in g) < len(g) for g in groups.values()),
    }


def run_bigmath_calibration(output_dir, parquet_path, cache_dir, commit):
    import pyarrow.parquet as pq
    from .sampling import SamplingConfig
    from .short_reasoning import CONCISE
    from .training import TrainConfig, Trainer, write_json

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    selected, rejected = select_examples(pq.read_table(parquet_path).to_pylist())
    setup = TrainConfig(
        model_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        enable_thinking=False,
    )
    write_json(output / "manifest.json", {
        "dataset": "SynthLabsAI/Big-Math-RL-Verified", "revision": REVISION,
        "selection": "source-balanced round robin, seeded hash order, unique parseable gold",
        "examples": selected, "rejected": rejected, "model": asdict(setup),
        "instruction": CONCISE, "group_size": 4, "cap": 2048,
        "temperature": 1, "alphas": [0, 0.5], "optimizer_steps": 0,
        "note": "Source-balanced diagnostic, not population accuracy or fair speed benchmark; sampler evaluates both contexts in both arms.",
    })
    commit()
    trainer = Trainer(setup, cache_dir)
    trainer.model.eval()
    eos = trainer.model.generation_config.eos_token_id
    extra_eos = tuple([eos] if isinstance(eos, int) else eos or [])
    eos_ids = set(extra_eos) | {trainer.tokenizer.eos_token_id}
    all_rows = {"student": [], "product": []}
    for offset in range(0, len(selected), 4):
        batch = selected[offset:offset + 4]
        expanded = [row for row in batch for _ in range(4)]
        pairs = [prompts(trainer.tokenizer, Example(r["id"], r["question"], r["answer"]), False, instruction=CONCISE) for r in expanded]
        student, teacher = map(list, zip(*pairs, strict=True))
        for arm, alpha in (("student", 0.0), ("product", 0.5)):
            start = perf_counter()
            results = trainer.sampler.generate(student, teacher, config=SamplingConfig(
                teacher_weight=alpha, temperature=1, top_k=None, top_p=None,
                max_new_tokens=2048, seed=42 + offset, extra_eos_token_ids=extra_eos,
            ))
            seconds = perf_counter() - start
            rows = []
            for index, (example, pair, result) in enumerate(zip(expanded, pairs, results, strict=True)):
                rows.append({
                    "arm": arm, "source": example["source"], "example_id": example["id"],
                    "question": example["question"], "answer": example["answer"],
                    "sample": index % 4, "student_prompt": pair[0], "teacher_prompt": pair[1],
                    **result.to_dict(),
                    "raw_text": trainer.tokenizer.decode(result.token_ids, skip_special_tokens=False),
                    "ended_with_eos": bool(result.token_ids and result.token_ids[-1] in eos_ids),
                    **verify_answer(result.text, example["answer"]),
                })
            write_json(output / f"{arm}-{offset:03d}.json", {"seconds": seconds, "rows": rows})
            all_rows[arm].extend(rows)
            summary = {name: {"overall": summarize(data), "by_source": {
                source: summarize([r for r in data if r["source"] == source])
                for source in sorted({r["source"] for r in data})
            }} for name, data in all_rows.items() if data}
            write_json(output / "summary.json", summary)
            commit()
            print(json.dumps({"event": "bigmath_batch", "offset": offset, "arm": arm, "seconds": seconds, "summary": summary[arm]["overall"]}), flush=True)
    assert trainer.step == 0
    return summary
