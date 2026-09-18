"""Prepare a fixed all-source A/B pilot, excluding calibration questions."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from product_distributions.bigmath_calibration import REVISION, select_examples
from product_distributions.data import example_id
from product_distributions.short_reasoning import CONCISE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    parser.add_argument("output")
    parser.add_argument("--large-4b", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = pq.read_table(args.parquet).to_pylist()
    calibration, _ = select_examples(rows)
    excluded = {r["id"] for r in calibration}
    pools = {"train": [], "eval": []}
    for row in rows:
        key = example_id(str(row["problem"]))
        if key in excluded:
            continue
        split = "eval" if int(hashlib.sha256(f"ab-v1:{key}".encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
        pools[split].append(row)
    manifest = {"dataset": "SynthLabsAI/Big-Math-RL-Verified", "revision": REVISION,
                "selection": "source-balanced, seeded hash order within disjoint hash partitions; calibration excluded",
                "calibration_excluded_ids": sorted(excluded), "splits": {}}
    for split, count in (("train", 2048), ("eval", 256 if args.large_4b else 44)):
        selected, rejected = select_examples(pools[split], count=count, seed=43)
        (output / f"{split}.jsonl").write_text("".join(
            json.dumps({k: r[k] for k in ("id", "question", "answer")}) + "\n" for r in selected))
        manifest["splits"][split] = {"examples": selected, "rejected": rejected}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    common = dict(model_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                  loss="imitation", beta=0, steps=32, prompts_per_step=4, group_size=4,
                  max_new_tokens=2048, max_prompt_tokens=2048, learning_rate=2e-5,
                  lora_rank=16, seed=42, enable_thinking=False, eval_every=16,
                  checkpoint_every=8, instruction=CONCISE, extra_eos_token_ids=[248044],
                  eval_batch_size=8)
    if args.large_4b:
        common.update(model_id="Qwen/Qwen3.5-4B",
                      model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                      steps=512, eval_every=64, checkpoint_every=32)
    for name, alpha in (("A", 0.0), ("B", 0.5)):
        (output / f"config-{name}.json").write_text(json.dumps({**common, "alpha": alpha}, indent=2))
    print(json.dumps({"output": str(output), "train": 2048, "eval": 256 if args.large_4b else 44, "calibration_excluded": len(excluded)}))


if __name__ == "__main__":
    main()
