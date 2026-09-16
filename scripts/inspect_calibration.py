"""Render complete saved transcripts with the pinned tokenizer's chat template."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from product_distributions.data import Example, prompts


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "artifacts/length-calibration-config.json").read_text())
    setup = config["training_config_for_model_setup_only"]
    tokenizer = AutoTokenizer.from_pretrained(
        setup["model_id"], revision=setup["model_revision"]
    )
    rows = [
        json.loads(line)
        for line in (root / "artifacts/length-calibration-rollouts.jsonl")
        .read_text()
        .splitlines()
    ]
    selected = [r for r in rows if r["max_new_tokens"] == 4096 and r["sample"] == 0][:2]
    output = root / "artifacts/prompt-inspection"
    output.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(selected, 1):
        student, teacher = prompts(
            tokenizer,
            Example(row["example_id"], row["question"], row["gold_answer"]),
            True,
        )
        response = tokenizer.decode(row["token_ids"], skip_special_tokens=False)
        text = (
            f"# Saved calibration sample {index}\n\n"
            f"Model: `{setup['model_id']}`; revision: `{setup['model_revision']}`.\n\n"
            "Product alpha=0.5, temperature=1, no top-k/top-p, thinking enabled.\n"
            "The response below is ONE shared product-sampled continuation, not an\n"
            "independent response from either prompt. Full rendered prompts and all\n"
            "generated tokens are shown, including special tokens. No text is omitted.\n\n"
            f"Verifier: `{row['status']}`; tokens: {row['num_tokens']}; "
            f"EOS: {row['ended_with_eos']}.\n\n"
            "## Student prompt (question only)\n\n```text\n" + student + "\n```\n\n"
            "## Teacher prompt (privileged answer)\n\n```text\n" + teacher + "\n```\n\n"
            "## Full product-sampled response\n\n```text\n" + response + "\n```\n\n"
            "End of saved response: stopped at the 4,096-token cap, not EOS.\n"
        )
        (output / f"sample-{index}.md").write_text(text)
        print(output / f"sample-{index}.md")
    (output / "selected.json").write_text(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
