"""Pinned DeepMath subsets, isolated prompts, and bounded answer verification."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answer: str


def example_id(question: str) -> str:
    return hashlib.sha256(" ".join(question.split()).encode()).hexdigest()


def last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 1
    for index in range(start + 7, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start : index + 1]
    return None


def parse_gold(answer: str):
    from math_verify import LatexExtractionConfig, parse

    boxed = last_boxed(answer) or "\\boxed{" + answer.strip().strip("$") + "}"
    result = parse(
        boxed,
        extraction_config=[LatexExtractionConfig()],
        fallback_mode="no_fallback",
        parsing_timeout=2,
    )
    if not result:
        raise ValueError(f"Unparseable gold answer: {answer[:100]!r}")
    return result


def verify_answer(text: str, answer: str) -> dict:
    from math_verify import LatexExtractionConfig, parse, verify

    gold = parse_gold(answer)
    # Only the final answer is rewarded; numbers in the reasoning do not count.
    boxed = last_boxed(text)
    if boxed is None:
        return {"reward": 0.0, "status": "missing_boxed_answer"}
    try:
        predicted = parse(
            boxed,
            extraction_config=[LatexExtractionConfig()],
            fallback_mode="no_fallback",
            parsing_timeout=2,
        )
        if not predicted:
            return {"reward": 0.0, "status": "unparseable_prediction"}
        correct = bool(verify(gold, predicted, timeout_seconds=2))
        return {
            "reward": float(correct),
            "status": "correct" if correct else "incorrect",
        }
    except Exception as error:  # noqa: BLE001 -- malformed generated math must not crash training
        return {"reward": 0.0, "status": "verifier_error", "error": str(error)}


def prompts(tokenizer, example: Example, enable_thinking: bool, *, instruction=None):
    if instruction is None:
        instruction = "Solve the problem. Explain your reasoning and put your final answer in \\boxed{}."
    student_content = f"{example.question}\n\n{instruction}"
    teacher_content = (
        student_content
        + "\n\nPrivileged information: the verified final answer is "
        + example.answer
        + ". Use it to guide a valid derivation; do not mention this information."
    )

    def chat(content):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    return chat(student_content), chat(teacher_content)


def prepare_deepmath(output: str, train_size=128, eval_size=16, seed=42):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        manifest_path = destination / "manifest.json"
        if manifest_path.exists() and all(
            (destination / f"{split}.jsonl").exists() for split in ("train", "eval")
        ):
            existing = json.loads(manifest_path.read_text())
            if all(
                existing.get(key) == value
                for key, value in {
                    "train_size": train_size,
                    "eval_size": eval_size,
                    "seed": seed,
                }.items()
            ):
                return existing
        raise FileExistsError(f"Refusing to overwrite prepared data: {destination}")
    repo = "zwhe99/DeepMath-103K"
    revision = HfApi().dataset_info(repo).sha
    rows = load_dataset(repo, revision=revision, split="train", streaming=True)
    splits = {"train": [], "eval": []}
    seen = set()
    excluded = []
    for row in rows:
        key = example_id(row["question"])
        if key in seen:
            continue
        seen.add(key)
        # Split before selection; duplicates of a question cannot cross the split.
        bucket = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) % 10
        split = "eval" if bucket == 0 else "train"
        limit = eval_size if split == "eval" else train_size
        if len(splits[split]) >= limit:
            continue
        try:
            parse_gold(row["final_answer"])
        except ValueError as error:
            excluded.append({"id": key, "reason": str(error)})
            continue
        splits[split].append(asdict(Example(key, row["question"], row["final_answer"])))
        if len(splits["train"]) == train_size and len(splits["eval"]) == eval_size:
            break
    if len(splits["train"]) != train_size or len(splits["eval"]) != eval_size:
        raise ValueError("Dataset exhausted before subset was filled")
    for split, examples in splits.items():
        (destination / f"{split}.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in examples)
        )
    manifest = {
        "dataset": repo,
        "revision": revision,
        "seed": seed,
        "train_size": train_size,
        "eval_size": eval_size,
        "selection": "first unique questions with parseable gold in deterministic hash split",
        "excluded_unparseable_gold": excluded,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_examples(path: str):
    examples = [
        Example(**json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not examples or len({e.id for e in examples}) != len(examples):
        raise ValueError("Dataset must be nonempty with unique example IDs")
    for e in examples:
        parse_gold(e.answer)
    return examples
