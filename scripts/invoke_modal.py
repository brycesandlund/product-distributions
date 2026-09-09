"""Invoke the deployed product-distributions Modal app."""

from __future__ import annotations

import argparse
import json

import modal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("cache", "health", "generate", "batch", "dual-health", "dual-batch"),
        nargs="?",
        default="health",
    )
    parser.add_argument(
        "--question",
        default="What is 17 times 19? Answer with only the integer.",
    )
    parser.add_argument(
        "--privileged-context",
        default="The verified answer is 323.",
    )
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "cache":
        cache_model = modal.Function.from_name("product-distributions", "cache_model")
        print(cache_model.remote())
        return

    class_name = "DualQwenWorker" if args.command.startswith("dual-") else "QwenWorker"
    worker_cls = modal.Cls.from_name("product-distributions", class_name)
    worker = worker_cls.with_options(gpu=args.gpu)()
    if args.command in ("health", "dual-health"):
        result = worker.health.remote()
    elif args.command == "generate":
        result = worker.product_generate.remote(
            args.question,
            args.privileged_context,
            teacher_weight=args.teacher_weight,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=False,
        )
    else:
        result = worker.product_generate_batch.remote(
            [args.question] * args.batch_size,
            [args.privileged_context] * args.batch_size,
            teacher_weight=args.teacher_weight,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=False,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
