"""Launch a deployed training job; download artifacts with `modal volume get`."""

import argparse
import json
from datetime import UTC, datetime

import modal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "train"])
    p.add_argument("--data-name", default="deepmath-v1")
    p.add_argument("--train-size", type=int, default=128)
    p.add_argument("--eval-size", type=int, default=16)
    p.add_argument("--config", help="Local JSON file with TrainConfig overrides")
    p.add_argument("--run-name")
    p.add_argument("--resume", help="Checkpoint directory on /results volume")
    p.add_argument(
        "--retention-path", help="Optional verifier-compatible JSONL on /results"
    )
    p.add_argument("--gpu", default="H100")
    args = p.parse_args()
    if args.command == "prepare":
        result = modal.Function.from_name(
            "product-distributions-training", "prepare_data"
        ).remote(args.data_name, args.train_size, args.eval_size)
    else:
        from pathlib import Path

        config = json.loads(Path(args.config).read_text()) if args.config else {}
        name = args.run_name or datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
        result = (
            modal.Function.from_name("product-distributions-training", "train")
            .with_options(gpu=args.gpu)
            .remote(config, name, args.data_name, args.resume, args.retention_path)
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
