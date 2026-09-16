"""Launch inference-only length calibration on the deployed Modal app."""

import argparse
import json
from datetime import UTC, datetime

import modal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name")
    parser.add_argument("--data-name", default="deepmath-v1")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--questions", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gpu", default="H100")
    args = parser.parse_args()
    name = args.run_name or datetime.now(UTC).strftime("lengths-%Y%m%d-%H%M%S")
    result = (
        modal.Function.from_name("product-distributions-training", "calibrate_lengths")
        .with_options(gpu=args.gpu)
        .remote(
            name,
            args.data_name,
            args.lengths,
            args.questions,
            args.group_size,
            args.alpha,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
