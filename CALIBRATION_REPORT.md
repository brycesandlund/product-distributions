# Inference-only length calibration — September 16, 2026

**Neither 2,048 nor 4,096 tokens produced a usable reward signal in this small
sample. No training updates or learning-method sweep were run.**

## Setup

- Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Same model with question-only student context and answer-privileged teacher
  context; product alpha=0.5. Existing training prompt and initial LoRA setup.
- Thinking enabled, temperature one, no top-k/top-p truncation, one H100.
- First four training questions from the pinned `deepmath-v1` subset, four
  rollouts per question. No held-out evaluation questions used for selection.
- Same seeds across caps; only 1/16 pairs had identical first-2K token prefixes.
  These are matched-seed runs, not exact paired trajectory extensions.
- A separate alpha=0 control used four rollouts on the first question at 4K.
  It still runs through the paired sampler, so its timing is not a single-context
  inference throughput benchmark.
- Zero backward passes, zero optimizer steps, no new model checkpoints.

## Results

| Configuration | Cap | Rollouts | EOS completions | Verifier successes | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product alpha=0.5, four questions | 2,048 | 16 | 0 | 0 | 18.91 |
| Product alpha=0.5, four questions | 4,096 | 16 | 0 | 0 | 19.66 |
| Question-only alpha=0, one question | 4,096 | 4 | 0 | 0 | 19.66 |

Every rollout reached its cap. All groups were all-failed; none provided
nonzero centered reward advantages. Guided generation took 581.03 seconds at
2K and 1,010.76 seconds at 4K. Aggregate throughput was 56.4 and 64.8 tokens/sec
respectively, including slower first batches; subsequent guided batches were
roughly 71–72 tokens/sec. Control generation took 420.94 seconds. These are
small cold/warm-mixed measurements, not controlled throughput comparisons.

Peak allocator-reserved memory was 26.85 GiB at 2K, 40.44 GiB at guided 4K, and
51.99 GiB for the control. Allocated and reserved memory are different metrics;
neither is a measurement of future training/backward-pass memory.

## Inspection and interpretation

Inspected traces often reached the right mathematical result but continued
re-deriving it, checking alternate methods, or questioning the problem wording
instead of finishing. The same pattern appeared without product guidance on
the control question. This does not establish that guidance has no effect on
length across the dataset.

Some raw boxed-text detections were repetitions of the instruction's empty
`\boxed{}` placeholder, not answers. The verifier rejected them correctly;
boxed-text presence alone is not a completion or correctness metric.

Four fixed questions are too few to estimate dataset-wide performance, and
their repeated rollouts are not independent questions. The inference-only
results also do not establish long-sequence training feasibility.

**Recommendation:** keep the learning sweep paused. Next, sanity-check the
decode path against native generation and calibrate larger caps (8K/16K) on a
bounded sample. Do not assume those caps will suffice. A thinking-disabled or
more concise prompt would be a separate setting, not a silent change to this
calibration. None of these follow-up jobs has been launched.

## Reproduction and artifacts

```bash
uv run python scripts/calibrate_modal.py --lengths 2048 4096 --questions 4 --group-size 4
uv run python scripts/calibrate_modal.py --lengths 4096 --questions 1 --group-size 4 --alpha 0
```

Results-volume directories:

- `/calibration/lengths-9b-self-20260916`
- `/calibration/lengths-9b-question-only-20260916`

Each contains configuration, full rollout text and token IDs, and summary
metrics. Local copies are in `artifacts/length-calibration-*.json*` (git-ignored).
The calibration entry point and summary have automated coverage; all 20 tests
passed, along with lint and whitespace checks.
