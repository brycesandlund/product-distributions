# Big-Math 4B A/B, 512 steps per arm

Submitted: 2026-09-17. No learning results claimed.

Initial Modal calls:
- A: `fc-01M2RXJCZ84K5PCTJCNPPZM92F`
- B: `fc-01M2RXJD4M785GD8NAX2W51BK9`

- Model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Separate A10 per arm, 4 CPU cores, 32GB host RAM. Fresh LoRA initialization;
  neither the hardware-benchmark adapter nor the 9B pilot is resumed.
- A: alpha=0 with student-only sampling. B: alpha=0.5 with the same current
  model under answer-privileged context. Both use beta=0 centered rewards,
  sampled student imitation loss, no clipping or importance correction.
- Non-thinking, concise calibrated instruction, temperature 1, no top-k/top-p,
  2,048 completion-token cap, EOS 248046 and 248044.
- Rank-16 LoRA, AdamW learning rate 2e-5, seed 42, four questions and four
  rollouts per question per update. 512 updates consume all 2,048 training
  questions exactly once, producing 8,192 training trajectories per arm.
- 256 held-out questions, one sampled student-only answer each, evaluated
  before training and every 64 updates. Fixed evaluation seeds/batch size 8.
- Dataset prepared by `scripts/prepare_bigmath_ab.py --large-4b` into
  `artifacts/bigmath-ab-4b-512-v1/`, uploaded to `/data/bigmath-ab-4b-512-v1/`.
  Pinned Big-Math revision, all sources retained, seeded source balancing,
  disjoint normalized-question hash partitions; calibration questions excluded.
- Local preflight verified every train/eval prompt: maximum 550 tokens, no
  overlong prompts; all 28 tests pass and `git diff --check` is clean.

## Execution and recovery

`train_small_ab_segment` executes at most 64 updates per invocation with a
12-hour timeout. It commits results after every step and checkpoints every 32
steps. On successful completion it verifies checkpoint reload, records
`progress.json`, and dispatches the next segment only if fewer than 512 steps
are complete. Baseline evaluation is not repeated at segment boundaries.
Optimizer state, adapters and RNG resume through Accelerate. There are no
automatic retries of failed segments and no continuation beyond 512 steps.

Results live under `/runs/bigmath-4b-512-A-20260917/` and
`/runs/bigmath-4b-512-B-20260917/` in the `product-distributions-results` volume.
Each segment has its own directory and the per-arm `progress.json` records the
last completed segment and the next call ID. A failed segment can be recovered
from its latest committed checkpoint into a new output directory.

The pilot baseline mismatch has not been conclusively diagnosed; separate
baseline results must be retained. The same seed is not a guarantee of
bit-identical GPU generation. This is a single-seed learning experiment, not
a robust multi-seed claim or a non-math retention evaluation. Hardware timings
suggest day-scale runtime; the account usage limit may interrupt the run.
