# Big-Math 2K A/B pilot — 2026-09-17

Status: submitted to Modal; learning results are not yet available.

## Matched setup

- Qwen/Qwen3.5-9B at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Non-thinking, the concise instruction from the length calibration, temperature
  1, no top-k/top-p, 2,048 generated-token cap. Recognize both tokenizer EOS
  248046 and model-generation EOS 248044.
- Student imitation loss, beta=0: reward minus within-question mean, without
  standard-deviation scaling. Fixed-length normalization, no clipping, no
  rollout importance correction. One fresh rollout batch per optimizer update.
- A: alpha=0 (ordinary student-policy reinforcement).
- B: alpha=0.5 (product rollouts with the current same model conditioned on the
  verified answer). Only question-context student log probabilities are trained.
- Rank-16 LoRA, learning rate 2e-5, seed 42, 32 optimizer steps, four questions
  and four samples per question per step. Each arm sees the same first 128
  training questions and generates 512 training trajectories.
- Student-only held-out evaluation before training and at steps 16 and 32;
  one sampled completion per held-out question, fixed seeds and batch size 8.
- Checkpoints every 8 steps; metrics and artifacts committed after every step.
- One H100 per arm. This is a short single-seed pilot, not a ceiling estimate
  or a statistical demonstration of better generalization. No unrelated-task
  retention suite is attached.

## Data

Big-Math-RL-Verified revision `c75d2f117cddfecb6bd08756e61e508e59732b21`.
All 11 sources retained; source-balanced selection rather than natural source
proportions. 2,048 prepared training questions and 44 evaluation questions.
Train/eval are separated by a normalized-question hash partition before
selection. All 128 earlier calibration questions are excluded. Answers must be
parseable by the verifier. No filtering on model success or length.

Preparation: `scripts/prepare_bigmath_ab.py`. Exact data, source labels,
exclusions, and A/B configs are in `artifacts/bigmath-ab-v1/` locally and
`/data/bigmath-ab-v1/` on the `product-distributions-results` Modal volume.
The held-out set is disjoint under normalized exact matching, not a claim of
semantic or pretraining decontamination.

## Submitted jobs

| Arm | Run directory under `/runs/` | Modal function call |
| --- | --- | --- |
| A | `bigmath-ab-2k-A-20260917` | `fc-01M2R5KYEWDSXBW5WDKC8E31CB` |
| B | `bigmath-ab-2k-B-20260917` | `fc-01M2R5KYKFRQ2ENV8X1G0SP6F7` |

The current sampler evaluates both contexts even for alpha=0. Consequently,
these jobs isolate rollout-distribution effects under the same implementation;
their elapsed-time comparison is not against an optimized student-only sampler.

Local verification before launch: 25 tests passed, including batched evaluation
prompt isolation and EOS handling; `git diff --check` passed.
