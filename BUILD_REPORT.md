# Training build verification — 2026-09-15

The Modal app `product-distributions-training` implements the agreed sampled-token
student-imitation and product-to-teacher OPD objectives. The initial teacher is
the same Qwen3.5-9B model under an answer-privileged context, with a detached
teacher forward. The model has 43,278,336 trainable rank-16 LoRA parameters.

## Hardware results

Actual runs on one NVIDIA H100 80GB, BF16, native Transformers and FLA kernels:

| Run | Rollouts/update | Output cap | Thinking | Updates | Peak allocated GiB | Reload logit error |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Student imitation, beta=1 | 2 | 64 | Off | 1 | 18.56 | 0 |
| Product-to-teacher OPD | 2 | 128 | Off | 1 | 18.68 | 0 |
| Student imitation, beta=0.5 | 2 | 512 | On | 2 | 20.28 | 0 |

All runs used alpha=0.5, temperature=1, no top-k/top-p truncation, no clipping,
and one optimizer update after accumulating all rollout losses. Both losses
produced finite nonzero gradients and measurable adapter updates.

The second 512-token update generated 1,024 shared output tokens in 26.67 seconds
(38.4 aggregate output tokens/second) and updated in 1.58 seconds. Initial updates
include substantial Triton compilation overhead (95–103 seconds in the short
tests). These are observations from a few prompts, not a comprehensive benchmark.

Maximum sampled student-logprob discrepancies between cached generation and
unbatched training recomputation were 0.080, 0.125, and 0.170 nats respectively.
Both paths use BF16; generation uses recurrent decoding while training uses
full-sequence kernels. Numerical equivalence is approximate, not bitwise.

Checkpoint verification perturbs all trainable adapter weights, reloads adapters,
optimizer and RNG through Accelerate, then checks identical evaluation logits.
The frozen base is not duplicated in the checkpoint.

A fresh Modal container also restored the two-update checkpoint, continued with
a fresh rollout batch at step 3, saved a new checkpoint, and passed exact logit
reload again. This exercises the deployed API and cross-container resume, not
just an in-process reload.
The resumed update peaked at 20.28 GiB and had a maximum logprob recomputation
discrepancy of 0.213 nats; the new checkpoint again reloaded with zero logit error.

## Data and tests

DeepMath revision: `5cf055d1fe3d7a2eb19719ac020211469736ae44`.
The prepared `deepmath-v1` subset has 128 training and 16 held-out questions,
split by deterministic question hash before selection. Seven unsupported gold
answers were excluded and listed in the dataset manifest. The separate
`deepmath-smoke` subset contains four training and one held-out question.

Thirteen local tests cover exact expected gradients of sampled OPD at alpha
0/0.4/1, teacher detachment, imitation endpoints, group weights, vocabulary
compatibility, prompt isolation, completion alignment, EOS, recorded logprobs,
and symbolic answer verification. Ruff and `git diff --check` pass.

These are systems smoke tests. The training rollouts reached their length caps
without verified final answers. No learning gain, ceiling improvement, or
retention improvement has been established. Before a sweep, calibrate response
length and difficulty so that reward-based training receives useful variation.
The stronger-teacher training option is implemented but was not GPU-tested in
this build; the earlier two-model inference test is separate evidence.

## Artifacts

Volume: `product-distributions-results`.

- `runs/imitation-20260915-231519`: 64-token imitation smoke.
- `runs/opd-20260915-231655`: 128-token OPD smoke.
- `runs/imitation-20260915-231844`: two-update thinking smoke.
- `runs/imitation-resume-smoke`: fresh-container continuation to step 3.

Local copies of metrics, reload checks, and data manifests are in `artifacts/`
(gitignored). See README for launch, resume, and artifact-download commands.
