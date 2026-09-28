# Privileged self-teacher OPD: preparation and preflight

Date: 2026-09-28. Only data preparation and the short pilot are authorized;
the 512-step experiment has not been launched.

## Planned experiment

Configuration: `configs/opd_4b_privileged_512.json`. Fresh Qwen3.5-4B rank-16
LoRA, non-thinking, temperature 1, no sampling truncation, 2,048-token cap,
LR 2e-5. Full-vocabulary token-local reverse KL, alpha=0. Rollouts use only
the question; the same current model scores them with the answer in context,
with teacher gradients detached. No verifier reward weighting.

Sixteen distinct questions and one response each per update. The prepared
8,192-question training pool supports 512 updates without repeating questions.
It extends the original hash-partitioned, source-balanced training selection,
retains its first 2,048 questions in order, excludes calibration IDs, and copies
the existing 256-question holdout byte-for-byte. Dataset name on Modal:
`bigmath-opd-8192-v1`. Preparation saves a manifest and validation report.

## Bounded test

Isolated deployment: `modal_opd_preflight.py`, app
`product-distributions-opd-preflight`. A10, four CPU cores, 32 GiB host RAM,
one-hour timeout, no successor calls or automatic launch of full training.

The pilot uses a disposable adapter. First, a synthetic 2,048-token continuation
tests teacher scoring plus full-vocabulary student KL backward with finite
gradients, without an optimizer step. This is a memory test, not a learning
measurement. Then three actual 16-question updates record token counts, timings,
memory, gradients, and raw rollouts. Finally it saves and reloads a checkpoint
and checks logits match exactly. Benchmark adapters must not initialize the
full experiment.

Compare learning curves by cumulative generated training tokens; separately
report unique questions, teacher scoring overhead, and GPU time. Equal update
counts or generated-token counts do not imply equal compute cost.

## Preparation and initial attempt

Preparation passed: 8,192 unique training IDs, 256 held-out questions, 128
calibration IDs excluded, zero overlap, old training prefix preserved. Local
copy: `artifacts/bigmath-opd-8192-v1/`. Holdout SHA-256:
`22e639b8d6bc2a8054b84f32023ad763759406fb91fad3f9a7c8b06f4025abe5`.

Initial pilot `4b-privileged-opd-a10-20260928`, call
`fc-01M3MJ2BCWH9YPB6Q7HHAZTAE0`, failed in the forced 2K KL calculation before
any updates. A10 reported 22.06 GiB total; PyTorch had allocated 20.33 GiB and
requested another 1.89 GiB. The exact full-vocabulary loss now chunks token
positions into blocks of 128 and checkpoints KL intermediates for backward.
This does not truncate vocabulary or change the objective. Tests compare values
and gradients against unchunked KL at alpha 0, 0.5 and 1.

The first retry (`4b-privileged-opd-a10-20260928-chunked`) reused an old warm
container and repeated the original failure; it did not test the new code.
The isolated app was stopped and redeployed. The fresh pilot
`4b-privileged-opd-a10-20260928-chunked-cold`, call
`fc-01M3MJAD8HNTM14P0YBDMBH60V`, records its actual loss source in `summary.json`.
Its 2K synthetic backward passed at 16.65 GiB allocated / 17.80 GiB reserved.

## Successful pilot results

All three updates and checkpoint reload passed. Reload max logit error: **0**.
Elapsed time including setup, stress check and checkpointing: **622.68 seconds**
(10.38 minutes). No full training run was launched.

| Update | Generated tokens | Rollout seconds | Update seconds | Peak allocated GiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14,674 | 157.84 | 26.90 | 16.77 |
| 2 | 11,167 | 153.79 | 22.75 | 17.02 |
| 3 | 12,843 | 150.73 | 21.12 | 17.02 |

All updates had finite nonzero gradients and adapter changes. Each batch had
four capped responses out of sixteen. Reward scores are batch diagnostics,
not held-out learning measurements. Peak reserved memory was 21.74 GiB on the
first update and 18.86 GiB thereafter; allocator retry warnings occurred but
did not prevent completion. The log-probability recomputation diagnostic ranged
0.24–0.36 max absolute difference between batched rollout and single-sequence
training evaluation; full-vocabulary OPD recomputes distributions and does not
use these old sampled log probabilities in its loss.

A10 with 16 distinct questions/update is feasible for this tested workload.
This is not a guarantee for every longer prompt in the expanded dataset.
The measured update range, 172–185 seconds, implies roughly 24–26 hours for
512 updates if lengths stay similar, **excluding held-out evaluation and other
overhead**. Full-run evaluation every 32 updates adds material compute.

Artifacts on Modal: `hardware-benchmarks/4b-privileged-opd-a10-20260928-chunked-cold/`.
Local copy: `artifacts/opd-preflight-20260928/4b-privileged-opd-a10-20260928-chunked-cold/`.
Validation: 37 local tests passed, including chunked/un-chunked KL value and
gradient equivalence. Pilot checkpoint is disposable, not a run initialization.
