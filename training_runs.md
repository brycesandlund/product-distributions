# Training runs and experimental results

Snapshot date: 2026-09-28. All reported accuracies below are verifier scores,
not human assessments of derivation quality. No new GPU jobs were launched to
prepare this overview.

## Where the data lives

Full local archive: `artifacts/modal-results-20260928/`, downloaded from the
Modal volume `product-distributions-results`. Remote files have not been removed.
Download verified: **2,110 files, 24.86 GB (23.15 GiB)**, with every local file
matching the remote byte size. This is a completeness/size check, not a checksum check.
The archive includes data, calibration results, training rollouts, metrics,
evaluation responses, adapters, optimizer state, and RNG state. It is gitignored;
this overview and the experiment code/configs are tracked separately.

Paths below are relative to that archive. `download_inventory.json` records the
remote file inventory and local byte-size verification. Within a run:

- `progress.json`: latest completed segment/checkpoint and, when applicable, next call.
- `segment-*/eval-*.json`: accuracy and individual generated evaluation answers.
- `segment-*/metrics.jsonl`: per-update rewards, loss, gradient/update norms,
  generation/update timings, token counts, and memory measurements.
- `segment-*/rollouts-*.json`: training responses, verifier grades and weights.
- `segment-*/checkpoint-*`: student adapters, optimizer, RNG and resume metadata.
- `config.json`, `runtime.json`, `data_manifest.json`: reproducibility metadata.

## Main result

September 28 run `bigmath-4b-opd-512-20260928` was **stopped early for collapse**.
Ordinary full-vocabulary OPD with a privileged 4B self-teacher (alpha=0), A10,
16 distinct questions per update, 8,192-question pool, planned 512 updates:

| Update | Cumulative generated training tokens | Held-out accuracy | Mean eval tokens |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 63.28% (162/256) | 876.07 |
| 32 | 427,160 | 0.78% (2/256) | 1,817.67 |

At step 32, 169/256 evaluation responses reached the cap. Inspected samples
show long generic mathematical prose, unlike B's short answer-only collapse.
The fixed evaluation question IDs match baseline. This is evidence of failure
under these settings, not a causal diagnosis or a general verdict on OPD.
The monitor stopped the dedicated app after 37 committed updates / 579,181
generated training tokens; the last saved checkpoint is step 32. No step-37
evaluation exists. These token counts exclude evaluation and the separate pilot.
Artifacts including the checkpoint are downloaded to
`artifacts/opd-stopped-20260928/bigmath-4b-opd-512-20260928/`.
See `OPD_512_RUN.md` for stop details and original configuration.

Ordinary RL improved the 4B student's held-out math accuracy. Product rollouts
using the same model with a privileged answer initially improved accuracy, then
collapsed into short, mostly incorrect answer-only responses. Replacing that
teacher with a frozen question-only 9B model avoided this collapse through the
192 steps tested and reached 70.7% accuracy. Its advantage over ordinary RL at
the same step was only two questions out of 256; this is not a decisive win.

### Shared setup

- Student: Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Non-thinking, concise prompt, temperature 1, no top-k/top-p, 2,048-token output cap.
- Rank-16 LoRA; AdamW LR 2e-5; seed 42. Four questions × four rollouts per update.
- Sampled student-imitation loss, with beta=0: reward minus within-question mean.
  No standard-deviation scaling, ratio clipping, or off-policy importance correction.
  Fixed-length loss normalization. One optimizer update per fresh rollout batch.
- Big-Math-RL-Verified revision `c75d2f117cddfecb6bd08756e61e508e59732b21`.
  2,048 prepared training questions, all sources retained using source-balanced
  selection, not the dataset's natural source proportions. Calibration questions excluded.
- 256 held-out questions, split by normalized-question hash before selection;
  evaluation always uses student-only, question-only prompts. One sampled answer
  per question, fixed evaluation seeds and batch size 8, before training and every 64 steps.
- Checkpoints every 32 steps, recovery-safe 64-step segments. No separate
  non-math retention suite and no multi-seed replication.

### Held-out accuracy by update

| Step | A: ordinary RL, alpha=0 | B: privileged self-teacher, alpha=0.5 | C: frozen question-only 9B, alpha=0.5 |
| ---: | ---: | ---: | ---: |
| 0 | 63.28% (162/256) | 63.28% (162/256) | 62.50% (160/256) |
| 64 | 66.02% (169/256) | 67.58% (173/256) | 65.23% (167/256) |
| 128 | 67.58% (173/256) | 67.19% (172/256) | 70.70% (181/256) |
| 192 | 69.92% (179/256) | 62.11% (159/256) | 70.70% (181/256) |
| 256 | 69.14% (177/256) | 17.58% (45/256) | — |
| 320 | 68.36% (175/256) | 17.58% (45/256) | — |
| 384 | 68.36% (175/256) | 16.02% (41/256) | — |
| 448 | 70.70% (181/256) | 14.06% (36/256) | — |
| 512 | 69.53% (178/256) | 13.67% (35/256) | — |

These are single-run measurements. C's baseline differs by two questions despite
matched student revision and seeds; execution hardware differs and numerical
reproducibility has not been conclusively diagnosed. Do not treat different
baseline scores as evidence of different intended student initialization.
The best observed checkpoint is selected using this holdout, not an independent test set.

### A: ordinary RL

Path: `runs/bigmath-4b-512-A-20260917/`.

- One A10, 4 CPU cores, 32GB host RAM. Alpha=0 evaluates only the student context.
- Completed 512 steps. Final gain: **+6.25 percentage points**; best observed
  checkpoint: step 448, **+7.42 points**. Final checkpoint reload had zero logit error.
- Mean evaluation length: 876 tokens initially, 632 at the final checkpoint.
- Recorded generation/update time: **22.16 hours including replayed updates**;
  21.36 hours when counting one record per step. Excludes evaluation, loading,
  verification and other overhead; this is not total billed runtime.
- Initial execution reached step 209, then a restarted invocation failed with
  `FileExistsError` on an existing output directory. Recovery resumed checkpoint 192.
  The old `segment-000256/` contains abandoned steps 193–209. **Exclude these
  from the learning curve**; use the resumed `segment-000256-attempt-*` records.
  Retain abandoned compute when estimating total resources consumed.

### B: moving privileged-context self-teacher

Path: `runs/bigmath-4b-512-B-20260917/`.

- One A10, 4 CPU cores, 32GB host RAM. Student and teacher share current weights;
  only the teacher context receives the verified answer. No separate teacher loss.
- Completed 512 steps; final checkpoint reload passed.
- Mean evaluation response length fell from **724 tokens at step 128 to 9 at
  step 256**. Outputs commonly became only a boxed answer. Final accuracy was
  13.67%, while mean privileged-product rollout reward over the last 64 updates
  remained 80.57%.
- This is consistent with a privileged-information shortcut and failed transfer
  to question-only inference. It does not isolate the exact causal mechanism or
  demonstrate catastrophic forgetting on unrelated tasks.
- Recorded generation/update time: **11.26 hours**, excluding other overhead.
  Its low cost is partly explained by collapsed short responses.

### C: frozen stronger teacher, no privileged answer

Path: `runs/bigmath-4b-teacher9b-192-20260921/`.
Config: `configs/teacher9b_192.json`.

- One L40S, 4 CPU cores, 64GB host RAM. Frozen Qwen3.5-9B teacher revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Neither context receives the answer.
- Fresh 4B student; no pilot or hardware-benchmark adapters reused. Same first
  768 training questions as A/B through step 192.
- Completed all 192 steps and stopped; final checkpoint reload passed.
- Final gain: **+8.20 percentage points**, 21 additional correct answers.
  No answer-only collapse through this horizon: final mean response length
  884 tokens; 51/256 evaluation responses hit the cap.
- Recorded generation/update time: **11.02 hours**, excluding evaluation/setup.
- Relative to B, both teacher identity and access to privileged answers change;
  this is not a one-variable ablation. General retention remains unmeasured.

## Earlier 9B training pilot

Paths: `runs/bigmath-ab-2k-A-20260917/` and `runs/bigmath-ab-2k-B-20260917/`.

Qwen3.5-9B non-thinking, 32 updates × four questions × four samples, 2K cap,
rank-16 LoRA, LR 2e-5. Same alpha/beta choices as A/B above, but only a
44-question holdout. Both completed and passed checkpoint reload verification.

| Step | A: ordinary RL | B: privileged product |
| ---: | ---: | ---: |
| 0 | 75.00% (33/44) | 77.27% (34/44) |
| 16 | 70.45% (31/44) | 68.18% (30/44) |
| 32 | 65.91% (29/44) | 72.73% (32/44) |

This pilot did not show improvement. The baseline mismatch affected 23 response
texts and three correctness outcomes, despite matching model/data/settings.
A had 12 zero-gradient steps; B had 15. Uniform-reward groups contribute zero
centered advantage, though Adam momentum can still move parameters on such steps.
Both peaked at 26.41 GiB allocated VRAM. Recorded rollout/update time was
70.3 minutes for A and 64.1 minutes for B. At that point alpha=0 still computed
the unused teacher context; it is not a fair optimized ordinary-RL speed baseline.

## Calibration before training

### Big-Math, 128 source-balanced questions

Path: `bigmath-calibration/bigmath-128-nonthinking-20260916/`.
9B, non-thinking, concise prompt, four samples per question per arm, 2K cap,
no optimizer updates. All 11 source labels represented.

| Metric | Student alpha=0 | Privileged product alpha=0.5 |
| --- | ---: | ---: |
| Correct | 314/512 (61.33%) | 332/512 (64.84%) |
| Truncated | 145/512 (28.32%) | 144/512 (28.13%) |
| Mean output tokens | 938.81 | 927.83 |
| Mixed-success question groups | 41/128 | 32/128 |

All verifier-correct responses also completed with EOS. Hard-source truncation
was substantial, particularly AMC/AIME, AoPS and OmniMath. Generation took
121.73 minutes total on H100, excluding setup/verification/storage overhead.
Both arms computed both contexts in this older calibration implementation.

### Prompt and thinking-mode checks

Paths: `calibration/`, `native-inspection/`, `short-reasoning/`.
See `CALIBRATION_REPORT.md`, `NATIVE_INSPECTION_REPORT.md`, and
`SHORT_REASONING_REPORT.md` for full protocols.

- Original thinking-enabled DeepMath calibration: all 32 rollouts at the tested
  2K/4K budgets hit the cap, with no correct answers.
- Native Transformers `generate()` check: all four 4K responses also hit the
  cap, so the product sampler alone did not explain the long traces.
- Concise non-thinking check on two DeepMath and two GSM8K questions: all four
  student and all four product responses completed correctly; mean lengths
  408 and 351.5 tokens respectively. Too small to establish population accuracy.
- Concise thinking-enabled product at 8K on those same questions: two completed
  correct answers; three verifier-correct if counting a capped trace containing
  an answer. This motivated separating correctness from clean completion.

## Hardware and systems tests

September 28 addition: the 4B privileged self-teacher **ordinary full-vocabulary
OPD** preflight passed on A10 after chunking/checkpointing KL intermediates to
fix a 2K-token OOM. Three updates, 16 distinct questions each, took 172–185
seconds/update and peaked at 17.02 GiB allocated. Checkpoint reload error was
zero. The dataset was expanded to 8,192 unique training questions with the
original holdout unchanged. This is a feasibility test, not a learning result;
the subsequent 512-step OPD launch is documented in `OPD_512_RUN.md`.
See `OPD_PREFLIGHT.md` for results,
failed-attempt provenance, and artifact locations outside the earlier archive.

Hardware tests use very small repeated prompt sets and disposable adapters.
They establish execution/memory feasibility, not learning efficacy.

| Configuration | Hardware | Measured result |
| --- | --- | --- |
| 4B self-teacher | A10 | 15.5 min total for four updates; warm student/product updates 170/199 sec; peak allocated 16.89 GiB |
| 4B self-teacher | L4 | 27.9 min total; warm student/product updates 319/362 sec; peak allocated 16.89 GiB |
| 4B + frozen question-only 9B | L40S | 15.4 min for three updates plus setup; warm updates about 278 sec; peak allocated 34.41 GiB, reserved 43.75 GiB |
| 9B + frozen 27B, sampled OPD | H100 | Two 512-token rollouts, one update; peak 71.63 GiB; reload passed |
| 9B + frozen 27B, full-vocabulary OPD | H100 | Two 512-token rollouts, one update; peak 72.58 GiB; reload passed |

The first L40S launch (`4b-student-9b-teacher-l40s-20260918`) failed during
container import because `modal_training` was missing from the packaged image.
It produced no benchmark results and the pending call was cancelled. The
successful retry is `hardware-benchmarks/4b-student-9b-teacher-l40s-20260920-retry/`.
4B A10/L4 tests are under the corresponding `qwen35-4b-*-20260917` directories.

Earlier training-system smoke tests also validated sampled imitation, sampled
OPD, full-vocabulary OPD, and cross-container checkpoint resume. Short capped
traces generally had zero reward; those tests should not be presented as
learning gains. Details and individual run names are in `BUILD_REPORT.md`.

## Interpretation and remaining questions

1. Ordinary RL is a successful baseline on this held-out math slice.
2. Privileged moving-self-teacher product imitation has a demonstrated collapse
   failure mode under the settings tested.
3. Frozen question-only 9B guidance is promising through 192 steps but has not
   established a statistically robust or compute-matched advantage over RL.
4. These experiments do not measure general instruction-following retention,
   non-math generalization, semantic train/test decontamination, or multi-seed variance.
5. Timings are measured compute phases, not invoices. Accuracy-selected best
   checkpoints and different model/hardware configurations require cautious comparisons.
