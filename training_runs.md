# Training runs and experimental results

Snapshot date: 2026-10-06. All reported accuracies below are verifier scores,
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

Arms D and E were archived separately after stopping:

- D: `artifacts/opd-stopped-20260928/bigmath-4b-opd-512-20260928/`.
- E: `artifacts/frozen-opd-stopped-20260929/bigmath-4b-frozen-opd-512-20260928/`.
  Download verified: 286 files / 2,687,178,310 bytes; all local sizes match remote.

Unless an explicit archive is given, paths below are relative to the original
archive. `download_inventory.json` records the
remote file inventory and local byte-size verification. Within a run:

- `progress.json`: latest completed segment/checkpoint and, when applicable, next call.
- `segment-*/eval-*.json`: accuracy and individual generated evaluation answers.
- `segment-*/metrics.jsonl`: per-update rewards, loss, gradient/update norms,
  generation/update timings, token counts, and memory measurements.
- `segment-*/rollouts-*.json`: training responses, verifier grades and weights.
- `segment-*/checkpoint-*`: student adapters, optimizer, RNG and resume metadata.
- `config.json`, `runtime.json`, `data_manifest.json`: reproducibility metadata.

## Main result

**A (ordinary RL)** improved held-out math accuracy. **B (product imitation with
a moving privileged self-teacher)** initially improved, then collapsed into
short answer-only responses. **C (product imitation with a frozen question-only
9B teacher)** avoided that collapse through 192 updates and reached 70.7%;
its advantage over A at the same step was only two questions out of 256.

The two subsequent experiments tested full-vocabulary on-policy self-distillation
(OPSD), using the same model with privileged answer context as teacher:
**D (moving privileged self-teacher)** collapsed by step 32, while **E (frozen
initial privileged self-teacher)** avoided that early collapse but showed no
clear learning gain and was stopped by the user after 205 updates. C completed
its original 192-update budget; on September 29 the user authorized 192 more
updates from checkpoint 192 (384 total). That continuation subsequently
degenerated and was stopped after 296 committed updates; see C below.
**F (ordinary OPD with frozen question-only 9B teacher)** completed 192 updates,
improving from 62.50% to 67.58% without gradient explosion. All A–F runs are
stopped or finished. A separate C replay from checkpoint 160 through 192 also
completed without comparable degeneration, reaching 72.27%; it is a diagnostic
branch, not a replacement for C's original learning curve. C's inference-only
cache diagnostic also completed; see the diagnostic results below.

**G (product imitation with frozen privileged self-teacher)** initially improved
to 66.80% at 64, then collapsed to **0.39% at 256**. Stopped after that evaluation
and checkpoint on October 6; all A–G runs are now stopped or finished.

### Shared setup

- Student: Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Non-thinking, concise prompt, temperature 1, no top-k/top-p, 2,048-token output cap.
- Rank-16 LoRA; AdamW LR 2e-5; seed 42. Sixteen rollouts per update.
- A–C: four questions × four rollouts, sampled student-imitation loss,
  with beta=0: reward minus within-question mean.
  No standard-deviation scaling, ratio clipping, or off-policy importance correction.
  Fixed-length loss normalization. One optimizer update per fresh rollout batch.
- Big-Math-RL-Verified revision `c75d2f117cddfecb6bd08756e61e508e59732b21`.
  A–C use 2,048 prepared training questions, all sources retained using source-balanced
  selection, not the dataset's natural source proportions. Calibration questions excluded.
- 256 held-out questions, split by normalized-question hash before selection;
  evaluation always uses student-only, question-only prompts. One sampled answer
  per question, fixed evaluation seeds and batch size 8, before training and every
  64 steps for A–C/F, every 32 for D–E.
- D–E: 16 distinct questions × one rollout; expanded 8,192-question pool,
  original training prefix retained and identical holdout. Full-vocabulary
  reverse KL to the privileged teacher, alpha=0 (student-only rollouts), no
  reward weighting or clipping. Teacher branch detached; D uses current shared
  weights, E disables LoRA for teacher forwards to expose the frozen initial base.
- Checkpoints every 32 steps, recovery-safe 64-step segments. No separate
  non-math retention suite and no multi-seed replication.
- F: the same 16-question, expanded-pool setup as D/E, but a separate frozen
  question-only 9B teacher; alpha=0, full-vocabulary reverse KL, no reward
  weighting or clipping.

### Held-out accuracy by update

A: ordinary RL (alpha=0); B: privileged moving-self-teacher product imitation
(alpha=0.5); C: frozen question-only 9B product imitation (alpha=0.5);
D/E: privileged-self-teacher OPSD (alpha=0), moving/frozen respectively.
F: frozen question-only 9B OPD (alpha=0).
G: frozen privileged-self-teacher product imitation (alpha=0.5).
An em dash means no evaluation at that step, not zero accuracy.

| Step | A: RL | B: product, self | C: product, 9B | D: OPSD, moving | E: OPSD, frozen | F: OPD, 9B | G: product, frozen self |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 63.28% (162/256) | 63.28% (162/256) | 62.50% (160/256) | 63.28% (162/256) | 63.28% (162/256) | 62.50% (160/256) | 63.28% (162/256) |
| 32 | — | — | — | 0.78% (2/256) | 63.67% (163/256) | — | — |
| 64 | 66.02% (169/256) | 67.58% (173/256) | 65.23% (167/256) | — | 61.72% (158/256) | 65.23% (167/256) | 66.80% (171/256) |
| 96 | — | — | — | — | 61.72% (158/256) | — | — |
| 128 | 67.58% (173/256) | 67.19% (172/256) | 70.70% (181/256) | — | 61.72% (158/256) | 67.19% (172/256) | 61.72% (158/256) |
| 160 | — | — | — | — | 64.45% (165/256) | — | — |
| 192 | 69.92% (179/256) | 62.11% (159/256) | 70.70% (181/256) | — | 63.67% (163/256) | 67.58% (173/256) | 60.94% (156/256) |
| 256 | 69.14% (177/256) | 17.58% (45/256) | 0.00% (0/256)* | — | — | — | 0.39% (1/256) |
| 320 | 68.36% (175/256) | 17.58% (45/256) | — | — | — | — | — |
| 384 | 68.36% (175/256) | 16.02% (41/256) | — | — | — | — | — |
| 448 | 70.70% (181/256) | 14.06% (36/256) | — | — | — | — | — |
| 512 | 69.53% (178/256) | 13.67% (35/256) | — | — | — | — | — |

These are single-run measurements. *C's step-256 result is from the restarted continuation
attempt; the first attempt did not publish a completed step-256 evaluation.
Its failure mechanism is unresolved, so this is not a clean algorithm comparison.

C's baseline differs by two questions despite
matched student revision and seeds; execution hardware differs and numerical
reproducibility has not been conclusively diagnosed. Do not treat different
baseline scores as evidence of different intended student initialization.
The best observed checkpoint is selected using this holdout, not an independent test set.
Equal update counts are not equal question exposure: D–F see four times as many
distinct training questions per update as A–C. Token counts, evaluation cadence,
and hardware also matter for efficiency comparisons.

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
- Completed the original 192 steps; checkpoint reload passed. On September 29,
  continuation to 384 total steps was submitted from checkpoint 192, preserving
  optimizer/RNG and all hyperparameters. Config: `configs/teacher9b_384.json`.
  Next 768 training questions, same holdout, evaluations at 256/320/384.
  Call `fc-01M3R9GWWKG6B21P2FS23DAZ2A`; details in `TEACHER9B_192_RUN.md`.
- Original step-192 gain: **+8.20 percentage points**, 21 additional correct answers.
  No answer-only collapse through this horizon: final mean response length
  884 tokens; 51/256 evaluation responses hit the cap.
- Recorded generation/update time: **11.02 hours**, excluding evaluation/setup.
- Relative to B, both teacher identity and access to privileged answers change;
  this is not a one-variable ablation. General retention remains unmeasured.

#### C continuation: degeneration and interrupted/replayed segment

- First attempt `segment-000256-attempt-9066c8f31fc8` recorded steps 193–256,
  but its latest committed checkpoint was 224; no completed eval-256 file.
- The same submitted Modal call returned a second attempt,
  `segment-000256-attempt-5d3452880ef2`, starting at 225 from checkpoint 224.
  The restart trigger is not established from saved artifacts. Steps 225–256
  are replayed work, not additional unique updates; do not double-count them
  in a learning curve (retain their cost in compute accounting).
- Broader text inspection corrected the earlier marker-based onset of 216:
  unmistakable multilingual/repetitive degeneration was present by rollout 205
  (generated using weights after update 204). Step 216 was merely the first
  occurrence of the specific Korean token previously tracked; 13/16 rollouts
  at step 238 contained that token. A malformed response already received
  reward 1 and weight +0.75 at step 206.
- Screening all 3,072 original responses found occasional language slips and
  repetitive mathematical reasoning, but no comparable unrelated word streams.
  The first resumed updates looked reasonable: step 193 gradient/update norms
  were 0.0557/0.0412. Gradient norm reached 0.2521 at 197 (original maximum
  0.2124), 1.2412 at 230, and 13.6834 at 238. Thus moderate growth preceded
  overt corruption, while the largest explosions followed it. This chronology
  does not identify the initiating cause. Step 205 had zero advantage/gradient,
  but Adam momentum still produced an adapter update norm of 0.0626.
- Restarted attempt: step-256 held-out accuracy **0/256**, all responses hit
  2,048 tokens; inspected outputs are repetitive gibberish. Checkpoint 256
  passed its reload check with zero logit error, which does not establish health.
- Successor `segment-000320-attempt-3b55e0f6ce26` recorded steps 257–296 with
  zero reward/gradient and all 16 responses capped on every update. Nonzero
  shrinking adapter updates persisted through Adam momentum. Last saved
  checkpoint is 288; original checkpoint 192 remains preserved remotely.
- Stopped September 30. This stop was issued before the user's read-only-first
  clarification arrived; no subsequent restart was performed. Remote progress
  may still say running; that is stale. Continuation artifacts remain on Modal
  and are not included in the September 28 local archive.
- Recorded configs, package versions, GPU type and dataset fingerprint match.
  Cached-rollout versus teacher-forced student log-probability discrepancies
  increased before overt collapse; numerical versus algorithmic causality is
  unresolved. Source comparison found no changes to C's sampler or imitation
  objective since the original run.

#### C diagnostics: cache consistency and checkpoint-160 replay

Artifacts below remain on the Modal results volume, outside the original local archive.

- **Inference-only cache test completed in 18.6 minutes**, with no optimization.
  Entrypoint `modal_c_cache_diagnosis.py`; call
  `fc-01M3YQNE74K36MB12TDNE5GSEN`; report
  `diagnostics/c-cache-20261002/report.json`. Compared checkpoints 192/224 on
  the same two saved sequences, 853 tokens each, in BF16 on A10. Fast-kernel
  batched-cache versus full-sequence absolute chosen-token log-prob error:
  mean **0.0326 → 0.0948**, P99 **0.313 → 0.919**, maximum **1.031 → 1.760**.
  Reference kernels did not remove the discrepancy (mean **0.0285 → 0.0999**).
  Single-sequence fast-cache/full-sequence maximum at 224 was **2.747**.
  Later weights exhibit greater execution-path sensitivity, but causality is
  unresolved. Two sequences, batch size two, evaluation mode, and A10 do not
  reproduce the complete L40S training path; no FP32 comparison was performed.
  These token-level means must not be confused with the earlier median of
  per-update maximum discrepancies over up to 32,768 tokens.
- **Checkpoint 160 → 192 replay completed**, preserving original configuration,
  question order, and the optimizer/RNG restoration path on L40S. Entrypoint
  `modal_c_resume_replay.py`; successful call `fc-01M3Z869BW55A15YDE1A1Q2K8W`.
  Remote directory: `diagnostics/c-resume160-replay-20261002/`; comparison in
  `comparison.json`, final evaluation in `eval-000192.json`, checkpoint in
  `checkpoint-000192/`, completion in `result.json`. No successor was spawned.
- Replay final accuracy: **72.27% (185/256)** versus original **70.70% (181/256)**.
  Mean evaluation length **769.11 tokens**, 36/256 capped. Maximum gradient
  norm **0.29045**; text screening found no comparable word-stream degeneration.
  Final checkpoint reload logit error **0.0**. The four-answer difference is
  not evidence of a reproducible improvement from resuming.
- Step 161 reproduced **16/16 responses token-for-token**, but gradient norms
  differed slightly: original **0.0264348**, replay **0.0264047**; adapter update
  norms **0.0534124** versus **0.0534078**. Exact response matches fell to 6/16
  at 162, 2/16 at 163, 5/16 at 164, and zero from 165 onward. Question IDs/order
  matched throughout. Numerical differences followed by sampling feedback are
  consistent with this pattern; correct optimizer restoration is not proven.
  The successful replay weakens a generic resume-causes-collapse hypothesis,
  but does not explain the failed continuation from checkpoint 192.
- Initial replay call `fc-01M3Z72PRKE0KW70Y1R7QAG696` crash-looped before training
  because `modal_training` was omitted from the image. It was cancelled, the
  module was packaged, and the successful replacement above was launched.

### D: OPSD with moving privileged-context self-teacher

Run: `bigmath-4b-opd-512-20260928`.
Local archive: `artifacts/opd-stopped-20260928/bigmath-4b-opd-512-20260928/`.
Config: `configs/opd_4b_privileged_512.json`.

- One A10, 4 CPU cores, 32GB host RAM. Fresh 4B student; teacher shares current
  weights but receives the verified answer. Full-vocabulary OPSD, not B's
  reward-weighted sampled imitation. Planned 512 updates.
- **Stopped for collapse after 37 committed updates**, 579,181 generated
  training tokens; last checkpoint 32. No step-37 evaluation exists.
- Step 32: **0.78% (2/256)** after **427,160 training tokens**, versus baseline
  63.28%. Mean response length rose from 876.07 to **1,817.67**; 169/256 capped.
  Samples became long generic mathematical prose, unlike B's short answers.
- Follow-up diagnostics found both student and privileged teacher went from
  3/3 to 0/3 on three selected easy questions within a 512-token budget.
  Their KL decreased on identical failed prefixes despite shared degradation.
  This supports investigating teacher drift, not a definitive causal diagnosis.
- App stopped and monitor disabled. Token counts exclude evaluations and pilot
  work. See `OPD_512_RUN.md` and `TEACHER_DRIFT_DIAGNOSIS.md`.

### E: OPSD with frozen privileged-context self-teacher

Run: `bigmath-4b-frozen-opd-512-20260928`.
Local archive: `artifacts/frozen-opd-stopped-20260929/bigmath-4b-frozen-opd-512-20260928/`.
Config: `configs/opd_4b_frozen_privileged_512.json`.

- One A10, 4 CPU cores, 32GB host RAM. Matched fresh-start ablation of D:
  same data, LR, loss, prompts and rollout settings, but privileged teacher
  frozen at initial base weights by disabling LoRA during teacher forwards.
  No clipping or EMA. GPU preflight verified exactly invariant teacher logits
  after three student updates; no diagnostic adapters reused.
- **Stopped by user for lack of learning after 205 committed updates**,
  **2,764,230 generated training tokens**; last checkpoint 192.
- Latest held-out accuracy: **63.67% (163/256)** at step 192, versus baseline
  63.28%. Best observed: **64.45% (165/256)** at step 160. Neither establishes
  a clear learning gain. Step-192 mean length was 821.06 tokens; 54 capped.
- Avoided D's early collapse, but ordinary RL A reached 69.92% at step 192.
  This is a single-run comparison, not a general verdict on frozen-teacher OPSD.
- App confirmed stopped with zero tasks; monitor paused. Checkpoint, optimizer,
  RNG, rollouts and evaluations archived; no post-stop checkpoint reload test.
  See `FROZEN_OPD_RUN.md` for provenance and stop details.

| E update | Cumulative generated training tokens | Held-out accuracy |
| ---: | ---: | ---: |
| 32 | 451,851 | 63.67% |
| 64 | 877,379 | 61.72% |
| 96 | 1,304,928 | 61.72% |
| 128 | 1,736,483 | 61.72% |
| 192 | 2,602,206 | 63.67% |

### F: OPD with frozen question-only 9B teacher

Config: `configs/opd_teacher9b_192.json`. Submitted October 2; all 192 updates
confirmed complete October 5.
Run: `bigmath-4b-opd-9b-F-192-20261002` on the results volume under `runs/`.
Call: `fc-01M3YQT1ZPNVCWMFMECJT7M3NA`.
Dedicated app: `product-distributions-arm-f-training`; entrypoint
`modal_arm_f_training.py::train_segment`. Fresh weights, no preflight adapter
reused. Baseline evaluation then evaluations at 64/128/192, checkpoints every
32. Three recovery-aware 64-update segments; hard ceiling 192, no extension
authorized. Consumes 3,072 distinct training questions from the expanded pool.
Completed segments: `segment-000064-attempt-1187419dace7`,
`segment-000128-attempt-72636f68491e`, `segment-000192-attempt-ce279aa2288e`.
Final checkpoint: `segment-000192-attempt-ce279aa2288e/checkpoint-000192`.
Artifacts remain on Modal, outside the original local archive.
Remote preflight: `hardware-benchmarks/arm-f-opd-9b-l40s-20260929/`.

- Fresh 4B student, frozen 9B teacher, neither context privileged; alpha=0,
  full-vocabulary reverse KL, no reward weighting/clipping. 16 independent
  questions per update, same expanded data as D/E, 2K cap, LR 2e-5.
- L40S / four CPU cores / 64 GiB host RAM. Forced 2K backward pass and three
  real updates passed, with finite/nonzero student gradients and unchanged teacher.
- Rollout plus update: 122.38, 121.43, 121.84 seconds. Peak allocated VRAM
  33.60 GiB; maximum reserved 43.74 GiB. Total preflight 443.61 seconds.
- Preflight checkpoint reload max logit error **0.0**. These are feasibility results,
  not evidence of learning. Disposable preflight adapters must not be reused.

| F update | Cumulative generated training tokens | Held-out accuracy |
| ---: | ---: | ---: |
| 0 | 0 | 62.50% (160/256) |
| 64 | 870,841 | 65.23% (167/256) |
| 128 | 1,702,293 | 67.19% (172/256) |
| 192 | 2,557,950 | 67.58% (173/256) |

- Final gain **+5.08 percentage points**, 13 additional correct answers.
  Training-token counts exclude evaluations and preflight work.
- Maximum gradient norm **0.11088** across the run; no observed gradient
  explosion. Final checkpoint reload max logit error **0.0**.
- Mean evaluation length fell from **847.22** to **807.32 tokens**; capped
  responses fell from 68/256 to 58/256.
- Final accuracy was below original C's 70.70% and A's 69.92% at step 192.
  This is a single-run, unequal-question-exposure comparison, not a
  compute-matched or statistically established ranking of the methods.

### G: product imitation with frozen privileged-context self-teacher

Submitted October 5 for 512 updates; **stopped at 256 on October 6** as authorized
after degeneration, once the evaluation and complete checkpoint were committed.
Run: `runs/bigmath-4b-product-frozen-self-G-512-20261005/` on Modal.
Call: `fc-01M46WJ297T7Q45416EVP5AG44`.
App: `product-distributions-arm-g-training`; entrypoint `modal_arm_g_training.py`.
Config: `configs/imitation_frozen_self_512.json`.

- Matched B settings: fresh 4B, alpha=0.5, beta=0, four questions × four
  rollouts, 2K cap, LR 2e-5, LoRA rank 16, seed 42, same 2,048-question pool
  and holdout. A10 / four CPU cores / 32 GiB host RAM. Evaluations every 64,
  checkpoints every 32; recovery-aware segments stop at 512 total updates.
- Teacher sees the verified answer but uses the frozen initial base weights.
  Rollouts batch student `default` LoRA rows with teacher `__base__` rows;
  backward uses only student imitation. This is B's frozen-teacher ablation,
  not E's full-vocabulary OPSD objective.
- Disposable mixed-adapter preflight passed teacher-logit invariance after a
  student update. Two-pair, 16-step cached decoding showed small mixed/separate
  numerical differences; it was not a production-length throughput benchmark.
  Report: `diagnostics/mixed-lora-20261005/report.json`. Local suite: 48 passed.
- Initial attempt reached 33, then recovery resumed checkpoint 32; step 33 was
  replayed. Restart cause remains unestablished. Do not double-count that update.
- Evaluation trajectory (accuracy / mean response tokens / capped):
  0: **63.28% / 876.07 / 61**; 64: **66.80% / 737.10 / 47**;
  128: **61.72% / 148.11 / 0**; 192: **60.94% / 448.31 / 1**;
  256: **0.39% (1/256) / 2,040.67 / 255**.
- Inspection at 226–228 found multilingual word salad, repeated digits and long
  loops of “similarly” or “temporarily,” alongside coherent responses. One
  malformed but verifier-correct response received positive advantage +0.25.
  Freezing the privileged teacher did not prevent collapse under these settings;
  similarity to C's symptoms does not establish a shared root cause.
- Final artifacts: `segment-000256-attempt-f7f05cb313e9/eval-000256.json` and
  `segment-000256-attempt-f7f05cb313e9/checkpoint-000256/`. Adapter, optimizer,
  RNG and state files verified present/nonempty. Reload max logit error **0.0**.
- Stop monitor confirmed both artifacts before stopping only G's app
  `ap-Qix7e8BnaEek74GAMGt5cQ` at **2026-10-06 13:25:52 America/Los_Angeles**.
  App confirmed stopped with zero tasks. Last recorded update **256**; no
  completed-update overshoot. Any successor startup was cut off by the app stop.
  Monitor disabled after completion; artifacts remain on Modal.

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
OPSD** preflight passed on A10 after chunking/checkpointing KL intermediates to
fix a 2K-token OOM. Three updates, 16 distinct questions each, took 172–185
seconds/update and peaked at 17.02 GiB allocated. Checkpoint reload error was
zero. The dataset was expanded to 8,192 unique training questions with the
original holdout unchanged. This is a feasibility test, not a learning result;
the subsequent 512-step OPSD launch is documented in `OPD_512_RUN.md`.
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
   C's continuation subsequently degenerated; the cause remains unresolved.
   F's ordinary OPD improved to 67.58% at 192 updates, below original C's
   70.70% at that horizon. C's separate checkpoint-160 replay reached 72.27%
   without collapse, weakening a generic resume-failure explanation but not
   resolving the checkpoint-192 continuation's degeneration.
4. Moving privileged-self-teacher OPSD (D) collapsed rapidly; freezing that teacher
   (E) avoided the same early collapse but produced no clear learning gain through
   the last evaluated checkpoint, step 192.
5. These experiments do not measure general instruction-following retention,
   non-math generalization, semantic train/test decontamination, or multi-seed variance.
6. Timings are measured compute phases, not invoices. Accuracy-selected best
   checkpoints and different model/hardware configurations require cautious comparisons.
