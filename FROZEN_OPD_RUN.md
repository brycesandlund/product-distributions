# Frozen privileged teacher OPD ablation

## Stopped by user — September 29, 2026

Dedicated app confirmed stopped with zero tasks at 17:59:54 UTC; monitor PAUSED.
205 updates committed, 2,764,230 generated training tokens. Last checkpoint is
192; later updates survive as metrics/rollouts only. No restart authorized.
Latest held-out score: step 192, 163/256 (63.67%), versus baseline 162/256
(63.28%); best observed 165/256 (64.45%) at step 160. No clear learning gain,
but no moving-teacher-style collapse. Step-192 mean response length 821.06,
54 capped, after 2,602,206 generated training tokens.
Local archive destination: `artifacts/frozen-opd-stopped-20260929/`.
Download complete: 286 files / 2,687,178,310 bytes, all local sizes match remote.
This verifies completeness and size, not checksums or checkpoint reload.
Remote originals retained. Remote progress.json may still say running; this
verified stop record supersedes it. Historical monitoring snapshots follow.

Status: running as of 2026-09-29 15:02 UTC, step 173 committed, 2,359,786
cumulative generated training tokens. Step-160 evaluation recovered to 165/256
(64.453125%), mean length 837.12, 60 capped; checkpoint 160 is present.
This breaks the exact 61.72% plateau but is only three questions above baseline,
not convincing evidence of a learning gain. No collapse or failure detected.
Prior saturation review remains documented below; no repeat alert for routine
score fluctuation.

Saturation review triggered at the 2026-09-29 09:23 UTC check.
Step 140 committed, 1,900,206 cumulative generated training tokens. Evaluations
at steps 64, 96, and 128 all scored 158/256 (61.71875%), below baseline
162/256. Step 96 used 1,304,928 training tokens; step 128 used 1,736,483.
Mean eval lengths at 96/128: 838.68/880.28 tokens; capped counts 62/68.
This meets the three-evaluation no-improvement review rule, not the collapse
stop rule. User notified; run left active pending direction. Checkpoint 128
is committed and the successor is running in `segment-000192-attempt-04c7c2eea1cf`,
call `fc-01M3PFGTEHBKDE7HH1NPZ3T63G`. Do not repeat the same saturation alert
unless new evidence materially changes the assessment. Earlier snapshot:

As of the 2026-09-29 06:26 UTC monitoring check:
Step 71 committed, 976,533 cumulative generated training tokens. Step-64 eval:
158/256 (61.71875%), mean 852.02 tokens, 66 capped; 877,379 training tokens
through step 64. No collapse threshold reached; only two post-training evals,
so no three-evaluation saturation trigger yet. Segment 128 is active at
`segment-000128-attempt-f69b0870fa8d`, call `fc-01M3NWKPBC54KCYNG37EA2MHVM`,
resumed from committed checkpoint 64. Earlier monitoring snapshot:
36 updates committed in `segment-000064-attempt-9dcb7c623968`, 504,832 generated
training tokens. Step-32 checkpoint adapter, optimizer, RNG and state files are
present and nonempty. Baseline accuracy is 162/256 (63.28125%); step 32 is
163/256 (63.671875%) after 451,851 generated training tokens. No early collapse
on this evaluation (moving teacher was 2/256 at step 32), but a one-question
increase is not evidence of meaningful improvement or long-term stability.
The segment-level progress file now reports completed_steps=128, target=192;
per-update metrics are the authoritative source within a running segment.
Initial detached call: `fc-01M3N54VNJ09Y22A6YE1JH944W`.
The existing monitor `monitor-4b-privileged-opd` now targets this frozen run
and is ACTIVE, checking every 30 minutes.

This isolates teacher drift from the failed moving-self-teacher run documented
in OPD_512_RUN.md. Fresh pinned Qwen3.5-4B and fresh LoRA, not checkpoint 32.
Teacher forwards disable adapters, exposing the immutable initial base weights;
student forwards and gradients use LoRA. No second model allocation, no EMA.
This mode currently supports only full-vocabulary OPD with alpha=0 and no
separately loaded teacher. Unsupported rollout modes fail validation.

- Config: `configs/opd_4b_frozen_privileged_512.json`.
- Dedicated app: `product-distributions-frozen-opd`, ID `ap-DwlJhLtxuDMeLTENjVQ3Iz`.
- Entry point: `modal_frozen_opd_training.py::train_segment`.
- Run: `bigmath-4b-frozen-opd-512-20260928`.
- Dataset: `bigmath-opd-8192-v1`, 8,192 train / same 256 held-out questions.
- GPU: one A10. Same 16 independent questions/update, 512 updates maximum.
- Full-vocabulary reverse KL, alpha=0, no reward weighting, **no clipping**.
- LR 2e-5, rank 16, seed 42, non-thinking, temperature 1, no truncation,
  2,048 generated-token cap, same concise instruction and stop tokens.
- Baseline evaluation, then evaluation/checkpoint every 32 updates.
- Recovery-aware 64-update segments spawn successors on Modal.

Local tests: 47 passed. Real-GPU preflight uses a disposable adapter and checks
exactly invariant teacher logits after three student updates, nonzero student
change, and only LoRA parameters trainable. No preflight weights are reused.
Passed: teacher maximum logit change **0.0**, student maximum change **2.484375**.
Diagnostic losses: 0.02613818, 0.01456556, 0.00419097 (short fixed prefix,
token-count normalization; not comparable to training's 2K-normalized losses).

Monitoring policy: check every 30 minutes when the local app can run. Notify on
failure, completion, collapse, convincing saturation, or required action; stay
quiet during routine progress. If held-out accuracy falls at least 15 percentage
points below this run's baseline, verify evaluation and committed checkpoint,
then stop only this dedicated app. Three evaluations without meaningful
improvement trigger review, not automatic stopping. No automatic failed-run
restart, configuration changes, or extension beyond 512 updates.

Remote output: volume `product-distributions-results`, under
`runs/bigmath-4b-frozen-opd-512-20260928/`. Progress records identify current
call, segment and checkpoint. Archive results and update training_runs.md when
complete or stopped. Compare held-out accuracy against generated training tokens.
