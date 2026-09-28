# Frozen privileged teacher OPD ablation

Status: dispatched 2026-09-28 after successful A10 preflight; waiting for
training startup. At handoff, the call is pending and progress.json is not yet
present. Do not treat this as evidence of a completed update or relaunch it.
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
