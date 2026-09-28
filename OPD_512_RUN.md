# 4B privileged self-teacher ordinary OPD — 512 steps

## Outcome: stopped early for collapse

Stopped September 28 at approximately 22:28 UTC under the agreed collapse rule.
The dedicated Modal app is confirmed stopped with zero tasks; monitor paused.
Step-32 evaluation fell from 162/256 (63.28%) to 2/256 (0.78%), a 62.50-point
drop after **427,160 generated training tokens** and 512 unique questions.
Mean evaluation length grew from 876.07 to 1,817.67 tokens; 169/256 responses
hit the cap. Inspected samples produce long, generic mathematical prose rather
than the earlier privileged-imitation run's short answer-only collapse.
This observation does not establish the underlying cause.

Training continued to 37 committed updates before the monitor stopped it:
579,181 generated training tokens total. The last saved checkpoint is step 32;
adapter, optimizer, RNG and state files were verified present and nonempty.
No checkpoint reload test was run after this stop. Steps 33–37 remain as
rollouts/metrics only. No restart is authorized.

Local archive: `artifacts/opd-stopped-20260928/bigmath-4b-opd-512-20260928/`.
Remote originals retained. The archived `progress.json` is stale (segment-level
running state); this stop record and the stopped Modal app supersede it.

## Original launch

Launched 2026-09-28, explicitly authorized by user after successful preflight.

- Run: `bigmath-4b-opd-512-20260928`.
- Initial call: `fc-01M3MK3SPQ3S7ZH2GZ4GRF5FN3`.
- Dedicated app: `product-distributions-opd-training`.
- Entry point: `modal_opd_training.py::train_segment`.
- Config: `configs/opd_4b_privileged_512.json`.
- Dataset: `bigmath-opd-8192-v1` (8,192 unique train, unchanged 256 eval).
- Volume: `product-distributions-results`, root `runs/bigmath-4b-opd-512-20260928/`.

Fresh pinned Qwen3.5-4B weights and rank-16 LoRA; **no pilot adapters reused**.
Ordinary OPD (`alpha=0`), full-vocabulary reverse KL with chunked/checkpointed
intermediates, question-only rollouts, same current weights with privileged
answer context as teacher. Teacher branch detached, no reward weighting.
16 distinct questions × 1 sample per update; max 512 steps, one training epoch.
Non-thinking, temp 1, no top-k/top-p, 2K cap, LR 2e-5, seed 42.

One A10, four CPU cores, 32 GiB host RAM. Recovery-aware 64-step segments,
each with a 12-hour timeout. Evaluations and checkpoints every 32 updates.
The deployed function checks that the memory-bounded loss is present.
`progress.json` is authoritative for the current call/checkpoint. Initial
progress verified running at step 0, target 64, checkpoint null.

The segments spawn their successors on Modal up to 512; laptop availability
does not control training. Local monitoring checks every 30 minutes when the
app/machine can execute it. Collapse rule: verified held-out accuracy at least
15 percentage points below this run's baseline, then stop this dedicated app
after checking a committed checkpoint. Saturation: review after three evals
without meaningful improvement; notify rather than automatically stop on noise.
No automatic failed-run relaunch or extension past 512 is authorized.

Preflight suggests 24–26 hours for updates alone, excluding evaluations,
loading, checkpoint verification and interruptions. This is not a billing cap.
Report accuracy versus cumulative generated training tokens, with GPU time and
unique question exposure separately. See `OPD_PREFLIGHT.md` for feasibility data.
