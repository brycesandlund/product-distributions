# Arm F: 4B student, frozen question-only 9B teacher OPD preflight

Submitted September 29, 2026. No full training run authorized or launched.
Call: `fc-01M3RA664K98GGCJM7QZ5QT3Y4`.
App: `product-distributions-opd-9b-preflight`.
Entrypoint: `modal_opd_9b_preflight.py::pilot`.
Config: `configs/opd_teacher9b_192.json` (prospective training config; pilot
hard-codes only three updates and never dispatches a successor).

Fresh pinned Qwen3.5-4B LoRA student and frozen pinned Qwen3.5-9B teacher,
question-only contexts, full-vocabulary reverse KL, alpha=0, no clipping or
reward weighting. 16 independent questions/update, 2K cap, temperature 1,
no top-k/top-p, LR 2e-5. Same expanded Big-Math data as D/E.
One L40S, four CPU cores, 64 GiB host RAM, one-hour timeout.

Reuses the OPD benchmark: forced 2,048-token backward on the longest prompt
among the first 48 pilot questions, then three real updates on those questions,
teacher parameter-version/no-gradient checks, finite/nonzero student gradient
and update checks, and checkpoint corruption/reload verification.
No preflight adapters will be reused for a learning run. C is untouched.

47 local tests pass. Initial remote result: full-length backward passed with
finite loss/gradients, 33.2346 GiB peak allocated, 34.6523 GiB reserved.
Model setup took 63.37 seconds. All three real updates completed successfully:
122.38, 121.43, 121.84 seconds for rollout plus update. Peak allocated memory
33.60 GiB, maximum reserved 43.74 GiB; teacher unchanged on all updates.
Checkpoint reload max logit error 0.0. Total elapsed 443.61 seconds.
As verified October 2: no full F training run has been launched.

Remote report: volume `product-distributions-results`,
`hardware-benchmarks/arm-f-opd-9b-l40s-20260929/summary.json`.
Rollouts and disposable checkpoint are saved alongside it.
