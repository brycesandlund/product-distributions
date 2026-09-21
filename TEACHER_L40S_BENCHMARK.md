# L40S: 4B student + frozen 9B teacher

Submitted 2026-09-18, Modal call `fc-01M2V7K64B1DWNK7PJAZ9VZE5B`.
Separate deployment `product-distributions-teacher-benchmark`; active A runner
was not redeployed. No full new learning arm has been launched.

- Student Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Teacher Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Both question-only, non-thinking, concise prompts. Gold answers are used
  only by the verifier, not provided to either model.
- Alpha=0.5, beta=0, sampled student imitation, rank-16 LoRA, LR=2e-5.
- Four questions, four rollouts each, 2K cap, temperature 1, no top-k/top-p.
- Three real updates on a disposable student adapter. First update includes
  warm-up; this is a hardware benchmark, not a learning-effectiveness study.
- One L40S, 4 CPU cores, 64GB host RAM, one-hour timeout.
- Teacher parameters have gradients disabled; version counters and absence
  of gradients are checked after each update.

Reports and full rollouts: `/hardware-benchmarks/4b-student-9b-teacher-l40s-20260918/`
on `product-distributions-results`. Report includes timings, allocated/reserved
GPU memory, host peak RAM, prompts, rewards and update diagnostics.
Local verification: 33 tests passed, including question-only teacher isolation.

## Packaging fix and retry — 2026-09-20

The original call failed during container import because `modal_training` was
not mounted in the benchmark image; it produced no benchmark results. That
pending call was cancelled. The benchmark image now explicitly includes
`modal_training`, and the remote CPU `check_imports` function confirmed the
module imports from `/root/modal_training.py` before the GPU retry was launched.
All 33 local tests passed again.

Only the unchanged three-update benchmark was resubmitted, as
`4b-student-9b-teacher-l40s-20260920-retry`, call
`fc-01M30PX9EGEPG28RX8VGMNY1ZF`. It now writes a loading-models status before
initialization so container startup can be distinguished from model readiness.
The training app and A/B jobs were not relaunched or redeployed.
