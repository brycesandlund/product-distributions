# Frozen question-only 9B teacher: 192-step run

Submitted 2026-09-21: `fc-01M32FRHE7ZVBQ4H9AHVF24V2G`.
Run name: `bigmath-4b-teacher9b-192-20260921`.
Configuration: `configs/teacher9b_192.json`.

Fresh Qwen3.5-4B student with rank-16 LoRA, frozen Qwen3.5-9B teacher.
Both models receive the question and concise instruction, never the gold answer.
Alpha=0.5, beta=0, sampled student imitation, 2K output cap, non-thinking,
temperature 1 without top-k/top-p. LR=2e-5, seed=42. Four questions times four
rollouts per update. No benchmark-adapter weights are reused.

The run uses the existing `bigmath-ab-4b-512-v1` data, consuming the same first
768 training questions as A/B through step 192. Evaluation uses the same 256
held-out questions before training and at steps 64, 128, 192, with student-only
question-context generation. Teacher weights are frozen by the existing trainer.

One L40S, four CPU cores, 64GB host RAM. Recovery-safe 64-step segments, with
checkpoints every 32 steps and metrics committed after every step. The shared
segment runner receives the L40S function as its successor, so subsequent
segments cannot fall back to the old A10 function. The entry point rejects
totals above 192. No automatic extension is authorized.

Artifacts: `/runs/bigmath-4b-teacher9b-192-20260921/` on the
`product-distributions-results` volume. `progress.json` identifies the current
call and completed segment. Estimated total cost discussed: $45–$60, not a hard
billing cap; actual lengths, evaluation time and interruptions affect cost.

Preflight: 33 local tests and diff checks passed; remote CPU import check
confirmed the bundled `modal_training` module, shared runner and question-only
teacher support. Only the teacher app was deployed; no old A/B runs restarted.
