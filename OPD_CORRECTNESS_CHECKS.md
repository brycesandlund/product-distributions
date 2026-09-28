# OPD end-to-end correctness checks — 2026-09-28

Bounded GPU diagnostics only. No production loss/trainer edits, no resumed
experiment, and no saved diagnostic adapters used for training.

## Scope

Pinned Qwen3.5-4B, production LoRA setup and full-vocabulary KL. One fixed balloon
arithmetic prompt with privileged answer 6. First probe: 28 continuation tokens.
Explicit-reference probes: a repeated fixed arithmetic response, 168 tokens,
crossing both recurrent and loss chunk boundaries. These are synthetic fixed
prefix checks, not evaluations of reasoning quality. Gradients include all
trainable adapter parameters. Reference comparisons also run after three local
AdamW steps, activating nonzero LoRA B and gradients through LoRA A.

Three disposable calls completed:

- `fc-01M3N3044WSGAAFMX3P1YG018P`: initial 28-token probe, 37.3 sec.
- `fc-01M3N33P0X8H9DC6W0D6MS4MGZ`: explicit reference, 168 tokens, 34.9 sec.
- `fc-01M3N36FMZFB90MTMPQC149BZ1`: highest-FP32-matmul-precision reference,
  168 tokens, 105.8 sec. Device reported A10G; earlier probes reported A10.

Times exclude remote provisioning. Each call had an 1,800-second cap.

## Results (final precision probe)

| Check | Result |
| --- | --- |
| Identical contexts, teacher eval/no-grad vs student train | Identical logits; KL 0; gradient norm 1.8e-6 versus ordinary signal norm 0.612 |
| Frozen teacher target, LR 2e-5 | KL 0.010719 → 0.007450 → 0.002923 → 0.001916 over three updates |
| Frozen teacher target, LR 2e-6 | KL 0.010719 → 0.010416 → 0.011446 → 0.009825; not monotonically descending |
| Checkpoint on/off, initial adapters | Identical logits/loss; gradient relative L2 error 1.24%; cosine 0.999924 |
| Checkpoint on/off, updated adapters | Identical logits/loss; gradient relative L2 error 1.16%; cosine 0.999934 |
| Explicit reference vs optimized kernel, initial adapters | Gradient relative L2 error 10.35%, cosine 0.99506; mean output KL 0.0001044; top-1 agreement 100% |
| Explicit reference vs optimized kernel, updated adapters | Gradient relative L2 error 22.13%, cosine 0.98010; mean output KL 0.0001338; top-1 agreement 99.40% |

Teacher logits stay cached from the production path during the gradient
comparisons, isolating student differences. Only the gated-delta chunk operation
is switched to the PyTorch reference; this is not a fully reference-only model.
The reference function's automatic kernel-dispatch decorator is explicitly
removed, and 48 reference forward calls are verified. Reference matmul precision
is highest in the final probe; production retains high. BF16 model activations
remain in use in both paths. No long-sequence or cached-decode gradient parity
claim follows from these short tests.

The initial probe's nominal reference was still decorated, so it cannot prove
reference execution. Its float32 CPU cosine calculation also returned invalid
values slightly above one; ignore those cosine values. Subsequent probes use
float64 cosine, and updated-adapter relative errors use float64 norms. All raw
reports are retained rather than overwritten.

## Interpretation

The tested production path moves toward a frozen teacher target, not away from
it. Identical-context consistency passes up to a tiny numerical gradient residue.
No obvious sign, alignment, or gross checkpoint-recomputation error was found.
The real reference/optimized gradient discrepancy remains unresolved: high
directional agreement and small output KL do not prove backward correctness or
explain the collapse. Low-rate descent noise also cautions against inferring
monotonicity under mixed precision. A single synthetic question cannot clear the
entire stack, teacher-prompt behavior, or long-horizon moving-teacher dynamics.

Next useful diagnostics are teacher/student behavior and token-level gradient
contributions at the initial and failed checkpoints, and controlled repeated
gradient/longer-sequence kernel comparisons. No further jobs were launched.

## Artifacts

Local reports (including installed reference and production source snapshots):
`artifacts/opd-correctness-20260928/`, with subdirectories
`opd-correctness-20260928`, `opd-correctness-reference-20260928`, and
`opd-correctness-precision-20260928`.
Remote originals: same subdirectory names under `diagnostics/` on
`product-distributions-results`. Latest reproducible diagnostic entry point:
`modal_opd_checks.py` (uses a unique output directory and refuses overwrite).
