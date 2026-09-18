# 4B hardware selection — 2026-09-17

Target model: Qwen/Qwen3.5-4B, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, non-thinking.

The alpha=0 sampler now evaluates only the student context, including with
recorded student log probabilities. Missing teacher log probabilities are
reported as null, not fabricated. Explicit teacher-score requests (sampled OPD)
or KL diagnostics retain teacher evaluation. Alpha>0 is unchanged.

28 local tests passed, including shared-model batch width, absence of separate
teacher forwards, and explicit OPD teacher-score requests at alpha=0.

Hardware tests submitted, not yet validated:

| GPU | Function call | Results directory under `/hardware-benchmarks/` |
| --- | --- | --- |
| L4 | `fc-01M2RPAPVTV1T58CY58VSA30P5` | `qwen35-4b-l4-20260917` |
| A10 | `fc-01M2RPAQC2MBPGGY6Y1JKYHYK4` | `qwen35-4b-a10-20260917` |

Each test requests 4 CPU cores and 32GB host RAM, with a one-hour timeout.
Four disposable-adapter updates: two alpha=0, then two alpha=0.5. Each update
uses the same first four Big-Math training questions, four samples per question,
2K token cap, beta=0, rank-16 LoRA, and learning rate 2e-5. This is a fit/timing
test, not an A/B learning comparison; model weights change between updates.
The first observation may include compilation overhead. Summaries include
allocated/reserved GPU memory, peak host memory, rollout/update timings, and
gradient diagnostics. No larger training run has been launched.
