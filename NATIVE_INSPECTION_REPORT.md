# Full-prompt inspection and native generation — September 16, 2026

Two fixed saved 4K samples were selected: sample zero for the first (limit) and
second (ODE) calibration questions. Complete student and privileged-teacher
prompts were reconstructed with the pinned tokenizer and original prompt code.
The full saved responses were decoded from token IDs with special tokens kept.
The transcripts contain no omitted sections.

## Native comparison

Each of the four prompts was passed independently through Transformers'
`model.generate()` in one left-padded batch. This is **not** native product
sampling: the two contexts each receive an independent continuation.

- Model: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Same initial LoRA model and FLA kernels as calibration; eval/inference mode.
- Thinking enabled; temperature 1; top-k 0 (disabled), top-p 1;
  repetition penalty 1; 4,096 new tokens; seed 42.
- Zero backward passes and optimizer updates. No training sweep launched.

| Prompt | Tokens | EOS | Exited thinking | Verifier result |
| --- | ---: | --- | --- | --- |
| Limit, question-only | 4,096 | No | No | Missing boxed answer |
| Limit, privileged answer | 4,096 | No | No | Missing boxed answer |
| ODE, question-only | 4,096 | No | No | Missing boxed answer |
| ODE, privileged answer | 4,096 | No | No | Unparseable prediction |

Native generation took 461.22 seconds for 16,384 tokens (35.5 aggregate tokens/s),
with 18.70 GiB peak allocated memory. This one-batch timing is not a controlled
speed comparison with the warmed product sampler.

The raw outputs contain neither `</think>` nor either end token below. As in
the custom-loop outputs, the model repeatedly re-derives results, reconsiders
problem wording, and checks alternatives without producing its final answer.

## Stopping-rule discrepancy

The tokenizer EOS is **248046 / `<|im_end|>`**, which the custom sampler uses.
The model's native generation config EOS is **248044 / `<|endoftext|>`**, which
this native run preserves. Neither occurs in any of the four native outputs
or the 32 guided calibration outputs, so switching between them would not
have shortened these traces. The correct chat stopping policy should be
resolved explicitly; the native config is not automatically proof that the
tokenizer's chat end marker is wrong. No sampler stopping behavior was changed.

## Conclusion and limits

The overlong reasoning reproduces outside our custom decoding loop, including
with privileged answers. That makes the loop alone an insufficient explanation.
It does **not** validate all incremental logits, rule out a shared kernel or
model setup issue, or establish that 8K/16K will be enough. Both paths still use
the same model, adapter setup, Transformers version, and FLA backend.

## Complete transcripts

Original product-sampled responses, each with both exact prompts:

- `artifacts/prompt-inspection/sample-1.md` — limit question.
- `artifacts/prompt-inspection/sample-2.md` — ODE question.

Native transcripts, each with its exact prompt and entire response:

- `artifacts/prompt-inspection/native/native-prompts-20260916/sample-1-student.md`
- `artifacts/prompt-inspection/native/native-prompts-20260916/sample-1-teacher.md`
- `artifacts/prompt-inspection/native/native-prompts-20260916/sample-2-student.md`
- `artifacts/prompt-inspection/native/native-prompts-20260916/sample-2-teacher.md`

JSON files alongside the Markdown preserve token IDs, prompt text, raw and
special-token-stripped responses, settings, and metrics. The remote copy is
`/native-inspection/native-prompts-20260916` on the results volume.

Implementation: `scripts/inspect_calibration.py` renders saved transcripts;
the deployed `inspect_native_generation` function in `modal_training.py` runs
the GPU check. Its source directory defaults to the completed calibration;
use a fresh output run name for another invocation.
