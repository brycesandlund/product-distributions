# Short-reasoning calibration — September 16, 2026

**The practical winner in this small screen is concise prompting with thinking
disabled.** It produced complete, correct worked answers on all four questions,
both with question-only sampling and with product guidance. No training updates
were performed, and training defaults have not been changed.

## Setup

One rollout per question per setting, using two existing DeepMath training
questions and the first two GSM8K training questions. This is a feasibility
screen, not an accuracy benchmark or a test of training effectiveness.

- Qwen3.5-9B, model revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- GSM8K revision `740312add88f781978c0658806c59bc2815b9866`.
- Same initial model, temperature 1, no top-k/top-p, no repetition/presence penalty.
- Product alpha=0.5, same-model privileged teacher. Student-only alpha=0.
- Seed 42; both contexts receive the same concise instruction.
- A 2,048-token output ceiling for thinking-off, 8,192 for thinking-on.
- Explicit recognition of both `<|im_end|>` and `<|endoftext|>` for this diagnostic.
  Every completed response actually ended with `<|im_end|>`.

Instruction:

> Solve the problem using one concise derivation. Avoid alternative methods and
> repeated checks once the result is established. Give only the requested
> quantity, with the final answer in \boxed{}.

Thinking-off uses the tokenizer's `enable_thinking=False` template, which closes
the thinking section in the prompt. It still permits a worked solution in the
visible answer; it does not force answer-only output.

## Results

| Setting | Mean output tokens | Finished correctly | Verifier successes, including capped traces | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Concise, thinking off, student only | 408 | 4/4 | 4/4 | 18.59 |
| Concise, thinking off, product | 351.5 | 4/4 | 4/4 | 18.50 |
| Concise, thinking on, product | 7,079.75 | 2/4 | 3/4 | 21.17 |

| Question | Student, thinking off | Product, thinking off | Product, thinking on |
| --- | ---: | ---: | --- |
| DeepMath limit | 1,149 | 915 | 8,192; capped, verifier failed |
| DeepMath auxiliary ODE equation | 323 | 333 | 7,862; complete and correct |
| GSM8K clips sold (48 + 24) | 71 | 68 | 4,073; complete and correct |
| GSM8K hourly earnings | 89 | 90 | 8,192; capped, correct box within reasoning |

All thinking-off entries in the second table are complete and correct. The
thinking-on hourly-earnings trace repeatedly planned its final answer, including
the correct box, but never finished. A correct box in a capped reasoning trace
must not be confused with a completed response.

For the actual thinking-on trajectories, a 2K cutoff would give zero EOS
completions and one verifier success; a 4K cutoff would give one completion and
two verifier successes. Those prefix measurements are taken from the same 8K
trajectories, not independent resampling.

Product rollouts used about 20 times fewer generated tokens with thinking off
in this four-question sample. Generation wall times were 161.71 seconds for
student/off, 67.83 for product/off, and 1,036.14 for product/on. Cold kernels and
fixed-size batching of unequal-length responses confound timing comparisons;
do not interpret the faster product/off time as an established benefit of
guidance. Startup and dataset/model loading are excluded from those times.

## Interpretation and next step

The concise instruction alone was insufficient to make thinking-on economical
here, even on elementary arithmetic. Easier questions therefore do not by
themselves solve the completion problem. Turning thinking off also worked on
the two DeepMath examples, so a dataset switch is not required just to obtain
short completions on those examples.

Recommended next calibration: keep the concise, thinking-off setting and a
2K output cap, then screen a larger fixed training-only sample for mixed
success/failure rates. These four easy questions provide no evidence about
the reward variance needed for policy-gradient learning. No broader data screen,
loss sweep, or training run was launched. Long-sequence backward-pass memory
was not measured.

## Artifacts and implementation

Remote results: `/short-reasoning/concise-modes-20260916` on the results volume.
Local complete copy: `artifacts/short-reasoning/concise-modes-20260916/`.
Each setting/question has JSON and Markdown containing both full rendered
prompts and the entire response, with special tokens preserved. Config and
summary files record revisions, question IDs, settings, and prefix scores.

Short example copies:

- `artifacts/short-reasoning/product-limit.md`
- `artifacts/short-reasoning/product-gsm.md`

The deployed `calibrate_short_reasoning` entry point runs the bounded screen.
Prompt customization and extra EOS markers are opt-in; existing training defaults
are unchanged. All 22 local tests, lint, and whitespace checks passed.
