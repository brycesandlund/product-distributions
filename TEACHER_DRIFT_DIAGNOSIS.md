# Privileged teacher drift diagnostic — 2026-09-28

Call `fc-01M3N3QN1GBE9RXV4PC5GQ2C88`, A10, elapsed 130.50 seconds
excluding provisioning. No optimizer steps or training restart.

## Design

Three deliberately selected easy held-out failures: balloons (8−2), apple
revenue (3×4×5×0.5), and spire height (1454−1250). Compare fresh pinned 4B
weights versus the failed step-32 adapter. Adapter loading reported no missing
LoRA keys or unexpected keys. Both contexts use the same weights at each state.

Generate one response per question and context, temperature 1, no top-k/top-p,
512-token cap, seed 123. These are new diagnostic samples, not replayed evals.
Separately score the same original correct and failed response prefixes under
both contexts and both weight snapshots (at most 128 prefix tokens). The first
position tests the empty continuation; the last predicts after the fixed prefix.

For each position compute exact reverse-KL logit gradient
`p(v) * (log p(v) - log q(v) - KL(p || q))`, with teacher detached. Negative
components increase the corresponding logit under direct logit-space descent.
These are **not parameter gradients**: the model Jacobian, optimizer, and
coupling across positions can change actual update effects.

## Teacher generations also fail

| Weights | Question-only student | Privileged teacher |
| --- | --- | --- |
| Initial | 3/3 correct; 43, 124, 100 tokens | 3/3 correct; 40, 96, 114 tokens |
| Step 32 | 0/3 correct; all 512 tokens | 0/3 correct; all 512 tokens |

The step-32 privileged teacher produces the same generic, irrelevant prose
despite receiving the correct answer. For balloons it begins:

> In the initial phase of observing the balloon distribution, Joan is presented with a collection of eight distinct orange objects. However, as the day progresses and external factors begin to influence the environment, the primary structural elements of this arrangement undergo significant natural degradation.

For the spire it invents a structural volume of 1,600 feet, despite having the
correct answer 204 in its prompt. These truncated samples establish failure to
answer within this diagnostic budget, not what would happen with unlimited tokens.

## Agreement increases on the same failed prefixes

Mean per-position full-vocabulary student-to-teacher KL:

| Question | Failed prefix, initial weights | Same failed prefix, step 32 |
| --- | ---: | ---: |
| Balloons | 0.02903 | 0.01079 |
| Apples | 0.03118 | 0.01192 |
| Spire | 0.05185 | 0.01064 |

Thus both contexts can become more consistent on bad continuations while both
are unable to answer. This is controlled for prefix text and question, unlike
the ordinary training-loss curve whose batches and prefixes change.

On original correct prefixes, the corresponding KL also declines: balloons
0.02459→0.01602, apples 0.01612→0.00989, spire 0.01679→0.01277. This does not
mean those prefixes remain likely to be generated; these measurements condition
on the correct text being supplied.

## What the logit-level signal emphasizes

At initial weights, top absolute-gradient-mass tokens on correct prefixes include:

- Balloons: ` amount`, `Jo`, `.`, ` $`, newlines, ` calculated`, ` number`.
- Apples: `The`, ` plot`, `Each`, ` total`, ` number`, `Total`, ` made`, ` earned`.
- Spire: `:`, ` of`, ` \\`, ` subtract`, ` we`, ` feet`, ` ft`, ` building`.

This supports inspecting wording/formatting supervision, not assuming the dense
signal primarily teaches arithmetic. It is a ranked list, not a formal
style-versus-math attribution or proof that these tokens caused collapse.
After collapse, failed-prefix signals include common words and generic
structural vocabulary; there is no demonstrated reliable corrective teacher.

## Interpretation and limitations

The privileged teacher is **not an intact oracle that only the student fails
to follow** in these cases. Both share degraded weights. The observations are
consistent with moving-teacher co-degradation and an agreement objective that
does not itself guarantee correctness. They do not establish whether the
initiating cause was prompt-induced style supervision, optimizer settings,
numerical behavior, or another implementation issue. No causal ablation was run.

Suggested isolating control: keep the initial teacher weights frozen while
retaining the same privileged prompt and objective. A prompt/style intervention
or divergence clipping would be separate ablations, not simultaneous fixes.
No new experiment has been launched.

Full probabilities, logit-gradient rankings, prompts and generated texts:
`artifacts/teacher-diagnosis-20260928/teacher-drift-20260928/report.json`.
Remote original: `diagnostics/teacher-drift-20260928/report.json` on
`product-distributions-results`. Entry point: `modal_teacher_diagnosis.py`.
