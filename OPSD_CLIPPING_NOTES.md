# OPSD/SDPO teacher handling and optional clipping

Checked 2026-09-28 against OPSD paper v3 and repository commit
`ae7d2519e94920c4eb6206c0c26de46d9c50abae`.

- OPSD paper section 3.2 describes shared current weights, detached teacher
  outputs: https://arxiv.org/html/2601.18734v3
- Released OPSD scripts `scripts/run_opsd_4b.sh`, `run_opsd_4b_nonthink.sh`,
  `run_opsd_8b.sh`, and `run_opsd_1b.sh` explicitly set `--fixed_teacher`.
  In `opsd_trainer.py`, this disables LoRA adapters during teacher forwards,
  retaining the initial base as teacher. The trainer also supports current and
  EMA teachers. Paper formulation and released scripts must not be conflated.
- SDPO uses a regularized moving teacher: EMA, or a geometric interpolation of
  initial/current teacher distributions. Section 4.3/Table 4 reports divergence
  for the unregularized current teacher. EMA update rates are 0.05 or 0.01
  depending on setup, not the product-rollout alpha used in our experiments:
  https://arxiv.org/html/2601.20802

## Implemented option

`TrainConfig.opd_pointwise_clip`, default `None`, supported only for `opd_full`.
At every sampled prefix and vocabulary entry, upper-cap the signed contribution
`mu(v) * (log mu(v) - log q(v))` before summing. This matches the reverse-KL
branch plus `jsd.clamp(max=token_clip)` in released `opsd_trainer.py` lines
441–464. No lower cap, no clipping of summed token KL, no PPO ratio clipping,
no gradient-norm clipping. Normalization and chunked full-vocabulary evaluation
are unchanged. Product alpha continues to work. Teacher remains detached.

Clipping signed entries changes the objective; the result need not be a
nonnegative KL divergence. Report it as a clipped surrogate, not raw KL.
Unclipped behavior and old checkpoint compatibility are preserved when None.
Tests compare values and gradients with the authors' elementwise PyTorch
formulation, across alpha 0/0.5/1 and thresholds 1e-6/0.05.

## Threshold and experiment caveat

Released 4B script uses `--beta 0` (forward KL in their code) and cap 0.05;
the both-non-thinking 4B script uses forward KL and cap 1e-6. The 8B script
uses 0.06. Their beta is a divergence selector, not our reward-weight beta.
Thus there is no single universal OPSD clipping threshold, and our reverse-KL
experiment with one of these thresholds is not an exact reproduction.

A clean clipping-only ablation would retain our current-weight teacher,
reverse KL, prompt, LR and data ordering, start from fresh weights, and change
only `opd_pointwise_clip` (0.05 is the main 4B script's starting value). A
fixed-teacher run should be a separate arm. No GPU experiment or deployment
was launched as part of this implementation.

Code reference:
https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/opsd_trainer.py
