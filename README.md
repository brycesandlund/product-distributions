# Product distributions

Experiments in guided on-policy distillation using a weighted geometric mean of
student and privileged-context next-token distributions.

## Local setup

Only the Modal client runs locally. CUDA and model dependencies are pinned in
`modal_app.py` and installed in the remote image.

```bash
uv sync
uv run modal setup
```

## Modal workflow

Deploy the app and populate the persistent Hugging Face cache without paying for
a GPU during the model download:

```bash
uv run modal deploy modal_app.py
uv run python scripts/invoke_modal.py cache
```

Invoke the deployed Qwen worker:

```bash
uv run python scripts/invoke_modal.py health
uv run python scripts/invoke_modal.py generate \
  --question "What is 17 times 19? Answer with only the integer." \
  --privileged-context "The verified answer is 323." \
  --teacher-weight 0.5 \
  --max-new-tokens 16
```

Benchmark multiple rollout pairs in one batch:

```bash
uv run python scripts/invoke_modal.py batch --batch-size 4 --max-new-tokens 64
```

The deployed dual-model worker pairs a Qwen3.5-9B student with a Qwen3.8-27B
teacher on the same GPU:

```bash
uv run python scripts/invoke_modal.py dual-health
uv run python scripts/invoke_modal.py dual-batch \
  --batch-size 4 \
  --teacher-weight 0.5 \
  --max-new-tokens 64
```

Before first use, cache the student checkpoint without a GPU:

```bash
uv run modal run modal_app.py::cache_model --model-id Qwen/Qwen3.5-9B
```

The client requests an H100 dynamically. Pass `--gpu H200` (or another Modal GPU)
to change the hardware without redeploying the app.

For fast iteration without looking up the deployed class:

```bash
uv run modal run modal_app.py
```

`ProductSampler` is the canonical implementation. It supports either one model
under two contexts or distinct student and teacher models with identical token
vocabularies. The shared-model path concatenates both views into one model batch;
the distinct-model path maintains independent devices and caches.

The deployed `product_generate_batch` method accepts multiple question/context
pairs so rollout throughput can scale with GPU batch size. Full-vocabulary KL
diagnostics are disabled by default and can be enabled explicitly when needed.

Qwen3.8's Gated DeltaNet layers use Flash Linear Attention kernels. The Modal
worker binds those kernels explicitly to avoid Transformers silently selecting
its much slower PyTorch fallback.

## Training

The training app is separate from the inference deployment. It uses native
Transformers, Accelerate, and PEFT LoRA (rank 16 by default), on one GPU. All
decoder linear layers receive adapters; the vision encoder and LM head stay
frozen. Rollouts and updates use the same adapter weights. The privileged
self-teacher forward is detached, and automatically changes after each update.
Set `teacher_model_id` for a separate frozen teacher with a compatible vocabulary.

```bash
uv run modal deploy modal_training.py
uv run python scripts/train_modal.py prepare --data-name deepmath-v1
uv run python scripts/train_modal.py train --config configs/imitation.json
uv run python scripts/train_modal.py train --config configs/opd.json
```

The example configs run 10 updates, not a full experimental sweep. Every update
collects fresh groups, accumulates gradients over one unpadded trajectory at a
time, and takes exactly one optimizer step. No PPO clipping, gradient clipping,
importance correction, reward standardization, or weight decay is applied.
The loss denominator is `number_of_rollouts * max_new_tokens`, independent of
actual completion lengths. Prompt tokens are excluded; generated EOS is included.
Rollouts use temperature 1 and no top-k/top-p truncation. Generation stops at EOS
or the configured length cap; incomplete responses and verifier statuses are logged.

Three loss modes are available:

- `imitation`: sampled student log-likelihood weighted by
  `beta * reward + reward_scale * (reward - group_mean_reward)`.
  If `reward_scale` is null, it is `1 - beta`, giving
  `w_beta = (1 - beta) * (R - mean(R)) + beta * R`. The mean is per question's
  rollout group. With binary rewards, `beta=0` is centered reinforcement and
  `beta=1` imitates only successful rollouts at weight one. All-failed groups
  have zero weight; all-successful groups have weight beta. Beta is no longer
  the mean weight: the group mean weight is `beta * mean(R)`. Non-binary reward
  calibration is left for later; no clipping or rescaling is applied implicitly.
  Old imitation checkpoints using constant-offset weights cannot be resumed
  silently under these new semantics; the checkpoint loader rejects them.
- `opd`: sampled, token-local product-to-teacher reverse-KL surrogate, with
  detached `teacher_logp - behavior_logp` advantages. No verifier reward weights
  enter this loss. Gradients pass through the student part of the normalized
  product; the teacher branch is detached. `beta` and `reward_scale` are ignored.
- `opd_full`: full-vocabulary `KL(product || teacher)` at each sampled prefix.
  Both the product probabilities and their log probabilities are differentiated
  through the student; the teacher is detached. No reward weights or future-state
  return-to-go are used. `beta` and `reward_scale` are ignored. At alpha=0 this is
  reverse-KL student-to-teacher OPD; at alpha=1 the gradient is zero.

The `imitation` and `opd` modes supervise sampled actions; `opd_full` sums over
all vocabulary actions, but still only at prefixes visited by the rollout.
Training recomputes teacher
logits for the product normalizer, one trajectory at a time, with no teacher
gradient graph. Old sampled probabilities are recorded during generation.
Full-vocabulary OPD avoids sampled-action gradient variance, not prefix sampling
variance, and uses additional training memory. Use `configs/opd_full.json` or
`configs/opd_full_teacher27b_smoke.json` to select it.

Data preparation pins a DeepMath revision and takes a small deterministic subset
after a question-hash train/eval split. It uses only questions and final answers,
not provided reasoning traces. Math-Verify checks the final boxed answer with
bounded parsing/verification time; missing or invalid answers receive zero.
Unsupported gold answers are excluded during preparation and listed in the
manifest; all selected gold answers must parse successfully before training. The held-out evaluator
receives question-only prompts. Its before/after outputs are saved for inspection.
The default evaluation subset is small: use larger subsets for actual comparisons.

Optional `--retention-path /results/...jsonl` adds a separate before/after retention
evaluation; without that input, no claim about retention is measured. Custom JSONL
rows use `{"id": "...", "question": "...", "answer": "..."}`. This evaluator covers
verifiable math answers, not general chat behavior.

Results live in the `product-distributions-results` volume at `/runs/<run-name>`:
config, dataset manifest, sampled trajectories and logprobs, metrics, evaluations,
and checkpoints. Model revisions and package versions are recorded. Checkpoints
contain adapters, optimizer, RNG, tokenizer, step metadata, and a dataset
fingerprint checked on resume. Frozen base weights remain in the HF cache. The final checkpoint is
verified by perturbing adapters, reloading, and comparing logits exactly.

```bash
# Resume into a new run directory; config must match except the total steps.
uv run python scripts/train_modal.py train --config configs/imitation.json \
  --resume /results/runs/PREVIOUS/checkpoint-000005
uv run modal volume get product-distributions-results runs/RUN artifacts/RUN
```

For **inference-only length calibration**, with no optimizer updates:

```bash
uv run python scripts/calibrate_modal.py --lengths 2048 4096 --questions 4 --group-size 4
```

This uses the first fixed training questions, matched seeds across caps,
Qwen3.5-9B with privileged self-context, alpha=0.5, thinking enabled, temperature
one, and no top-k/top-p. It reports EOS completion separately from boxed-answer
presence and verifier reward, along with mixed-success group counts, actual
token lengths, generation throughput, and inference peak GPU memory. Full
outputs and pinned configuration are stored under `/calibration/<run-name>` on
the results volume. This does not measure backward-pass memory or learning.
See [CALIBRATION_REPORT.md](CALIBRATION_REPORT.md) for the initial 2K/4K results:
both caps truncated every sampled response, so neither is yet a validated
training length budget. Matched seeds did not guarantee identical prefixes.
The follow-up [native generation inspection](NATIVE_INSPECTION_REPORT.md)
contains full-transcript locations and results from the same prompts through
Transformers `model.generate()`; all four native responses also reached 4K.

For training hardware smoke tests:

```bash
uv run modal run modal_training.py::prepare_data \
  --name deepmath-smoke --train-size 4 --eval-size 1
uv run modal run modal_training.py --loss imitation --beta 1
uv run modal run modal_training.py --loss opd
```

Smoke defaults are 64 output tokens, two rollouts, and thinking disabled. They
exercise memory, gradients, and checkpointing; they do not measure learning
quality. Research configs enable thinking and allow 512 tokens; the appropriate
length budget needs calibration on the chosen problems.

Local objective and verifier checks (no CUDA required):

```bash
uv run --with torch==2.14.0 --with numpy --with pytest \
  --with 'math-verify[antlr4_13_2]==0.9.0' python -m pytest -q
```
