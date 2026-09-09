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
