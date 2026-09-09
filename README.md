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

The client requests an H100 dynamically. Pass `--gpu H200` (or another Modal GPU)
to change the hardware without redeploying the app.

For fast iteration without looking up the deployed class:

```bash
uv run modal run modal_app.py
```

The `product_generate` method evaluates the ordinary and privileged contexts as
a batch through one shared Qwen3.8-27B instance. It combines their logits before
sampling and returns behavior log-probability and teacher/student KL diagnostics
that can feed the future RLVR implementation.
