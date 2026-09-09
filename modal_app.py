"""Modal infrastructure for Qwen product-distribution experiments."""

from __future__ import annotations

import modal


APP_NAME = "product-distributions"
MODEL_ID = "Qwen/Qwen3.8-27B"
DEFAULT_GPU = "H100"
HF_CACHE_PATH = "/cache/huggingface"
RESULTS_PATH = "/results"

app = modal.App(APP_NAME)

runtime_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==5.16.1",
        "accelerate==1.14.0",
        "huggingface-hub==1.30.0",
        "safetensors>=0.6.2",
    )
    .add_local_python_source("product_distributions")
)

model_cache = modal.Volume.from_name(
    "product-distributions-hf-cache", create_if_missing=True
)
results_volume = modal.Volume.from_name(
    "product-distributions-results", create_if_missing=True
)

volume_mounts = {
    HF_CACHE_PATH: model_cache,
    RESULTS_PATH: results_volume,
}


@app.function(
    image=runtime_image,
    cpu=4,
    memory=16_384,
    timeout=2 * 60 * 60,
    volumes={HF_CACHE_PATH: model_cache},
)
def cache_model(model_id: str = MODEL_ID) -> str:
    """Download a model once without paying for a GPU during transfer."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id, cache_dir=HF_CACHE_PATH)
    model_cache.commit()
    return path


@app.cls(
    image=runtime_image,
    cpu=8,
    memory=131_072,
    timeout=60 * 60,
    scaledown_window=5 * 60,
    volumes=volume_mounts,
)
class QwenWorker:
    """A persistent Qwen3.8 worker with direct access to full token logits."""

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        torch.set_float32_matmul_precision("high")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=HF_CACHE_PATH,
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            cache_dir=HF_CACHE_PATH,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        model_cache.commit()

    def _chat_prompt(self, content: str, *, enable_thinking: bool) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    @modal.method()
    def health(self) -> dict[str, object]:
        import torch

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "model_id": MODEL_ID,
            "gpu": torch.cuda.get_device_name(0),
            "cuda": str(torch.version.cuda),
            "torch": str(torch.__version__),
            "transformers": __import__("transformers").__version__,
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "free_gib": free_bytes / 2**30,
            "total_gib": total_bytes / 2**30,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
        }

    @modal.method()
    def product_generate(
        self,
        question: str,
        privileged_context: str,
        *,
        teacher_weight: float = 0.5,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.95,
        seed: int | None = 0,
        enable_thinking: bool = False,
    ) -> dict[str, object]:
        """Generate one shared rollout from ordinary and privileged contexts."""
        from product_distributions.sampling import product_generate_two_contexts

        student_prompt = self._chat_prompt(
            question, enable_thinking=enable_thinking
        )
        teacher_prompt = self._chat_prompt(
            f"{question}\n\nPrivileged information:\n{privileged_context}",
            enable_thinking=enable_thinking,
        )
        result = product_generate_two_contexts(
            self.model,
            self.tokenizer,
            student_prompt,
            teacher_prompt,
            teacher_weight=teacher_weight,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        return result.to_dict()


@app.local_entrypoint()
def main(
    question: str = "What is 17 times 19? Answer with only the integer.",
    privileged_context: str = "The verified answer is 323.",
    teacher_weight: float = 0.5,
    max_new_tokens: int = 16,
) -> None:
    worker = QwenWorker.with_options(gpu=DEFAULT_GPU)()
    print(worker.health.remote())
    print(
        worker.product_generate.remote(
            question,
            privileged_context,
            teacher_weight=teacher_weight,
            max_new_tokens=max_new_tokens,
            enable_thinking=False,
        )
    )
