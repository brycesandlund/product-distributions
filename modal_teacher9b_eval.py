"""Standalone frozen 9B evaluation on the A–G holdout; no optimization."""
import modal
from modal_training import training_image
from modal_app import HF_CACHE_PATH, RESULTS_PATH, results_volume, volume_mounts

app = modal.App("product-distributions-teacher9b-eval")
image = training_image.add_local_python_source("modal_training")


@app.function(image=image, gpu="L40S", cpu=4, memory=32768,
              timeout=3 * 3600, retries=0, volumes=volume_mounts)
def evaluate():
    import hashlib
    import json
    from pathlib import Path
    from time import perf_counter
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from product_distributions.training import enable_qwen_kernels, write_json, prompts, verify_answer
    from product_distributions.data import read_examples
    from product_distributions.sampling import ProductSampler, SamplingConfig
    from product_distributions.short_reasoning import CONCISE

    results_volume.reload()
    root = Path(RESULTS_PATH) / "diagnostics/teacher9b-holdout256-20261006"
    root.mkdir(parents=True, exist_ok=False)
    path = Path(RESULTS_PATH) / "data/bigmath-ab-4b-512-v1/eval.jsonl"
    examples = read_examples(path)
    assert len(examples) == 256
    revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    report = {"status": "loading", "model_id": "Qwen/Qwen3.5-9B",
              "revision": revision, "eval_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "instruction": CONCISE, "seed": 42, "batch_size": 8,
              "max_new_tokens": 2048, "temperature": 1, "enable_thinking": False,
              "top_k": None, "top_p": None, "extra_eos_token_ids": [248044],
              "records": []}
    start = perf_counter()
    def save():
        report["elapsed_seconds"] = perf_counter() - start
        write_json(root / "report.json", report)
        results_volume.commit()
    save()
    try:
        enable_qwen_kernels()
        torch.set_float32_matmul_precision("high")
        tokenizer = AutoTokenizer.from_pretrained(report["model_id"], revision=revision, cache_dir=HF_CACHE_PATH)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(report["model_id"], revision=revision,
            cache_dir=HF_CACHE_PATH, dtype=torch.bfloat16, device_map={"": "cuda"})
        model.requires_grad_(False).eval()
        sampler = ProductSampler(model, tokenizer)
        report.update(status="evaluating", gpu=torch.cuda.get_device_name())
        for index in range(0, len(examples), 8):
            batch = examples[index:index+8]
            contexts = [prompts(tokenizer, e, False, instruction=CONCISE)[0] for e in batch]
            outputs = sampler.generate(contexts, contexts, config=SamplingConfig(
                teacher_weight=0, top_k=None, top_p=None, temperature=1,
                max_new_tokens=2048, seed=42+100000+index, extra_eos_token_ids=(248044,)))
            for e, r in zip(batch, outputs, strict=True):
                report["records"].append({"example_id": e.id, "text": r.text,
                    "num_generated_tokens": r.num_generated_tokens,
                    "ended_with_eos": r.token_ids[-1] in {tokenizer.eos_token_id, 248044},
                    **verify_answer(r.text, e.answer)})
            report["completed_questions"] = len(report["records"])
            save()
            print(json.dumps({"completed_questions": report["completed_questions"]}), flush=True)
        records = report["records"]
        report.update(status="complete", accuracy=sum(r["reward"] for r in records)/256,
            mean_tokens=sum(r["num_generated_tokens"] for r in records)/256,
            capped=sum(not r["ended_with_eos"] for r in records))
        save()
        return {k:v for k,v in report.items() if k != "records"}
    except Exception as e:
        report.update(status="failed", error=repr(e)); save(); raise
