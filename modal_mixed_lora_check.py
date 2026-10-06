"""Disposable mixed-adapter rollout checks; no training successor."""
import modal
from modal_training import training_image
from modal_app import HF_CACHE_PATH, RESULTS_PATH, results_volume, volume_mounts

app = modal.App("product-distributions-mixed-lora-check")
image = training_image.add_local_python_source("modal_training")


@app.function(image=image, gpu="A10", cpu=4, memory=32768,
              timeout=1800, retries=0, volumes=volume_mounts)
def check(run_name: str):
    import json
    from pathlib import Path
    from contextlib import nullcontext
    from time import perf_counter
    import torch
    from product_distributions.training import Trainer, TrainConfig, completion_logits, write_json
    from product_distributions.sampling import _tokenize
    from product_distributions.data import read_examples

    assert Path(run_name).name == run_name
    results_volume.reload()
    root = Path(RESULTS_PATH) / "diagnostics" / run_name
    root.mkdir(exist_ok=False, parents=True)
    report = {"status": "loading", "note": "One disposable student update; no G run."}
    def save():
        write_json(root / "report.json", report)
        results_volume.commit()
    save()
    try:
        trainer = Trainer(TrainConfig(model_id="Qwen/Qwen3.5-4B",
            model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            enable_thinking=False, teacher_privileged=True), HF_CACHE_PATH)
        model = trainer.model
        peft = trainer.accelerator.unwrap_model(model)
        examples = read_examples(f"{RESULTS_PATH}/data/bigmath-ab-4b-512-v1/train.jsonl")[:2]
        pairs = [trainer._prompts(e) for e in examples]
        prompts = [p[0] for p in pairs] + [p[1] for p in pairs]
        encoded = _tokenize(trainer.tokenizer, prompts, trainer.accelerator.device)
        n = 2
        names = ["default"] * n + ["__base__"] * n
        def forward(ids, mask, cache=None, adapters=None):
            extra = {} if adapters is None else {"adapter_names": adapters}
            return model(input_ids=ids, attention_mask=mask, past_key_values=cache,
                         use_cache=True, logits_to_keep=1, return_dict=True, **extra)
        @torch.no_grad()
        def probe(mixed):
            model.eval()
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
            outputs = []
            caches = [None] if mixed else [None, None]
            for step in range(17):
                if mixed:
                    o = forward(ids, mask, caches[0], names)
                    caches[0] = o.past_key_values
                    logits = o.logits[:, -1].float()
                else:
                    ls = []
                    for branch in range(2):
                        sl = slice(branch*n, (branch+1)*n)
                        with peft.disable_adapter() if branch else nullcontext():
                            o = forward(ids[sl], mask[sl], caches[branch])
                        caches[branch] = o.past_key_values
                        ls.append(o.logits[:, -1].float())
                    logits = torch.cat(ls)
                outputs.append(logits.cpu())
                # Identical forced prefix for both paths and both contexts.
                token = trainer.tokenizer.encode(" solution", add_special_tokens=False)[0]
                ids = torch.full((4, 1), token, device=mask.device, dtype=torch.long)
                mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            return torch.stack(outputs)
        def compare(a, b):
            d = (a-b).abs()
            return {"mean_abs_logit_error": d.mean().item(), "max_abs_logit_error": d.max().item(),
                    "top1_agreement": (a.argmax(-1)==b.argmax(-1)).float().mean().item()}
        before = probe(True)
        report["initial_mixed_vs_separate"] = compare(before, probe(False))
        # Make LoRA nonzero with a real backward/Adam step on a short answer.
        frozen_versions = {name: p._version for name,p in model.named_parameters() if not p.requires_grad}
        model.train()
        tokens = trainer.tokenizer.encode(" The answer is \\boxed{"+examples[0].answer+"}.", add_special_tokens=False)
        logits = completion_logits(model, trainer.tokenizer, pairs[0][0], tokens)
        loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor(tokens, device=logits.device))
        trainer.accelerator.backward(loss)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        report["student_grad_norm"] = sum(g.float().square().sum() for g in grads).sqrt().item()
        assert report["student_grad_norm"] > 0
        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)
        del logits, loss, grads
        assert frozen_versions == {name:p._version for name,p in model.named_parameters() if not p.requires_grad}
        after = probe(True)
        report["trained_mixed_vs_separate"] = compare(after, probe(False))
        report["teacher_before_after"] = compare(before[:, n:], after[:, n:])
        report["student_before_after"] = compare(before[:, :n], after[:, :n])
        assert torch.equal(before[:, n:], after[:, n:]), "Frozen teacher changed"
        assert not torch.equal(before[:, :n], after[:, :n]), "Student did not change"
        # Warm, equal-work timing of this short fixed-prefix microbenchmark.
        report["timings_seconds"] = {}
        for mixed in (False, True):
            ts=[]
            for _ in range(2):
                torch.cuda.synchronize(); start=perf_counter()
                probe(mixed)
                torch.cuda.synchronize(); ts.append(perf_counter()-start)
            report["timings_seconds"]["mixed" if mixed else "separate"] = ts
        report.update(status="complete", peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      caveat="2 pairs, 16 forced decode steps; timing includes CPU logit copies, not production throughput.")
        save()
        return report
    except Exception as e:
        report.update(status="failed", error=repr(e)); save(); raise
