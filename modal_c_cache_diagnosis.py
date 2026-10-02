"""Fixed-token, inference-only cache/reference comparison for C; no training."""
import modal
from modal_training import training_image
from modal_app import HF_CACHE_PATH, RESULTS_PATH, results_volume, volume_mounts

app = modal.App("product-distributions-c-cache-diagnosis")
image = training_image.add_local_python_source("modal_training")


@app.function(image=image, gpu="A10", cpu=4, memory=32768,
              timeout=1800, volumes=volume_mounts)
def diagnose():
    import ast
    import inspect
    import json
    from pathlib import Path
    from time import perf_counter
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from product_distributions.training import Trainer, TrainConfig, completion_logits, write_json
    from product_distributions.data import read_examples
    from product_distributions.sampling import _forward, _advance, _tokenize

    started = perf_counter()
    root = Path(RESULTS_PATH)/"runs/bigmath-4b-teacher9b-192-20260921"
    out = Path(RESULTS_PATH)/"diagnostics/c-cache-20261002"
    results_volume.reload()
    out.mkdir(parents=True, exist_ok=False)
    report = {"status":"loading", "checks":[], "note":"Inference only; fixed tokens, no optimizer steps. Two selected trajectories, not a population estimate."}
    def save():
        report["elapsed_seconds"] = perf_counter()-started
        write_json(out/"report.json", report)
        results_volume.commit()
    save()
    try:
        # Bypass automatic kernel dispatch on the installed reference functions.
        references = {}
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            tree = ast.parse(inspect.getsource(getattr(modeling,name)))
            tree.body[0].decorator_list = []
            ns = dict(vars(modeling))
            exec(compile(tree,"<explicit_reference>","exec"),ns)
            references[name] = ns[tree.body[0].name]
        cfg = json.loads((root/"segment-000192-attempt-b008b86f396f/config.json").read_text())
        cfg.update(teacher_model_id=None, teacher_revision=None)
        t = Trainer(TrainConfig(**cfg), HF_CACHE_PATH)
        fast = {name:getattr(modeling,name) for name in references}
        examples = {e.id:e for e in read_examples(Path(RESULTS_PATH)/"data/bigmath-ab-4b-512-v1/train.jsonl")}
        records = json.loads((root/"segment-000256-attempt-9066c8f31fc8/rollouts-000216.json").read_text())
        selected = []
        for r in sorted(records,key=lambda r:len(r["rollout"]["token_ids"]),reverse=True):
            if not any(s["example_id"]==r["example_id"] for s in selected): selected.append(r)
            if len(selected)==2: break
        prompts = [t._prompts(examples[r["example_id"]])[0] for r in selected]
        sequences = [r["rollout"]["token_ids"] for r in selected]
        n = min(1024, *(len(s) for s in sequences))
        sequences = [s[:n] for s in sequences]
        report.update(tokens=n, example_ids=[r["example_id"] for r in selected],
                      prompt_lengths=[len(t.tokenizer.encode(p,add_special_tokens=False)) for p in prompts])

        def compact(logits, token):
            lp = logits.float().log_softmax(-1)
            return torch.stack((lp[torch.arange(len(token),device=lp.device),token],lp.argmax(-1).float()),-1).cpu()

        @torch.no_grad()
        def cached(ps, seqs):
            encoded = _tokenize(t.tokenizer, ps, t.model.get_input_embeddings().weight.device)
            state = _forward(t.model, **encoded)
            rows = []
            for pos in range(n):
                ids = torch.tensor([s[pos] for s in seqs],device=t.accelerator.device)
                rows.append(compact(state.next_logits,ids))
                if pos+1<n:
                    state = _advance(t.model,state,ids,torch.ones(len(ps),dtype=torch.bool,device=ids.device))
            return torch.stack(rows,1)

        def compare(a,b):
            err=(a[...,0]-b[...,0]).abs().flatten()
            return {"mean_abs_logp":err.mean().item(),"p50":err.quantile(.5).item(),
                    "p95":err.quantile(.95).item(),"p99":err.quantile(.99).item(),
                    "max":err.max().item(),"fraction_above_1":(err>1).float().mean().item(),
                    "top1_agreement":(a[...,1]==b[...,1]).float().mean().item()}
        for step,segment in [(192,"segment-000192-attempt-b008b86f396f"),(224,"segment-000256-attempt-9066c8f31fc8")]:
            result=set_peft_model_state_dict(t.accelerator.unwrap_model(t.model),load_file(str(root/segment/f"checkpoint-{step:06d}/adapter/adapter_model.safetensors")))
            assert not result.unexpected_keys and not [k for k in result.missing_keys if "lora_" in k]
            t.model.eval()
            outputs={}
            for backend,funcs in [("fast",fast),("reference",references)]:
                for name,fn in funcs.items(): setattr(modeling,name,fn)
                full=[]
                with torch.no_grad():
                    for p,s in zip(prompts,sequences):
                        logits=completion_logits(t.model,t.tokenizer,p,s)
                        full.append(compact(logits,torch.tensor(s,device=logits.device)))
                        del logits
                full=torch.stack(full)
                single=torch.cat([cached([p],[s]) for p,s in zip(prompts,sequences)])
                batched=cached(prompts,sequences)
                outputs[backend]=(full,single,batched)
                report["checks"].append({"step":step,"backend":backend,
                    "single_cache_vs_full":compare(single,full),"batched_cache_vs_full":compare(batched,full),
                    "batched_vs_single_cache":compare(batched,single)})
                save()
            report["checks"].append({"step":step,"cross_backend":{
                label:compare(a,b) for label,a,b in zip(["full","single_cache","batched_cache"],outputs["fast"],outputs["reference"])}})
            save()
        report["status"]="complete"
        save()
        return report
    except Exception as error:
        report.update(status="failed",error=repr(error))
        save()
        raise
