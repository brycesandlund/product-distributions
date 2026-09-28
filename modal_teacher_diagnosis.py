"""Read-only teacher/gradient diagnostics at initial and failed adapter weights."""
import modal
from modal_training import training_image
from modal_app import HF_CACHE_PATH, RESULTS_PATH, volume_mounts, results_volume, kernel_cache

app = modal.App("product-distributions-teacher-diagnosis")
image = training_image.add_local_python_source("modal_training")


@app.function(image=image, gpu="A10", cpu=4, memory=32768, timeout=1800, volumes=volume_mounts)
def diagnose():
    import json
    from pathlib import Path
    from time import perf_counter
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from product_distributions.training import Trainer, TrainConfig, completion_logits, write_json
    from product_distributions.data import read_examples, verify_answer
    from product_distributions.sampling import SamplingConfig

    started=perf_counter()
    run=Path(RESULTS_PATH)/"runs/bigmath-4b-opd-512-20260928/segment-000064-attempt-70c362a5e022"
    out=Path(RESULTS_PATH)/"diagnostics/teacher-drift-20260928"
    out.mkdir(parents=True,exist_ok=False)
    cfg=TrainConfig(**json.loads((run/"config.json").read_text()))
    t=Trainer(cfg,HF_CACHE_PATH)
    examples=read_examples(Path(RESULTS_PATH)/"data/bigmath-opd-8192-v1/eval.jsonl")
    ids=["b71b9802fe75f97791917b0d9cd9acd5f3468fdf8ecd640a37fe9af8b9c31eef",
         "508f6f8e107e7f13c7a8d9994fc291ba67644057e024c88cf8912fcd66a608d6",
         "3742231fdea6a83f903aa097caa3b2f86754e3e8b0b2a6bdcbee9ed5e2fa371d"]
    examples=[next(e for e in examples if e.id==i) for i in ids]
    before={r["example_id"]:r for r in json.loads((run/"eval-before.json").read_text())["records"]}
    after={r["example_id"]:r for r in json.loads((run/"eval-000032.json").read_text())["records"]}
    report={"status":"running","states":{},"questions":[vars(e) for e in examples],
            "note":"Three selected failure cases, not a representative evaluation. No optimizer steps. Logit gradients are not parameter gradients."}

    def save():
        report["elapsed_seconds"]=perf_counter()-started
        write_json(out/"report.json",report)
        results_volume.commit()

    def entries(values,slog,qlog,k=10,absolute=False):
        indices=(values.abs() if absolute else values).topk(k).indices.tolist()
        return [{"id":i,"token":t.tokenizer.decode([i]),"value":values[i].item(),
                 "student_p":slog[i].exp().item(),"teacher_p":qlog[i].exp().item()} for i in indices]

    try:
        for state in (0,32):
            if state:
                result=set_peft_model_state_dict(t.accelerator.unwrap_model(t.model),
                    load_file(str(run/"checkpoint-000032/adapter/adapter_model.safetensors")))
                report["adapter_unexpected_keys"]=list(result.unexpected_keys)
                report["adapter_missing_lora_keys"]=[k for k in result.missing_keys if "lora_" in k]
                assert not report["adapter_unexpected_keys"] and not report["adapter_missing_lora_keys"]
            t.model.eval()
            rows=[]
            report["states"][str(state)]={"prefix_checks":rows,"generations":[]}
            for e in examples:
                student,teacher=t._prompts(e)
                for label,text in (("correct_step0",before[e.id]["text"]),("failed_step32",after[e.id]["text"])):
                    prefix=t.tokenizer.encode(text,add_special_tokens=False)[:128]
                    # Add a dummy prediction target: logits now include next-token distribution after the prefix.
                    with torch.no_grad():
                        sl=completion_logits(t.model,t.tokenizer,student,prefix+[0]).float().log_softmax(-1)
                        ql=completion_logits(t.model,t.tokenizer,teacher,prefix+[0]).float().log_softmax(-1)
                        p=sl.exp()
                        ratio=sl-ql
                        kl=(p*ratio).sum(-1)
                        grad=p*(ratio-kl[:,None])
                        mass=grad.abs().sum(0)
                        top=mass.topk(15).indices.tolist()
                        checks=[]
                        positions=sorted({0,min(8,len(prefix)),len(prefix)})
                        for pos in positions:
                            checks.append({"position":pos,"prefix":t.tokenizer.decode(prefix[:pos]),"kl":kl[pos].item(),
                                "teacher_top":entries(ql[pos].exp(),sl[pos],ql[pos]),
                                "increase_under_logit_descent":entries(-grad[pos],sl[pos],ql[pos]),
                                "decrease_under_logit_descent":entries(grad[pos],sl[pos],ql[pos])})
                        rows.append({"example_id":e.id,"prefix_source":label,"positions":len(prefix)+1,
                            "mean_kl":kl.mean().item(),"mean_abs_logit_gradient":grad.abs().sum(-1).mean().item(),
                            "top_gradient_mass":[{"id":i,"token":t.tokenizer.decode([i]),"absolute_mass":mass[i].item(),"signed_sum":grad[:,i].sum().item()} for i in top],
                            "checks":checks})
                    del sl,ql,p,ratio,kl,grad,mass
                save()
            # Generate from both contexts directly; alpha=0 is just the efficient single-context path.
            prompts=[prompt for e in examples for prompt in t._prompts(e)]
            results=t.sampler.generate(prompts,prompts,config=SamplingConfig(teacher_weight=0,temperature=1,
                top_k=None,top_p=None,max_new_tokens=512,seed=123,extra_eos_token_ids=(248044,)))
            for i,r in enumerate(results):
                report["states"][str(state)]["generations"].append({"example_id":examples[i//2].id,
                    "context":"student" if i%2==0 else "privileged_teacher",**r.to_dict(),
                    **verify_answer(r.text,examples[i//2].answer)})
            save()
        report["status"]="complete"
        save()
        return {"status":report["status"],"elapsed_seconds":report["elapsed_seconds"],"output":str(out)}
    except Exception as error:
        report.update(status="failed",error=repr(error))
        save()
        raise
    finally:
        kernel_cache.commit()
