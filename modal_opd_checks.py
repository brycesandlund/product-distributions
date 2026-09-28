"""Bounded, disposable end-to-end gradient diagnostics; no training successors."""
import modal
from modal_training import training_image
from modal_app import HF_CACHE_PATH, RESULTS_PATH, volume_mounts, results_volume, kernel_cache

app = modal.App("product-distributions-opd-checks-precision")
image = training_image.add_local_python_source("modal_training")


@app.function(image=image, gpu="A10", cpu=4, memory=32768, timeout=1800, volumes=volume_mounts)
def check():
    import inspect
    import json
    from pathlib import Path
    from time import perf_counter
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from product_distributions.training import Trainer, TrainConfig, completion_logits, write_json
    from product_distributions.losses import full_vocab_opd_loss
    from product_distributions.data import Example
    from product_distributions.short_reasoning import CONCISE

    root = Path(RESULTS_PATH) / "diagnostics/opd-correctness-precision-20260928"
    root.mkdir(parents=True, exist_ok=False)
    report = {"status": "loading", "checks": {}}
    started = perf_counter()

    def save():
        report["elapsed_seconds"] = perf_counter()-started
        write_json(root / "report.json", report)
        results_volume.commit()
        print(json.dumps({k:v for k,v in report.items() if k != "sources"}), flush=True)

    original_chunk = modeling.torch_chunk_gated_delta_rule
    original_recurrent = modeling.torch_recurrent_gated_delta_rule
    # Execute the installed reference body without its automatic hub-kernel decorator.
    import ast
    tree = ast.parse(inspect.getsource(original_chunk))
    tree.body[0].decorator_list = []
    namespace = dict(vars(modeling))
    exec(compile(tree, "<explicit_torch_reference>", "exec"), namespace)
    reference_body = namespace["torch_chunk_gated_delta_rule"]
    reference_calls = [0]
    def reference_chunk(*args, **kwargs):
        reference_calls[0] += 1
        return reference_body(*args, **kwargs)
    try:
        report["sources"] = {"reference_chunk": inspect.getsource(original_chunk),
                             "loss": inspect.getsource(full_vocab_opd_loss)}
        save()
        cfg = TrainConfig(model_id="Qwen/Qwen3.5-4B", model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                          loss="opd_full", alpha=0, group_size=1, prompts_per_step=1,
                          enable_thinking=False, instruction=CONCISE, learning_rate=2e-5)
        t = Trainer(cfg, HF_CACHE_PATH)
        fast_chunk = modeling.torch_chunk_gated_delta_rule
        fast_recurrent = modeling.torch_recurrent_gated_delta_rule
        report["sources"]["gated_delta_forward"] = inspect.getsource(modeling.Qwen3_5GatedDeltaNet.forward)
        params = [p for p in t.model.parameters() if p.requires_grad]
        initial = [p.detach().cpu().clone() for p in params]
        e = Example("diagnostic", "Joan has 8 orange balloons but lost 2. How many remain?", "6")
        student, teacher = t._prompts(e)
        tokens = t.tokenizer.encode("Joan starts with 8 balloons and loses 2, so 8 - 2 = 6.\n\\boxed{6}", add_special_tokens=False) * 6
        report.update(status="checking", tokens=len(tokens), prompts={"student":student,"teacher":teacher},
                      gpu=torch.cuda.get_device_name())

        def mode(fast=True, checkpointing=True):
            torch.set_float32_matmul_precision("high" if fast else "highest")
            modeling.torch_chunk_gated_delta_rule = fast_chunk if fast else reference_chunk
            modeling.torch_recurrent_gated_delta_rule = fast_recurrent if fast else original_recurrent
            if checkpointing:
                t.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            else:
                t.model.gradient_checkpointing_disable()

        def probability_difference(left,right):
            lp,rp=left.float().log_softmax(-1),right.float().log_softmax(-1)
            return {"mean_kl_production_to_variant":(lp.exp()*(lp-rp)).sum(-1).mean().item(),
                    "top1_agreement":(lp.argmax(-1)==rp.argmax(-1)).float().mean().item(),
                    "max_logprob_difference":(lp-rp).abs().max().item()}

        def gradients(target):
            t.model.train()
            t.optimizer.zero_grad(set_to_none=True)
            logits = completion_logits(t.model,t.tokenizer,student,tokens)
            loss = full_vocab_opd_loss(logits,target,0,len(tokens))
            t.accelerator.backward(loss)
            g = torch.cat([(p.grad.detach().float().cpu() if p.grad is not None else torch.zeros_like(p,device="cpu")).flatten() for p in params])
            return loss.item(),g,logits.detach().cpu()

        mode()
        t.model.eval()
        with torch.no_grad():
            identical = completion_logits(t.model,t.tokenizer,student,tokens).detach()
            target = completion_logits(t.model,t.tokenizer,teacher,tokens).detach()
        value,g,out = gradients(identical)
        report["checks"]["identical_context"] = {"kl":value,"grad_norm":g.norm().item(),
            "max_logit_difference":(out-identical.cpu()).abs().max().item()}
        del g,out,identical
        save()

        base_loss,base_grad,base_logits = gradients(target)
        report["checks"]["production_gradient"] = {"loss":base_loss,"norm":base_grad.norm().item(),
                                                    "finite":bool(torch.isfinite(base_grad).all())}
        for name,fast,ckpt in (("no_checkpoint",True,False),("reference_kernel",False,False)):
            try:
                mode(fast,ckpt)
                value,g,out = gradients(target)
                report["checks"][name] = {"loss":value,"grad_norm":g.norm().item(),
                    "relative_gradient_error":((g-base_grad).norm()/base_grad.norm().clamp_min(1e-20)).item(),
                    "gradient_cosine":torch.nn.functional.cosine_similarity(g.double(),base_grad.double(),dim=0).item(),
                    "max_logit_difference":(out-base_logits).abs().max().item(),
                    **probability_difference(base_logits,out)}
                del g,out
            except Exception as error:
                report["checks"][name] = {"error":repr(error)}
            save()
        mode()
        # A separate disposable AdamW optimizer and reset weights for each rate.
        for lr in (2e-6,2e-5):
            with torch.no_grad():
                for p,v in zip(params,initial): p.copy_(v.to(p.device))
            opt = torch.optim.AdamW(params,lr=lr,weight_decay=0)
            losses = []
            for step in range(4):
                t.model.train()
                opt.zero_grad(set_to_none=True)
                logits = completion_logits(t.model,t.tokenizer,student,tokens)
                loss = full_vocab_opd_loss(logits,target,0,len(tokens))
                losses.append(loss.item())
                if step<3:
                    t.accelerator.backward(loss)
                    opt.step()
                del logits,loss
            report["checks"][f"frozen_target_lr_{lr}"] = losses
            del opt
            save()
        # Compare again with nonzero LoRA B, so gradients through LoRA A are exercised.
        mode()
        base_loss,base_grad,base_logits = gradients(target)
        for name,fast,ckpt in (("updated_no_checkpoint",True,False),("updated_reference_kernel",False,False)):
            try:
                mode(fast,ckpt)
                value,g,out = gradients(target)
                report["checks"][name] = {"loss":value,"production_loss":base_loss,
                    "relative_gradient_error":((g.double()-base_grad.double()).norm()/base_grad.double().norm().clamp_min(1e-20)).item(),
                    "gradient_cosine":torch.nn.functional.cosine_similarity(g.double(),base_grad.double(),dim=0).item(),
                    "max_logit_difference":(out-base_logits).abs().max().item(),
                    **probability_difference(base_logits,out)}
                del g,out
            except Exception as error:
                report["checks"][name] = {"error":repr(error)}
            save()
        report["reference_forward_calls"] = reference_calls[0]
        with torch.no_grad():
            for p,v in zip(params,initial): p.copy_(v.to(p.device))
        report["status"] = "complete"
        save()
        return {k:v for k,v in report.items() if k!="sources"}
    except Exception as error:
        report.update(status="failed",error=repr(error))
        save()
        raise
    finally:
        kernel_cache.commit()
