"""Matched A100-40GB comparisons; no changes to the active L40S deployment."""
import modal
from modal_training import training_image
from modal_app import volume_mounts

app = modal.App("product-distributions-08b-a100-preflight")
image = training_image.add_local_python_source("modal_training", "modal_08b_preflight")


@app.function(image=image, gpu="A100-40GB", cpu=4, memory=32768,
              timeout=7200, retries=0, volumes=volume_mounts)
def product_a100(run_name: str):
    from modal_08b_preflight import pilot
    return pilot("product", "product-a100-40gb", run_name)


@app.function(image=image, gpu="A100-40GB", cpu=4, memory=32768,
              timeout=7200, retries=0, volumes=volume_mounts)
def opd_a100(run_name: str):
    from modal_08b_preflight import pilot
    return pilot("opd", "opd-a100-40gb", run_name)
