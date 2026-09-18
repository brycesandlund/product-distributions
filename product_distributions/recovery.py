"""Non-destructive selection of committed training checkpoints."""

import json
from pathlib import Path
from uuid import uuid4


def compatible(left, right):
    from dataclasses import asdict
    from .training import TrainConfig

    def normalize(config):
        result = asdict(TrainConfig(**config))
        result.pop("steps")
        result["extra_eos_token_ids"] = list(result["extra_eos_token_ids"])
        return result

    return normalize(left) == normalize(right)


def checkpoint_state(path, config):
    path = Path(path)
    if not path.name.startswith("checkpoint-"):
        return None
    required = ("state.json", "adapter/adapter_model.safetensors", "optimizer.bin", "random_states_0.pkl")
    if not all((path / name).is_file() and (path / name).stat().st_size for name in required):
        return None
    try:
        state = json.loads((path / "state.json").read_text())
        if not compatible(state["config"], config) or state.get("imitation_weighting") != "reward_interpolation_v1":
            return None
        if not 0 <= state["step"] <= config["steps"]:
            return None
        return state
    except (ValueError, KeyError, TypeError):
        return None


def recovery_plan(root, config, resume=None):
    root = Path(root)
    candidates = []
    if resume:
        state = checkpoint_state(resume, config)
        if state is None:
            raise ValueError("Explicit resume checkpoint is incomplete or incompatible")
        candidates.append((state["step"], str(resume)))
    for marker in root.glob("**/checkpoint-*/state.json"):
        state = checkpoint_state(marker.parent, config)
        if state is not None:
            candidates.append((state["step"], str(marker.parent)))
    previous, selected = max(candidates) if candidates else (0, None)
    target = min(((previous // 64) + 1) * 64, config["steps"])
    return {
        "previous": previous, "target": target, "resume": selected,
        "output_dir": str(root / f"segment-{target:06d}-attempt-{uuid4().hex[:12]}"),
    }
