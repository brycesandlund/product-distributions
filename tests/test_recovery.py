import json
from dataclasses import asdict

from product_distributions.recovery import recovery_plan
from product_distributions.training import TrainConfig


def checkpoint(root, step, config, complete=True):
    path = root / f"segment-{step:06d}" / f"checkpoint-{step:06d}"
    path.mkdir(parents=True)
    (path / "state.json").write_text(json.dumps({
        "step": step, "config": config, "imitation_weighting": "reward_interpolation_v1"}))
    if complete:
        (path / "adapter").mkdir()
        for name in ("adapter/adapter_model.safetensors", "optimizer.bin", "random_states_0.pkl"):
            (path / name).write_bytes(b"test")
    return path


def test_restart_preserves_failed_segment_and_picks_latest_complete_checkpoint(tmp_path):
    config = asdict(TrainConfig(steps=512))
    old = checkpoint(tmp_path, 192, {**config, "steps": 192})
    latest = checkpoint(tmp_path, 224, {**config, "steps": 256})
    checkpoint(tmp_path, 256, config, complete=False)
    plan = recovery_plan(tmp_path, config, str(old))
    assert plan["resume"] == str(latest)
    assert plan["previous"] == 224 and plan["target"] == 256
    assert plan["output_dir"] != recovery_plan(tmp_path, config, str(old))["output_dir"]
    assert (tmp_path / "segment-000256").exists()


def test_ignore_incompatible_and_temporary_checkpoints(tmp_path):
    config = asdict(TrainConfig(steps=512))
    good = checkpoint(tmp_path, 192, config)
    checkpoint(tmp_path, 224, {**config, "alpha": 0.9})
    incomplete = checkpoint(tmp_path, 256, config)
    incomplete.rename(incomplete.with_name(".checkpoint-000256.incomplete-test"))
    plan = recovery_plan(tmp_path, config)
    assert plan["resume"] == str(good)


def test_complete_run_is_not_extended(tmp_path):
    config = asdict(TrainConfig(steps=512))
    checkpoint(tmp_path, 512, config)
    plan = recovery_plan(tmp_path, config)
    assert plan["previous"] == plan["target"] == 512


def test_reject_invalid_explicit_resume(tmp_path):
    import pytest
    config = asdict(TrainConfig(steps=512))
    bad = checkpoint(tmp_path, 192, config, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        recovery_plan(tmp_path, config, str(bad))
