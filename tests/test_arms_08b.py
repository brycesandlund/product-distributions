import pytest
from product_distributions.arms_08b import arm_config
from product_distributions.recovery import compatible


def test_pinned_arms():
    i, j, k = [arm_config(a) for a in "IJK"]
    for c in (i, j, k):
        assert c["steps"] == 256 and c["eval_every"] == c["checkpoint_every"] == 32
        assert c["prompts_per_step"] * c["group_size"] == 16
        assert c["max_new_tokens"] == 2048 and not c["teacher_privileged"]
        assert compatible({**c, "steps": 64}, c)
    assert i["teacher_model_id"] is None and i["alpha"] == 0
    assert j["teacher_model_id"] == k["teacher_model_id"] == "Qwen/Qwen3.5-9B"
    assert j["alpha"] == 0.5 and j["beta"] == 0 and j["loss"] == "imitation"
    assert k["alpha"] == 0 and k["loss"] == "opd_full" and k["group_size"] == 1
    with pytest.raises(ValueError):
        arm_config("H")
