import pytest

from product_distributions.calibration import summarize


def test_summary_separates_completion_and_success():
    rows = [
        {
            "example_id": "a",
            "reward": 1,
            "ended_with_eos": False,
            "has_boxed_answer": True,
            "num_tokens": 20,
        },
        {
            "example_id": "a",
            "reward": 0,
            "ended_with_eos": True,
            "has_boxed_answer": False,
            "num_tokens": 10,
        },
        {
            "example_id": "b",
            "reward": 1,
            "ended_with_eos": True,
            "has_boxed_answer": True,
            "num_tokens": 8,
        },
        {
            "example_id": "b",
            "reward": 1,
            "ended_with_eos": True,
            "has_boxed_answer": True,
            "num_tokens": 6,
        },
    ]
    result = summarize(rows)
    assert result["eos_fraction"] == 0.75
    assert result["reward_mean"] == 0.75
    assert result["completed_correct_fraction"] == 0.5
    assert result["mixed_success_groups"] == 1
    assert result["all_successful_groups"] == 1
    assert result["all_failed_groups"] == 0
    assert result["mean_tokens"] == 11
    with pytest.raises(ValueError):
        summarize([])


def test_calibration_is_generation_only(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from product_distributions import calibration
    from product_distributions.data import Example

    calls = []

    class FakeTrainer:
        def __init__(self, config, cache_dir):
            self.model = self.teacher = SimpleNamespace(eval=lambda: None)
            self.tokenizer = SimpleNamespace(eos_token_id=9)
            self.sampler = SimpleNamespace(generate=self.generate)
            self.step = 0

        def _prompts(self, example):
            return "question", "question + answer"

        def generate(self, students, teachers, *, config):
            assert config.top_k is None and config.top_p is None
            assert config.temperature == 1.0
            calls.append((config.max_new_tokens, config.seed))
            return [
                SimpleNamespace(text="answer", token_ids=[1, 9], num_generated_tokens=2)
                for _ in students
            ]

    monkeypatch.setattr(calibration, "Trainer", FakeTrainer)
    monkeypatch.setattr(
        calibration, "read_examples", lambda path: [Example("a", "q", "a")]
    )
    monkeypatch.setattr(
        calibration,
        "verify_answer",
        lambda text, answer: {"reward": 1, "status": "correct"},
    )
    for name in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(calibration.torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(calibration.torch.cuda, name, lambda: 1024)
    monkeypatch.setattr(calibration.torch.cuda, "get_device_name", lambda index: "fake")
    (tmp_path / "manifest.json").write_text("{}")
    output = tmp_path / "out"
    result = calibration.run_calibration(
        output, tmp_path, "cache", lengths=(2, 4), questions=1, group_size=2
    )
    assert result["optimizer_steps"] == 0
    assert calls == [(2, 42), (4, 42)]
    assert len((output / "rollouts.jsonl").read_text().splitlines()) == 4
    assert len(json.loads((output / "summary.json").read_text())) == 2
