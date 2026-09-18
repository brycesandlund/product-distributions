from unittest.mock import patch

from product_distributions.bigmath_calibration import select_examples, summarize


def test_selection_balanced_unique_and_deterministic():
    rows = [{"problem": f"q{i}", "answer": "1", "source": s}
            for s, start in (("a", 0), ("b", 10)) for i in range(start, start + 10)]
    rows.append(rows[0])
    with patch("product_distributions.bigmath_calibration.parse_gold"):
        selected, rejected = select_examples(rows, count=8)
        reordered, _ = select_examples(list(reversed(rows)), count=8)
    assert selected == reordered
    assert len({r["id"] for r in selected}) == 8
    assert sum(r["source"] == "a" for r in selected) == 4
    assert not rejected


def test_summary_distinguishes_truncated_correct_and_mixed_groups():
    rows = [{"example_id": "a", "reward": reward, "ended_with_eos": eos,
             "num_generated_tokens": 10} for reward, eos in ((1, False), (0, True))]
    result = summarize(rows)
    assert result["accuracy"] == 0.5
    assert result["completed_correct"] == 0
    assert result["truncated"] == 1
    assert result["mixed_groups"] == 1
