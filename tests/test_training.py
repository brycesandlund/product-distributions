from types import SimpleNamespace

import torch

from product_distributions.training import TrainConfig, Trainer, completion_logits


def test_completion_alignment_excludes_prompt_and_predicts_eos():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [1, 2, 3]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(9, 2)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, attention_mask, logits_to_keep, **kwargs):
            assert input_ids.tolist() == [[1, 2, 3, 4, 5]]
            assert attention_mask.tolist() == [[1, 1, 1, 1, 1]]
            # The model predicts token n+1 after input token n.
            logits = torch.nn.functional.one_hot(input_ids + 1, num_classes=9).float()
            return SimpleNamespace(logits=logits[:, -logits_to_keep:])

    logits = completion_logits(Model(), Tokenizer(), "prompt", [4, 5, 6])
    assert logits.argmax(-1).tolist() == [4, 5, 6]


def test_config_rejects_invalid_sampling_and_loss():
    import pytest

    for override in (
        {"alpha": 2},
        {"group_size": 1},
        {"max_new_tokens": 0},
        {"loss": "unknown"},
    ):
        with pytest.raises(ValueError):
            TrainConfig(**override).validate()


def test_resume_rejects_old_imitation_weight_semantics(tmp_path):
    import pytest

    (tmp_path / "state.json").write_text("{}")
    trainer = object.__new__(Trainer)
    trainer.config = TrainConfig(loss="imitation")
    with pytest.raises(ValueError, match="old weight semantics"):
        trainer.resume(str(tmp_path))
