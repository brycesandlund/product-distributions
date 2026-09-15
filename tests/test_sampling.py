from types import SimpleNamespace

import torch

from product_distributions.sampling import ProductSampler, SamplingConfig


class FakeTokenizer:
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 7

    def __call__(self, prompts, **_kwargs):
        rows = [[1, 4] if "teacher" in prompt else [1] for prompt in prompts]
        width = max(map(len, rows))
        input_ids = [[0] * (width - len(row)) + row for row in rows]
        attention_mask = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }

    def get_vocab(self):
        return {str(index): index for index in range(8)}

    def decode(self, token_ids, **_kwargs):
        return " ".join(str(token) for token in token_ids if token != self.eos_token_id)


class FakeModel(torch.nn.Module):
    def __init__(self, preferred_token):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 2)
        self.preferred_token = preferred_token
        self.calls = 0

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, past_key_values=None, **_kwargs):
        self.calls += 1
        token = self.preferred_token if past_key_values is None else 7
        logits = torch.full((input_ids.shape[0], 1, 8), -100.0)
        logits[..., token] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=self.calls)


def config(teacher_weight):
    return SamplingConfig(
        teacher_weight=teacher_weight,
        top_k=1,
        top_p=None,
        max_new_tokens=3,
    )


def test_shared_model_batches_both_contexts_in_one_forward_per_step():
    tokenizer = FakeTokenizer()
    model = FakeModel(preferred_token=2)

    results = ProductSampler(model, tokenizer).generate(
        ["student one", "student two"],
        ["teacher one", "teacher two"],
        config=config(teacher_weight=0.5),
    )

    assert [result.token_ids for result in results] == [[2, 7], [2, 7]]
    assert model.calls == 2


def test_distinct_models_keep_independent_caches_and_use_teacher_logits():
    tokenizer = FakeTokenizer()
    student = FakeModel(preferred_token=2)
    teacher = FakeModel(preferred_token=3)

    result = ProductSampler(
        student,
        tokenizer,
        teacher,
        FakeTokenizer(),
    ).generate(
        "student",
        "teacher",
        config=config(teacher_weight=1.0),
    )[0]

    assert result.token_ids == [3, 7]
    assert student.calls == 2
    assert teacher.calls == 2


def test_distinct_token_vocabularies_are_rejected():
    student_tokenizer = FakeTokenizer()
    teacher_tokenizer = FakeTokenizer()
    teacher_tokenizer.get_vocab = lambda: {"different": 0}

    try:
        ProductSampler(
            FakeModel(2),
            student_tokenizer,
            FakeModel(3),
            teacher_tokenizer,
        )
    except ValueError as error:
        assert "identical token-to-ID vocabularies" in str(error)
    else:
        raise AssertionError("expected incompatible tokenizers to fail")


def test_recorded_logprobs_are_full_support_sampling_probabilities():
    tokenizer = FakeTokenizer()
    model = FakeModel(preferred_token=2)
    result = ProductSampler(model, tokenizer).generate(
        "student",
        "teacher",
        config=SamplingConfig(
            top_k=None,
            top_p=None,
            record_logprobs=True,
            max_new_tokens=3,
        ),
    )[0]
    assert result.token_ids == [2, 7]
    assert result.student_logprobs == [0.0, 0.0]
    assert result.teacher_logprobs == [0.0, 0.0]
    assert result.behavior_logprobs == [0.0, 0.0]
