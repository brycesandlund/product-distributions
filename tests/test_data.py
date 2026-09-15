from product_distributions.data import Example, last_boxed, prompts, verify_answer


def test_verifier_final_answer_and_symbolic_equivalence():
    assert (
        verify_answer(r"Reasoning: 1. Final: \boxed{\frac{1}{2}}", "0.5")["reward"] == 1
    )
    assert verify_answer(r"\boxed{2} then \boxed{3}", "2")["reward"] == 0
    assert verify_answer("The answer might be 2", "2")["reward"] == 0
    assert last_boxed(r"\boxed{\frac{1}{2}}") == r"\boxed{\frac{1}{2}}"


def test_privileged_answer_never_enters_student_prompt():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    student, teacher = prompts(
        Tokenizer(), Example("id", "Compute the secret", "7391"), False
    )
    assert "7391" not in student
    assert "7391" in teacher
