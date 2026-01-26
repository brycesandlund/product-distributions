if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    print(tokenizer.eos_token)
    print(tokenizer.eos_token_id)