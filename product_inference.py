import torch
import torch.nn.functional as F
from typing import Optional


def product_distribution_inference(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    prompt1: str,
    prompt2: str,
    tokenizer1,
    tokenizer2,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    lam: float = 0.5,
    device: Optional[str] = None
) -> str:
    """
    Generate text using a weighted product of two models' probability distributions.
    Uses KV cache for efficient generation.
    
    The combined distribution is: P ∝ P1^λ * P2^(1-λ)
    
    Args:
        model1: First PyTorch model (nn.Module)
        model2: Second PyTorch model (nn.Module)
        prompt1: Input prompt for model1
        prompt2: Input prompt for model2
        tokenizer1: Tokenizer for model1
        tokenizer2: Tokenizer for model2
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        lam: Interpolation weight (0 <= lam <= 1). lam=1 uses only model1,
             lam=0 uses only model2, lam=0.5 weights both equally
        device: Device to run inference on (auto-detects if None)
        
    Returns:
        The generated text (shared tokens sampled from product distribution)
    """
    # Auto-detect device with MPS support
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    model1.eval()
    model2.eval()
    model1.to(device)
    model2.to(device)
    
    # Enable cache
    model1.config.use_cache = True
    model2.config.use_cache = True
    
    # Check if we can skip duplicate model calls (same model AND same prompt)
    same_model_and_prompt = (model1 is model2) and (prompt1 == prompt2)
    
    # Tokenize initial prompts
    # Shape: (1, seq_len1) and (1, seq_len2) - batch size 1, variable sequence lengths
    input_ids1 = tokenizer1.encode(prompt1, return_tensors="pt").to(device)
    input_ids2 = tokenizer2.encode(prompt2, return_tensors="pt").to(device)
    
    # Initialize cache as None
    past_key_values1 = None
    past_key_values2 = None
    
    # Track sequence lengths for attention mask
    seq_len1 = input_ids1.shape[1]
    seq_len2 = input_ids2.shape[1]
    
    # Pre-allocate attention masks at max size to avoid repeated allocations
    # Shape: (1, max_seq_len) - we slice these during generation
    max_seq_len1 = seq_len1 + max_new_tokens
    max_seq_len2 = seq_len2 + max_new_tokens
    attention_mask1_full = torch.ones(1, max_seq_len1, device=device, dtype=torch.long)
    attention_mask2_full = torch.ones(1, max_seq_len2, device=device, dtype=torch.long)
    
    # Pre-allocate buffer for generated tokens
    # Shape: (1, max_new_tokens) - stores generated token IDs
    generated_buffer = torch.zeros(1, max_new_tokens, device=device, dtype=torch.long)
    num_generated = 0
    
    with torch.inference_mode():
        for step in range(max_new_tokens):
            # For first step, pass full sequence; for subsequent steps, only last token
            if step == 0:
                # current_input shape: (1, seq_len) - full prompt
                # attention_mask shape: (1, seq_len) - all ones for prompt tokens
                current_input1 = input_ids1
                current_input2 = input_ids2
                attention_mask1 = attention_mask1_full[:, :seq_len1]
                attention_mask2 = attention_mask2_full[:, :seq_len2]
            else:
                # current_input shape: (1, 1) - just the last generated token
                # attention_mask shape: (1, seq_len + step) - grows by 1 each step
                current_input1 = next_token
                current_input2 = next_token
                # Slice pre-allocated attention mask
                attention_mask1 = attention_mask1_full[:, :seq_len1 + step]
                attention_mask2 = attention_mask2_full[:, :seq_len2 + step]
            
            # Get logits from model(s) with cache
            # outputs.logits shape: (1, input_seq_len, vocab_size)
            #   - Step 0: (1, seq_len, vocab_size)
            #   - Step 1+: (1, 1, vocab_size) since we only pass the last token
            outputs1 = model1(
                current_input1,
                attention_mask=attention_mask1,
                past_key_values=past_key_values1,
                use_cache=True
            )
            
            # Skip second model call if same model and prompt
            if same_model_and_prompt:
                outputs2 = outputs1
            else:
                outputs2 = model2(
                    current_input2,
                    attention_mask=attention_mask2,
                    past_key_values=past_key_values2,
                    use_cache=True
                )
            
            # Update cache for next iteration
            # past_key_values: tuple of (num_layers) tuples, each containing:
            #   (key, value) tensors of shape (batch, num_heads, seq_len, head_dim)
            past_key_values1 = outputs1.past_key_values
            if not same_model_and_prompt:
                past_key_values2 = outputs2.past_key_values
            
            # Get logits for next token (last position)
            # Shape: (1, vocab_size) - select last token's logits from each model
            logits1 = outputs1.logits[:, -1, :] / temperature
            logits2 = outputs2.logits[:, -1, :] / temperature
            
            # Compute weighted product distribution in log space (more efficient & stable)
            # P ∝ P1^λ * P2^(1-λ) => log P ∝ λ*log(P1) + (1-λ)*log(P2)
            # Shape: (1, vocab_size) for all log_probs and combined
            log_probs1 = F.log_softmax(logits1, dim=-1)
            log_probs2 = F.log_softmax(logits2, dim=-1)
            combined_log_probs = lam * log_probs1 + (1 - lam) * log_probs2
            
            # Convert back to probabilities (softmax normalizes automatically)
            # Shape: (1, vocab_size) - normalized probability distribution
            product_probs = F.softmax(combined_log_probs, dim=-1)
            
            # Sample from product distribution
            # Shape: (1, 1) - single sampled token ID
            next_token = torch.multinomial(product_probs, num_samples=1)
            
            # Store in pre-allocated buffer
            # next_token.squeeze(-1) shape: (1,) - remove last dim for assignment
            generated_buffer[:, num_generated] = next_token.squeeze(-1)
            num_generated += 1
            
            # Stop on EOS token
            if next_token.item() == tokenizer1.eos_token_id:
                break
    
    # Decode the generated tokens
    # Shape: (1, num_generated) - slice buffer to only include actual generated tokens
    generated_token_ids = generated_buffer[:, :num_generated]
    # Shape: (num_generated,) - remove batch dimension for decoder
    generated_text = tokenizer1.decode(generated_token_ids[0], skip_special_tokens=True)
    
    return generated_text


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen3-1.7B"
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    # Use Flash Attention on CUDA, eager on MPS
    attn_impl = "eager" if device == "mps" else "flash_attention_2"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use fp16 for memory efficiency
        attn_implementation=attn_impl,
    )

    numbers_str = "1, 2, 5"
    target = 10

    prompt_content_1 = f"Using the numbers {numbers_str} exactly once, create a mathematical expression using +, -, *, /, and/or () that equals {target}. Please reason step by step, and put your final expression in <answer></answer> tags, for example, <answer>4*5-4</answer>."
    prompt_content_2 = prompt_content_1 + " The answer is 1*2*5=10."
    
    # Format with chat template so the model knows where instructions end
    messages_1 = [{"role": "user", "content": prompt_content_2}]
    messages_2 = [{"role": "user", "content": prompt_content_2}]
    formatted_prompt_1 = tokenizer.apply_chat_template(
        messages_1, 
        add_generation_prompt=True, 
        tokenize=False
    )
    formatted_prompt_2 = tokenizer.apply_chat_template(
        messages_2, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    # Test prompts - using the same model twice for demonstration
    prompt1 = formatted_prompt_1
    prompt2 = formatted_prompt_2
    
    print(f"\nPrompt 1: {prompt1}")
    print(f"Prompt 2: {prompt2}")   
    print("\nGenerating with product distribution...")
    
    # Run product distribution inference
    # Using the same model and tokenizer for both since we're testing with one model
    generated_text = product_distribution_inference(
        model1=model,
        model2=model,
        prompt1=prompt1,
        prompt2=prompt2,
        tokenizer1=tokenizer,
        tokenizer2=tokenizer,
        max_new_tokens=250,
        temperature=0.7,
        lam=0.5
    )
    
    print(f"\nGenerated text:\n{generated_text}")

