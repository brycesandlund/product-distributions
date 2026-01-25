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
    device: Optional[str] = None
) -> str:
    """
    Generate text using the product of two models' probability distributions.
    Uses KV cache for efficient generation.
    
    Args:
        model1: First PyTorch model (nn.Module)
        model2: Second PyTorch model (nn.Module)
        prompt1: Input prompt for model1
        prompt2: Input prompt for model2
        tokenizer1: Tokenizer for model1
        tokenizer2: Tokenizer for model2
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
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
    
    # Tokenize initial prompts
    input_ids1 = tokenizer1.encode(prompt1, return_tensors="pt").to(device)
    input_ids2 = tokenizer2.encode(prompt2, return_tensors="pt").to(device)
    
    # Initialize cache as None
    past_key_values1 = None
    past_key_values2 = None
    
    # Track sequence lengths for attention mask
    seq_len1 = input_ids1.shape[1]
    seq_len2 = input_ids2.shape[1]
    
    # Track generated tokens
    generated_tokens = []
    
    with torch.inference_mode():
        for step in range(max_new_tokens):
            # For first step, pass full sequence; for subsequent steps, only last token
            if step == 0:
                current_input1 = input_ids1
                current_input2 = input_ids2
                attention_mask1 = torch.ones_like(input_ids1)
                attention_mask2 = torch.ones_like(input_ids2)
            else:
                current_input1 = input_ids1[:, -1:]
                current_input2 = input_ids2[:, -1:]
                # Attention mask must cover full sequence including cached tokens
                attention_mask1 = torch.ones(1, seq_len1 + step, device=device, dtype=torch.long)
                attention_mask2 = torch.ones(1, seq_len2 + step, device=device, dtype=torch.long)
            
            # Get logits from both models with cache
            outputs1 = model1(
                current_input1,
                attention_mask=attention_mask1,
                past_key_values=past_key_values1,
                use_cache=True
            )
            outputs2 = model2(
                current_input2,
                attention_mask=attention_mask2,
                past_key_values=past_key_values2,
                use_cache=True
            )
            
            # Update cache for next iteration
            past_key_values1 = outputs1.past_key_values
            past_key_values2 = outputs2.past_key_values
            
            # Get logits for next token (last position)
            logits1 = outputs1.logits[:, -1, :] / temperature
            logits2 = outputs2.logits[:, -1, :] / temperature
            
            # Convert to probabilities
            probs1 = F.softmax(logits1, dim=-1)
            probs2 = F.softmax(logits2, dim=-1)
            
            # Compute product distribution
            product_probs = probs1 * probs2
            
            # Renormalize
            product_probs = product_probs / product_probs.sum(dim=-1, keepdim=True)
            
            # Sample from product distribution
            next_token = torch.multinomial(product_probs, num_samples=1)
            
            # Track and append to both sequences
            generated_tokens.append(next_token)
            input_ids1 = torch.cat([input_ids1, next_token], dim=1)
            input_ids2 = torch.cat([input_ids2, next_token], dim=1)
            
            # Stop on EOS token
            if next_token.item() == tokenizer1.eos_token_id:
                break
    
    # Decode the generated tokens
    generated_token_ids = torch.cat(generated_tokens, dim=1)
    generated_text = tokenizer1.decode(generated_token_ids[0], skip_special_tokens=True)
    
    return generated_text


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen3-1.7B"
    
    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use fp16 for memory efficiency
        attn_implementation="eager",  # Use eager attention for MPS compatibility
    )
    
    # Test prompts - using the same model twice for demonstration
    prompt1 = "The future of artificial intelligence is"
    prompt2 = "The future of artificial intelligence is"
    
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
        max_new_tokens=100,
        temperature=0.7
    )
    
    print(f"\nGenerated text:\n{generated_text}")

