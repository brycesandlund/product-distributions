"""
Evaluate two models on the MATH dataset using product distribution inference.
"""

import torch
import re
import json
from typing import Optional
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from product_inference import product_distribution_inference


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the answer from \\boxed{...} format used in MATH dataset."""
    # Match \boxed{...} with nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_answer_tag(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    if answer is None:
        return ""
    # Remove whitespace
    answer = answer.strip()
    # Remove common LaTeX formatting
    answer = answer.replace("\\!", "")
    answer = answer.replace("\\,", "")
    answer = answer.replace("\\;", "")
    answer = answer.replace("\\:", "")
    answer = answer.replace("\\ ", "")
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace("\\text", "")
    answer = answer.replace("\\mathrm", "")
    answer = answer.replace("\\dfrac", "\\frac")
    # Remove spaces
    answer = answer.replace(" ", "")
    return answer


def answers_match(pred: str, gold: str) -> bool:
    """Check if predicted and gold answers match."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    return pred_norm == gold_norm


def format_prompt(problem: str, tokenizer) -> str:
    """Format a MATH problem as a chat prompt."""
    prompt_content = (
        f"{problem}\n\n"
        "Please solve this problem step by step. "
        "Put your final answer in \\boxed{{}} format."
    )
    
    messages = [{"role": "user", "content": prompt_content}]
    formatted = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    return formatted


def evaluate_math(
    model1_name: str,
    model2_name: str,
    num_samples: Optional[int] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    lam: float = 0.5,
    split: str = "test",
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
):
    """
    Evaluate two models on MATH dataset using product distribution inference.
    
    Args:
        model1_name: HuggingFace model name for first model
        model2_name: HuggingFace model name for second model
        num_samples: Number of samples to evaluate (None = all)
        max_new_tokens: Maximum tokens to generate per problem
        temperature: Sampling temperature
        lam: Interpolation weight (0.5 = equal weighting)
        split: Dataset split to use ("test" or "train")
        device: Device to run on (auto-detects if None)
        dtype: Model dtype (float16 or bfloat16)
    """
    # All available subareas in MATH dataset
    MATH_SUBAREAS = [
        'algebra',
        'counting_and_probability', 
        'geometry',
        'intermediate_algebra',
        'number_theory',
        'prealgebra',
        'precalculus'
    ]
    
    print(f"Loading MATH dataset (split: {split})...")
    
    # Load and concatenate all subareas
    from datasets import concatenate_datasets
    datasets_list = []
    for subarea in MATH_SUBAREAS:
        print(f"  Loading {subarea}...")
        ds = load_dataset("EleutherAI/hendrycks_math", subarea, split=split)
        # Add subarea as a column for tracking
        ds = ds.add_column("subarea", [subarea] * len(ds))
        datasets_list.append(ds)
    
    dataset = concatenate_datasets(datasets_list)
    print(f"  Total: {len(dataset)} problems")
    
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} problems")
    
    # Load models and tokenizers
    print(f"\nLoading model 1: {model1_name}")
    tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
    
    # Use Flash Attention on CUDA, eager on MPS (for compatibility)
    attn_impl = "eager" if device == "mps" else "flash_attention_2"
    print(f"Using attention implementation: {attn_impl}")
    
    model1 = AutoModelForCausalLM.from_pretrained(
        model1_name,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    
    if model1_name == model2_name:
        print("Using same model for both (self-product)")
        tokenizer2 = tokenizer1
        model2 = model1
    else:
        print(f"Loading model 2: {model2_name}")
        tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_name,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
    
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()
    
    # Compile models for faster inference (PyTorch 2.0+)
    # Skip on MPS as torch.compile support is limited
    if device != "mps":
        print("Compiling models with torch.compile...")
        model1 = torch.compile(model1)
        if model1_name != model2_name:
            model2 = torch.compile(model2)
    
    # Evaluate
    results = []
    correct = 0
    
    print("\nRunning evaluation...")
    for i, example in enumerate(tqdm(dataset)):
        problem = example["problem"]
        gold_solution = example["solution"]
        gold_answer = extract_boxed_answer(gold_solution)
        level = example.get("level", "unknown")
        subarea = example.get("subarea", "unknown")
        
        # Format prompt (same for both models)
        prompt = format_prompt(problem, tokenizer1)
        
        # Generate using product distribution
        try:
            generated = product_distribution_inference(
                model1=model1,
                model2=model2,
                prompt1=prompt,
                prompt2=prompt,
                tokenizer1=tokenizer1,
                tokenizer2=tokenizer2,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                lam=lam,
                device=device,
            )
        except Exception as e:
            print(f"\nError on problem {i}: {e}")
            generated = ""

        print(f"Prompt 1: {prompt}")
        print(f"Generated: {generated}")
        
        # Extract predicted answer
        pred_answer = extract_boxed_answer(generated)
        if pred_answer is None:
            pred_answer = extract_answer_tag(generated)
        
        # Check correctness
        is_correct = answers_match(pred_answer or "", gold_answer or "")
        if is_correct:
            correct += 1
        
        result = {
            "index": i,
            "problem": problem,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "generated": generated,
            "correct": is_correct,
            "level": level,
            "subarea": subarea,
        }
        results.append(result)
        
        # Print progress
        if (i + 1) % 10 == 0:
            acc = correct / (i + 1) * 100
            print(f"\nProgress: {i+1}/{len(dataset)}, Accuracy: {acc:.1f}%")
    
    # Final statistics
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model 1: {model1_name}")
    print(f"Model 2: {model2_name}")
    print(f"Lambda: {lam}")
    print(f"Temperature: {temperature}")
    print(f"Total problems: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Breakdown by level
    print("\nAccuracy by level:")
    levels = set(r["level"] for r in results)
    for level in sorted(levels):
        level_results = [r for r in results if r["level"] == level]
        level_correct = sum(1 for r in level_results if r["correct"])
        level_acc = level_correct / len(level_results) * 100 if level_results else 0
        print(f"  {level}: {level_correct}/{len(level_results)} = {level_acc:.1f}%")
    
    # Breakdown by subarea
    print("\nAccuracy by subarea:")
    subareas = set(r["subarea"] for r in results)
    for subarea in sorted(subareas):
        subarea_results = [r for r in results if r["subarea"] == subarea]
        subarea_correct = sum(1 for r in subarea_results if r["correct"])
        subarea_acc = subarea_correct / len(subarea_results) * 100 if subarea_results else 0
        print(f"  {subarea}: {subarea_correct}/{len(subarea_results)} = {subarea_acc:.1f}%")
    
    # Save results
    output_file = "math_eval_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "model1": model1_name,
                "model2": model2_name,
                "lam": lam,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "split": split,
            },
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results, accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate models on MATH using product distribution")
    parser.add_argument("--model1", type=str, default="Qwen/Qwen3-1.7B",
                        help="First model name")
    parser.add_argument("--model2", type=str, default="nvidia/AceMath-1.5B-Instruct",
                        help="Second model name")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of samples to evaluate (default: 20)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--lam", type=float, default=0.5,
                        help="Interpolation weight (0-1)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="Dataset split to use")
    
    args = parser.parse_args()

    device = None
    # Auto-detect device with MPS support
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    evaluate_math(
        model1_name=args.model1,
        model2_name=args.model2,
        num_samples=args.num_samples,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        lam=args.lam,
        split=args.split,
        device=device,
    )

