"""
HellaSwag Evaluation for Language Models

HellaSwag tests commonsense reasoning: given a context, pick the correct ending from 4 options.
We compute log-likelihood of each ending conditioned on the context, pick the highest.

Usage:
    python eval_hellaswag.py --checkpoint checkpoints/checkpoint_step_500.pt
    python eval_hellaswag.py --checkpoint checkpoints/checkpoint_step_500.pt --num_examples 100
"""

import os
import json
import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm

from transformer import Transformer


def download_hellaswag():
    """Download HellaSwag validation set if not present."""
    import urllib.request
    
    data_dir = "./data/hellaswag"
    os.makedirs(data_dir, exist_ok=True)
    
    val_path = os.path.join(data_dir, "hellaswag_val.jsonl")
    if not os.path.exists(val_path):
        print("Downloading HellaSwag validation set...")
        url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
        urllib.request.urlretrieve(url, val_path)
        print(f"Saved to {val_path}")
    
    return val_path


def load_hellaswag(path: str, num_examples: int = None):
    """Load HellaSwag examples from JSONL file."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            examples.append({
                'context': example['ctx'],
                'endings': example['endings'],
                'label': int(example['label']),
                'activity_label': example.get('activity_label', ''),
            })
    
    if num_examples:
        examples = examples[:num_examples]
    
    return examples


def render_example(context: str, ending: str) -> str:
    """Combine context and ending into a single string for scoring."""
    # HellaSwag contexts sometimes end with incomplete sentences
    # The ending continues the sentence
    return context + " " + ending


@torch.no_grad()
def compute_completion_loss(
    model, 
    tokenizer, 
    context: str, 
    ending: str, 
    device: str,
    max_length: int = 1024,
) -> float:
    """
    Compute average negative log-likelihood of the ending tokens,
    conditioned on the context.
    
    Returns: average NLL per token (lower is better for this ending)
    """
    # Tokenize context and full sequence
    context_tokens = tokenizer.encode(context)
    full_text = render_example(context, ending)
    full_tokens = tokenizer.encode(full_text)
    
    # Truncate if necessary
    if len(full_tokens) > max_length:
        full_tokens = full_tokens[:max_length]
    
    # Get where the ending starts
    ending_start = len(context_tokens)
    
    # Need at least 1 ending token to evaluate
    if ending_start >= len(full_tokens):
        return float('inf')
    
    # Prepare input
    tokens = torch.tensor(full_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Forward pass
    with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.bfloat16):
        logits, _ = model(tokens)  # (1, seq_len, vocab_size)
    
    # Compute loss only on ending tokens
    # For position i, logits[i] predicts token[i+1]
    # So to evaluate ending tokens [ending_start:], we need logits[ending_start-1:-1]
    
    # Shift for next-token prediction
    shift_logits = logits[0, ending_start-1:-1, :]  # Logits that predict ending tokens
    shift_labels = tokens[0, ending_start:]          # The ending tokens themselves
    
    # Per-token cross entropy
    loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')
    
    return loss.item()


@torch.no_grad()
def evaluate_hellaswag(
    model,
    tokenizer,
    examples: list,
    device: str,
    max_length: int = 1024,
    store_detailed_results: bool = False,
) -> dict:
    """
    Evaluate model on HellaSwag examples.
    
    For each example:
    1. Compute loss for each of 4 possible endings
    2. Pick the ending with lowest loss (highest likelihood)
    3. Check if it matches the ground truth label
    
    Args:
        store_detailed_results: If False, only returns accuracy (memory-efficient).
                               If True, stores per-example results.
    """
    model.eval()
    
    correct = 0
    total = 0
    results = [] if store_detailed_results else None
    
    for example in tqdm(examples, desc="Evaluating HellaSwag"):
        context = example['context']
        endings = example['endings']
        label = example['label']
        
        # Score each ending
        losses = []
        for ending in endings:
            loss = compute_completion_loss(
                model, tokenizer, context, ending, device, max_length
            )
            losses.append(loss)
        
        # Predict ending with lowest loss
        prediction = min(range(len(losses)), key=lambda i: losses[i])
        is_correct = prediction == label
        
        if is_correct:
            correct += 1
        total += 1
        
        if store_detailed_results:
            results.append({
                'context': context[:100] + '...' if len(context) > 100 else context,
                'prediction': prediction,
                'label': label,
                'correct': is_correct,
                'losses': losses,
            })
    
    # Clear CUDA cache after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results,
    }


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    """Load model from training checkpoint."""
    from training import Config  # Lazy import to avoid circular dependency
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = Config(**{k: v for k, v in config_dict.items() if hasattr(Config, k)})
    else:
        config = Config()
    
    # Build model
    model = Transformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
    print(f"Training loss was: {checkpoint.get('loss', 'unknown'):.4f}")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on HellaSwag")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples (default: all ~10k)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    parser.add_argument("--output", type=str, default=None, help="Save detailed results to JSON")
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # Compile for speed (optional, comment out if issues)
    # model = torch.compile(model)
    
    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Download and load HellaSwag
    hellaswag_path = download_hellaswag()
    examples = load_hellaswag(hellaswag_path, args.num_examples)
    print(f"Loaded {len(examples)} HellaSwag examples")
    
    # Evaluate (with detailed results for standalone runs)
    results = evaluate_hellaswag(model, tokenizer, examples, device, max_length=config.seq_len, store_detailed_results=True)
    
    # Print results
    print("\n" + "="*50)
    print(f"HellaSwag Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct: {results['correct']} / {results['total']}")
    print("="*50)
    
    # Reference scores (rough):
    print("\nReference scores (approximate):")
    print("  Random baseline: 25.0%")
    print("  GPT-2 (124M):    ~29%")
    print("  GPT-2 (1.5B):    ~45%")
    print("  GPT-3 (175B):    ~78%")
    
    # Save detailed results
    if args.output:
        # Don't save full results list to keep file size reasonable
        save_data = {
            'accuracy': results['accuracy'],
            'correct': results['correct'],
            'total': results['total'],
            'checkpoint': args.checkpoint,
            'sample_results': results['results'][:20],  # First 20 for inspection
        }
        with open(args.output, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()

