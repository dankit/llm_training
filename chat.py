"""Interactive chat with inference benchmarking."""
import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import tiktoken

from transformer import Transformer


@dataclass
class Config:
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 16
    vocab_size: int = 50257
    ffn_hidden_mult: float = 3.5
    seq_len: int = 1024
    gradient_checkpointing: bool = False


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def generate(model, tokens, max_new_tokens, temperature=0.8, use_kv_cache=True, debug=False):
    """Generate tokens with optional KV cache. Returns (output_tokens, tokens_generated, time_taken)."""
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    B, prompt_len = tokens.shape
    use_amp = torch.cuda.is_available()
    
    cuda_sync()
    start = time.perf_counter()
    
    if use_kv_cache:
        kv_caches = model.allocate_kv_cache(B, prompt_len + max_new_tokens, device)
        
        # Prefill: process entire prompt
        with torch.autocast('cuda', torch.bfloat16, enabled=use_amp):
            logits, kv_caches = model(tokens, start_pos=0, kv_caches=kv_caches)
        
        if debug:
            print(f"[DEBUG] Prefill: input {tokens.shape}, cache {kv_caches[0][0].shape}")
        
        # Generate one token at a time
        for i in range(max_new_tokens):
            next_token = torch.multinomial(F.softmax(logits[:, -1] / temperature, dim=-1), 1)
            tokens = torch.cat([tokens, next_token], dim=1)
            with torch.autocast('cuda', torch.bfloat16, enabled=use_amp):
                logits, kv_caches = model(next_token, start_pos=prompt_len + i, kv_caches=kv_caches)
    else:
        # No cache: recompute full sequence each step
        for i in range(max_new_tokens):
            with torch.autocast('cuda', torch.bfloat16, enabled=use_amp):
                logits, _ = model(tokens)
            if debug and i == 0:
                print(f"[DEBUG] No cache: full sequence {tokens.shape}")
            next_token = torch.multinomial(F.softmax(logits[:, -1] / temperature, dim=-1), 1)
            tokens = torch.cat([tokens, next_token], dim=1)
    
    cuda_sync()
    return tokens, max_new_tokens, time.perf_counter() - start


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    cfg = ckpt.get('config', {})
    config = Config(**{k: cfg[k] for k in Config.__dataclass_fields__ if k in cfg})
    
    model = Transformer(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Loaded step {ckpt.get('step', '?')} | {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    model = load_model(args.checkpoint, args.device)
    enc = tiktoken.get_encoding("gpt2")
    
    print(f"\nReady (tokens={args.max_tokens}, temp={args.temperature}, device={args.device})")
    print("Commands: /bench [prompt], /compare [prompt], /quit\n")
    
    default_prompt = "The meaning of life is"
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input or user_input == '/quit':
            if user_input == '/quit':
                break
            continue
        
        # /compare: benchmark with vs without KV cache
        if user_input.startswith('/compare'):
            prompt = user_input[8:].strip() or default_prompt
            tokens = torch.tensor([enc.encode(prompt)])
            print(f"Comparing ({args.max_tokens} tokens)...\n")
            
            results = {}
            for name, cache in [("WITH cache", True), ("WITHOUT cache", False)]:
                print(f"{name}:")
                _, n, t = generate(model, tokens, args.max_tokens, args.temperature, cache, debug=True)
                results[name] = n / t
                print(f"  {results[name]:.1f} tok/s ({t:.2f}s)\n")
            
            print(f"Speedup: {results['WITH cache'] / results['WITHOUT cache']:.2f}x")
            continue
        
        # /bench: run 3 times and average
        if user_input.startswith('/bench'):
            prompt = user_input[6:].strip() or default_prompt
            tokens = torch.tensor([enc.encode(prompt)])
            print("Benchmarking 3 runs...")
            
            runs = []
            for i in range(3):
                _, n, t = generate(model, tokens, args.max_tokens, args.temperature, debug=(i == 0))
                runs.append((n / t, t))
                print(f"  Run {i+1}: {runs[-1][0]:.1f} tok/s ({t:.2f}s)")
            
            print(f"  Avg: {sum(r[0] for r in runs)/3:.1f} tok/s | {sum(r[1] for r in runs)/3:.2f}s")
            continue
        
        # Normal chat
        tokens = torch.tensor([enc.encode(user_input)])
        output, n, t = generate(model, tokens, args.max_tokens, args.temperature)
        print(f"Model: {enc.decode(output[0].tolist())}")
        print(f"  [{n} tokens in {t:.2f}s = {n/t:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
