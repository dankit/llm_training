import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent HuggingFace tokenizer deadlocks with multiprocessing

import json
import math
import time
from datetime import datetime
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

from transformer import Transformer
from data_pipeline import TrainDataLoader, DATA_DIR
from eval_hellaswag import download_hellaswag, load_hellaswag, evaluate_hellaswag


# Set spawn method for DataLoader workers (avoids CUDA fork issues on Linux)
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Model - ~416M params for 10B token training set, using more modern techniques: SwiGLU, RMSNorm, RoPE
# attention insight: GQA is really only needed at 7B+ to optimize for memory, would use something like head_dim=128 (2:1 ratio of heads to kv heads) - at the cost of some quality
# but at <1B, we want to optimize for quality, memory is not as much of an issue.
# below, can configure for either multi-head attention (MHA) or grouped query attention (GQA)
@dataclass
class Config:
    dim: int = 1024               
    n_layers: int = 24
    n_heads: int = 16             # head_dim=64 → more attention diversity (better at small scale), compared to head_dim=128
    n_kv_heads: int = 16          # MHA - GQA saves <1% memory at 420M, hurts quality
    vocab_size: int = 50257
    ffn_hidden_mult: float = 3.5  # 3584/1024 - SwiGLU
    
    # Training (tuned for 8×A100 40GB, 10B tokens). 
    # Was going OOM at batch size 32, reduced to 24 (oom again), then 16. Activations were taking up too much memory. Enabled gradient checkpointing to reduce memory usage.
    # increased batch size back to 40 to fill up vram. 40 worked ~.5% better than 32.
    total_tokens: int = 10_000_000_000  # Target: 10B tokens
    batch_size: int = 40         
    seq_len: int = 1024           # Context length  
    grad_accum_steps: int = 2     # Effective batch = 40*1024*8*2 = 655,360 tokens/step
    expected_world_size: int = 8  # GPUs (for max_steps calc, actual determined at runtime)
    max_steps: int = None         # Computed in __post_init__
    
    def __post_init__(self):
        if self.max_steps is None:
            tokens_per_step = self.batch_size * self.seq_len * self.expected_world_size * self.grad_accum_steps
            self.max_steps = self.total_tokens // tokens_per_step
    max_lr: float = 3e-4          # Scaled for ~420M model
    min_lr: float = 3e-5          # 10% of max_lr (standard)
    warmup_ratio: float = 0.02    # ~382 warmup steps (2% is modern standard)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    
    # Data - run `python data_pipeline.py download` to prepare
    data_dir: str = str(DATA_DIR)  # ./data/ready (pre-chunked, loads instantly)
    num_workers: int = None        # DataLoader workers (None = auto-detect for your hardware)
    prefetch_factor: int = 4       # Batches to prefetch per worker
    
    # Logging & Checkpointing
    log_every: int = 10           # Save metrics to JSON every N steps
    checkpoint_every: int = 2000   # Save model every ~1B tokens (5 checkpoints total)
    resume_from: str = None
    
    # Evaluation
    hellaswag_eval: bool = True   # Run HellaSwag at checkpoints
    hellaswag_samples: int = 200  # Subset for speed (full=10042, use None for all)
    
    # Hardware & Memory
    gpu_type: str = "a100"        # Options: a100, h100, 4090, 3090, a6000
    gradient_checkpointing: bool = True 


def configure_optimizers(model, lr, weight_decay=0.1):
    """Separate params into decay/no-decay groups."""
    base = model.module if hasattr(model, 'module') else model
    decay = [p for p in base.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in base.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=torch.cuda.is_available())


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Warmup + cosine decay schedule."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if max_steps <= warmup_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def estimate_mfu(n_params, batch_size, seq_len, dt, world_size=1, gpu_type="a100"):
    """
    Estimate Model FLOPs Utilization — what fraction of GPU peak you're achieving.
    
    FLOPs per forward pass ≈ 2 * N * T (matmuls) where N=params, T=tokens
    Backward ≈ 2x forward, so total ≈ 6 * N * T per step
    
    Good MFU: 40-60% (A100), >50% means your pipeline is efficient.
    Low MFU (<30%) suggests bottlenecks: data loading, CPU overhead, small batches.
    """
    # Peak FLOPS for different GPUs (BF16/FP16 tensor core)
    peak_flops = {
        "a100": 312e12,      # A100 40/80GB
        "a100_sxm": 312e12,
        "h100": 989e12,      # H100 SXM
        "h100_pcie": 756e12,
        "4090": 330e12,      # RTX 4090
        "3090": 142e12,      # RTX 3090
        "a6000": 155e12,     # RTX A6000
    }
    
    tokens_per_step = batch_size * seq_len * world_size
    flops_per_step = 6 * n_params * tokens_per_step  # 6N approximation
    flops_achieved = flops_per_step / dt
    
    gpu_peak = peak_flops.get(gpu_type.lower(), 312e12) * world_size
    return flops_achieved / gpu_peak


def sample_tokens(logits, temperature=0.8):
    """Sample from logits with temperature."""
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens, device, temperature=0.8):
    """Generate tokens with temperature sampling (memory-optimized)."""
    model.eval()
    B, T = prompt_tokens.shape
    
    # Pre-allocate output buffer to avoid repeated tensor concatenation
    max_len = T + max_new_tokens
    tokens = torch.zeros((B, max_len), dtype=torch.long, device=device)
    tokens[:, :T] = prompt_tokens.to(device)
    
    for i in range(max_new_tokens):
        # Only pass the generated portion so far (avoids processing padding)
        current_len = T + i
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = model(tokens[:, :current_len])
        next_token = sample_tokens(logits[:, -1, :], temperature)
        tokens[:, current_len] = next_token
    
    model.train()
    return tokens


if __name__ == "__main__":
    config = Config()
    
    # DDP setup
    ddp = 'RANK' in os.environ
    if ddp:
        dist.init_process_group(backend='nccl')
        rank, local_rank, world_size = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        master = rank == 0
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        master = True

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Model
    model = Transformer(config).to(device)
    if ddp:
        model = DDP(
            model, 
            device_ids=[local_rank],
            gradient_as_bucket_view=True,  # Reduce memory copy overhead
        )
    model = torch.compile(model, dynamic=False)
    raw_model = model.module if ddp else model
    n_params = sum(p.numel() for p in raw_model.parameters())
    
    if master:
        print(f"Model: {n_params/1e6:.1f}M params | Device: {device} | DDP: {ddp}")

    # Optimizer & Dataloaders
    optimizer = configure_optimizers(model, lr=config.max_lr, weight_decay=config.weight_decay)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    train_loader = TrainDataLoader(
        data_dir=config.data_dir,
        batch_size=config.batch_size, 
        seq_len=config.seq_len,
        rank=rank, 
        world_size=world_size, 
        seed=config.seed + rank,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
    )
    
    if master:
        cpu_count = os.cpu_count() or 8
        print(f"DataLoader: {train_loader.num_workers} workers/GPU x {world_size} GPUs = {train_loader.num_workers * world_size} total")
        print(f"prefetch_factor={config.prefetch_factor} (CPUs: {cpu_count})")
        print(f"Max steps: {config.max_steps}")

    # Training state
    warmup_steps = int(config.warmup_ratio * config.max_steps)
    tokens_per_step = config.batch_size * config.seq_len * world_size * config.grad_accum_steps
    total_tokens, ema_loss, start_step = 0, None, 0
    metrics_log = []
    log_file = f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("checkpoints", exist_ok=True)

    # Resume from checkpoint
    if config.resume_from and os.path.exists(config.resume_from):
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step'] + 1
        total_tokens = ckpt['total_tokens']
        ema_loss = ckpt.get('ema_loss')
        if 'dataloader_batch_idx' in ckpt:
            train_loader.skip_to(ckpt['dataloader_batch_idx'], epoch=ckpt.get('dataloader_epoch', 0))
        if master:
            print(f"Resumed from step {ckpt['step']}, loss: {ckpt['loss']:.4f}")

    # Training loop - use CUDA events for non-blocking timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for step in range(start_step, config.max_steps):
        start_event.record()
        optimizer.zero_grad()
        
        # Gradient accumulation
        loss_accum = 0.0
        for micro_step in range(config.grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Only sync gradients on the last micro-step (DDP optimization)
            is_last_micro = (micro_step == config.grad_accum_steps - 1)
            sync_context = nullcontext() if not ddp or is_last_micro else model.no_sync()
            
            with sync_context:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / config.grad_accum_steps
                #otherwise ddp would sync gradients after every micro-step
                loss.backward()
            loss_accum += loss.item()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
        
        # LR schedule
        lr = get_lr(step, warmup_steps, config.max_steps, config.max_lr, config.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        optimizer.step()
        
        if ddp:
            loss_tensor = torch.tensor(loss_accum, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            loss_val = loss_tensor.item()
        else:
            loss_val = loss_accum
        
        end_event.record()
        end_event.synchronize()  # Wait only when we need timing (non-blocking until here)
        step_time = start_event.elapsed_time(end_event) / 1000  # ms to seconds
        total_tokens += tokens_per_step
        ema_loss = loss_val if ema_loss is None else 0.99 * ema_loss + 0.01 * loss_val
        mfu = estimate_mfu(n_params, config.batch_size * config.grad_accum_steps, 
                           config.seq_len, step_time, world_size, config.gpu_type)
        
        if master:
            # Memory stats:
            # - allocated: actual tensor memory in use right now
            # - reserved: PyTorch's cached memory (what nvidia-smi shows, determines OOM)
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            
            print(f"Step {step:4d} | Loss: {loss_val:.4f} | EMA: {ema_loss:.4f} | "
                  f"PPL: {math.exp(loss_val):.1f} | Grad: {grad_norm:.2f} | "
                  f"MFU: {mfu*100:.1f}% | Tok/s: {tokens_per_step/step_time:.0f} | LR: {lr:.2e} | "
                  f"Mem: {mem_reserved:.1f}GB")
            
            metrics_log.append({
                "step": step, "loss": loss_val, "ema_loss": ema_loss,
                "perplexity": math.exp(loss_val), "grad_norm": grad_norm,
                "tokens_per_sec": tokens_per_step / step_time, "lr": lr,
                "total_tokens": total_tokens, "step_time": step_time,
                "mfu": mfu, 
                "memory_allocated_gb": mem_allocated,  # Tensor memory in use
                "memory_reserved_gb": mem_reserved,    # PyTorch cache (nvidia-smi, OOM threshold)
            })
            
            # Save metrics frequently for live plotting
            if step % config.log_every == 0:
                with open(log_file, 'w') as f:
                    json.dump(metrics_log, f, indent=2)
        
        # Checkpoint
        if (step > 0 and step % config.checkpoint_every == 0) or step == config.max_steps - 1:
            if ddp:
                dist.barrier()
            
            if master:
                # Generation sample (use raw_model to avoid DDP sync issues)
                prompt = torch.tensor(tokenizer.encode("Hello, ")).unsqueeze(0)
                generated = generate(raw_model, prompt, max_new_tokens=20, device=device)
                print(f"Generated: {tokenizer.decode(generated[0].tolist())}")
                
                # HellaSwag evaluation
                hellaswag_acc = None
                if config.hellaswag_eval:
                    hellaswag_path = download_hellaswag()
                    hellaswag_examples = load_hellaswag(hellaswag_path, config.hellaswag_samples)
                    hellaswag_results = evaluate_hellaswag(
                        raw_model, tokenizer, hellaswag_examples, device, 
                        max_length=config.seq_len, store_detailed_results=False
                    )
                    hellaswag_acc = hellaswag_results['accuracy']
                    print(f"HellaSwag: {hellaswag_acc*100:.2f}% ({hellaswag_results['correct']}/{hellaswag_results['total']})")
                    model.train()  # Restore training mode after evaluation
                    
                    # Add to metrics log for tracking over training
                    metrics_log.append({
                        "step": step, "hellaswag_acc": hellaswag_acc, "is_checkpoint": True,
                    })
                
                # Save
                torch.save({
                    'step': step, 'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val, 'total_tokens': total_tokens,
                    'ema_loss': ema_loss, 'dataloader_batch_idx': train_loader.batch_idx,
                    'dataloader_epoch': train_loader.epoch, 'config': config.__dict__,
                    'hellaswag_acc': hellaswag_acc,
                }, f"checkpoints/checkpoint_step_{step}.pt")
                
                with open(log_file, 'w') as f:
                    json.dump(metrics_log, f, indent=2)
                print(f"Checkpoint saved")
            
            if ddp:
                dist.barrier()

    if master:
        print(f"\nTraining complete. Metrics: {log_file}")
    
    if ddp:
        dist.destroy_process_group()
