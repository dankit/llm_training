import os
import multiprocessing
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Prevent HuggingFace tokenizer deadlocks with multiprocessing
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# =============================================================================
# CONFIGURATION
# =============================================================================


DATA_DIR = Path("./data/ready")

# HuggingFace repos
HUB_PRETOKENIZED = "dhlak/finewebedu-10b-gpt2-tokenized"  # Raw tokenized (needs chunking)
RAW_DATASET = "HuggingFaceFW/fineweb-edu"
RAW_SUBSET = "sample-10BT"

# Training sequence length (chunk size = seq_len + 1 for next-token prediction)
DEFAULT_SEQ_LEN = 1024


# =============================================================================
# DATASET PREPARATION (run once before training)
# =============================================================================

def download_pretokenized(output_dir: Path = DATA_DIR, seq_len: int = DEFAULT_SEQ_LEN):
    """
    This is the fastest way to get started - downloads ~20GB, chunks locally.
    Result: A training-ready dataset that loads in seconds.
    """
    from datasets import load_dataset
    import tiktoken
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading pre-tokenized dataset from {HUB_PRETOKENIZED}.. This will download ~20GB to ~/.cache/huggingface/datasets/")
    
    # Load from hub (this caches to ~/.cache/huggingface/)
    num_proc = min(multiprocessing.cpu_count(), 64)
    dataset = load_dataset(HUB_PRETOKENIZED, split="train", num_proc=num_proc)
    print(f"Loaded {len(dataset):,} examples")
    
    # Chunk into training sequences
    _chunk_and_save(dataset, output_dir, seq_len, token_col="input_ids")
    
    print(f"Dataset ready at {output_dir}")


def prepare_from_scratch(output_dir: Path = DATA_DIR, seq_len: int = DEFAULT_SEQ_LEN):
    """
    Download raw FineWeb-Edu, tokenize with GPT-2 tokenizer, and chunk for training.
    This is slower but doesn't require trusting pre-tokenized data.
    Can take an hour or longer depending on cpu count.
    """
    from datasets import load_dataset
    import tiktoken
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_proc = multiprocessing.cpu_count()
    tokenizer = tiktoken.get_encoding("gpt2")
    eot = tokenizer.eot_token
    
    print(f"Loading {RAW_DATASET}/{RAW_SUBSET}...")
    dataset = load_dataset(RAW_DATASET, RAW_SUBSET, split="train", num_proc=num_proc)
    print(f"Loaded {len(dataset):,} documents")
    
    # Tokenize
    def tokenize_batch(examples):
        tokens = []
        for text in examples["text"]:
            tokens.extend(tokenizer.encode(text, disallowed_special=()) + [eot])
        return {"tokens": tokens}
    
    print(f"Tokenizing with {num_proc} CPUs...")
    tokenized = dataset.map(
        tokenize_batch, 
        batched=True, 
        batch_size=1000,
        num_proc=num_proc, 
        remove_columns=dataset.column_names,
    )
    
    # Chunk into training sequences
    _chunk_and_save(tokenized, output_dir, seq_len, token_col="input_ids")
    print(f"Dataset ready at {output_dir}")


def _chunk_and_save(dataset, output_dir: Path, seq_len: int, token_col: str = "tokens"):
    """
    Chunk a tokenized dataset into fixed-length sequences and save.
    Each chunk is seq_len+1 tokens: first seq_len are input, last seq_len are target.
    """
    num_proc = min(multiprocessing.cpu_count(), 64)
    chunk_size = seq_len + 1  # +1 for next-token prediction
    
    # batch_size must be a multiple of chunk_size to avoid dropping tokens
    # (each batch's remainder is lost, so we ensure remainder = 0)
    batch_size = chunk_size * 2000  # ~2M tokens per batch, zero token loss
    
    def chunk_tokens(examples):
        """Flatten all tokens and split into fixed-size chunks."""
        ids = examples[token_col]
        # Flatten if nested (each example may be a list of token lists)
        if isinstance(ids[0], list):
            ids = [t for seq in ids for t in seq]
        # Split into chunks, dropping the last incomplete chunk
        chunks = [ids[i:i + chunk_size] for i in range(0, len(ids) - chunk_size + 1, chunk_size)]
        return {"input_ids": chunks}
    
    print(f"Chunking with {num_proc} CPUs...")
    chunked = dataset.map(
        chunk_tokens, 
        batched=True, 
        batch_size=batch_size,
        remove_columns=dataset.column_names, 
        num_proc=num_proc
    )
    
    # Save to disk in Arrow format (memory-maps for instant loading)
    # Fewer shards = faster loading?
    num_shards = min(16, num_proc)
    chunked.save_to_disk(str(output_dir), num_shards=num_shards)
    
    # Count total tokens
    total_chunks = len(chunked)
    total_tokens = total_chunks * chunk_size
    print(f"Saved {total_chunks:,} chunks ({total_tokens/1e9:.2f}B tokens) to {output_dir}")


# =============================================================================
# TRAINING DATALOADER (used during training)
# =============================================================================

def _collate_fn(batch):
    """Split chunks into (input, target) pairs for next-token prediction."""
    chunks = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
    return chunks[:, :-1], chunks[:, 1:]  # input, target


class TrainDataLoader:
    """
    DataLoader for distributed training with checkpoint resume support.
    
    Features:
    - Automatic DDP sharding (each GPU sees unique data)
    - Deterministic shuffling with epoch-based seeds
    - Skip-to support for checkpoint resumption
    - Optimized for high-CPU systems with auto-scaled workers
    
    Args:
        data_dir: Path to the prepared dataset (default: ./data/ready)
        batch_size: Per-GPU batch size
        seq_len: Sequence length (must match what dataset was prepared with)
        rank: DDP rank (0 for single GPU)
        world_size: Number of GPUs
        seed: Random seed for shuffling
        num_workers: DataLoader workers (None = auto-detect)
        prefetch_factor: Batches to prefetch per worker
    """
    
    def __init__(
        self, 
        data_dir: str = None, 
        batch_size: int = 8, 
        seq_len: int = DEFAULT_SEQ_LEN, 
        rank: int = 0, 
        world_size: int = 1, 
        seed: int = 42,
        num_workers: int = None,
        prefetch_factor: int = 4,
    ):
        from datasets import load_from_disk
        
        data_dir = Path(data_dir) if data_dir else DATA_DIR
        
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        
        # Auto-scale workers: (CPUs - reserved) / GPUs, capped at 16
        if num_workers is None:
            cpu_count = multiprocessing.cpu_count()
            reserved = world_size * 2  # Reserve 2 CPUs per GPU for main process
            num_workers = min(16, max(4, (cpu_count - reserved) // world_size))
        
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        # Load dataset (Arrow format is memory-mapped)
        if rank == 0:
            print(f"Loading dataset from {data_dir}...")
        
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_dir}.\n"
                f"Run one of:\n"
                f"  python data_pipeline.py download  # Download pre-tokenized (fastest)\n"
                f"  python data_pipeline.py prepare   # Tokenize from scratch"
            )
        
        self.dataset = load_from_disk(str(data_dir))
        
        # Verify chunk size matches expected seq_len
        sample_len = len(self.dataset[0]["input_ids"])
        expected_len = seq_len + 1
        if sample_len != expected_len:
            raise ValueError(
                f"Dataset chunk size ({sample_len}) doesn't match seq_len+1 ({expected_len}).\n"
                f"Re-prepare the dataset with seq_len={sample_len - 1} or change your training config."
            )
        
        # Shard for DDP (each GPU gets unique slice)
        if world_size > 1:
            self.dataset = self.dataset.shard(num_shards=world_size, index=rank)
        
        if rank == 0:
            print(f"Loaded {len(self.dataset):,} chunks ({len(self.dataset) * seq_len / 1e9:.2f}B tokens per GPU)")
        
        # Create initial dataloader
        self._create_loader()
    
    def _create_loader(self):
        """Create DataLoader with current epoch's seed for deterministic shuffling."""
        # Explicitly delete old iterator to release worker resources
        if hasattr(self, 'iterator'):
            del self.iterator
        if hasattr(self, 'loader'):
            del self.loader
        
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.loader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            collate_fn=_collate_fn,
            pin_memory=True, 
            drop_last=True, 
            generator=generator,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
        self.iterator = iter(self.loader)
        self.batch_idx = 0
        self.total_batches = len(self.loader)
    
    def next_batch(self):
        """Get next batch, automatically handling epoch boundaries."""
        try:
            batch = next(self.iterator)
            self.batch_idx += 1
            return batch
        except StopIteration:
            self.epoch += 1
            self._create_loader()
            batch = next(self.iterator)
            self.batch_idx += 1
            return batch
    
    def skip_to(self, batch_idx: int, epoch: int = 0):
        """Skip to a specific position for checkpoint resumption."""
        self.epoch = epoch
        self._create_loader()
        target = batch_idx % self.total_batches
        for _ in range(target):
            try:
                next(self.iterator)
                self.batch_idx += 1
            except StopIteration:
                break


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import tiktoken
    
    parser = argparse.ArgumentParser(
        description="Prepare FineWeb-Edu dataset for GPT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_pipeline.py download   # Download pre-tokenized from HuggingFace (fastest)
  python data_pipeline.py prepare    # Tokenize from scratch
  python data_pipeline.py test       # Test the dataloader
        """
    )
    parser.add_argument("command", nargs="?", choices=["download", "prepare", "test"],
                        help="Command to run")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"Output directory for prepared dataset (default: {DATA_DIR})")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help=f"Sequence length for chunking (default: {DEFAULT_SEQ_LEN})")
    args = parser.parse_args()
    
    if args.command == "download":
        download_pretokenized(args.data_dir, args.seq_len)
        
    elif args.command == "prepare":
        prepare_from_scratch(args.data_dir, args.seq_len)
        
    elif args.command == "test":
        print("Testing dataloader...")
        loader = TrainDataLoader(str(args.data_dir), batch_size=4, seq_len=args.seq_len)
        x, y = loader.next_batch()
        print(f"Batch shape: x={x.shape}, y={y.shape}")
        
        tokenizer = tiktoken.get_encoding("gpt2")
        print(f"Sample text: {tokenizer.decode(x[0, :50].tolist())}...")
        
    else:
        parser.print_help()
