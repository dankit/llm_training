#!/bin/bash
# =============================================================================
# Launch Script for 8xA100 40GB Training
# =============================================================================
set -e

NGPUS=${NGPUS:-8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "        GPT Training Launch Script"
echo "=============================================="

# -----------------------------------------------------------------------------
# Step 1: Install Python dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# -----------------------------------------------------------------------------
# Step 2: Download pre-tokenized dataset from HuggingFace Hub
# -----------------------------------------------------------------------------
echo ""
echo "[2/3] Downloading pre-tokenized dataset from HuggingFace Hub..."
echo "      (This will cache to ~/.cache/huggingface/datasets/)"
python data_pipeline.py download

# -----------------------------------------------------------------------------
# Step 3: Launch distributed training
# -----------------------------------------------------------------------------
echo ""
echo "[3/3] Launching training on $NGPUS GPUs..."
echo "=============================================="

# PyTorch CUDA memory management (optional but helps on long runs)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1     # Better error recovery
export NCCL_DEBUG=WARN                       # Log warnings only (change to INFO for debugging)

torchrun --standalone --nproc_per_node=$NGPUS training.py "$@"
