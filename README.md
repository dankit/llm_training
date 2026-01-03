# LLM Training

416M parameter transformer trained on FineWeb-Edu 10B tokens. The model was trained on 8xA100 gpus and after one pass over 10B data,
it is starting to show some aspects of learned language. It is not trained enough, or presumably big enough, to complete any useful tasks.
Saved checkpoint can be found: https://huggingface.co/dhlak/416m-gpt

The checkpoint is close to chinchilla optimal (currently at ~24x), leaving some room for more training to see if any usefulness can be obtained. This model size was chosen because I've had a lot of utility with xlm-roberta, a 560m parameter encoder only model. This was to see how the decoder-only aspect could compare. While this idea fell short due to compute budget, there was a lot of valuable learning in setting up the transformer -> data collection/processing -> training loop -> distributed data parallel -> memory optimizations etc.

## Setup

**One-liner (installs deps, downloads data, trains):**
```bash
./launch_training.sh
```

otherwise, step by step:

```bash
pip install -r requirements.txt
```

## 1. Prepare Data

Download pre-tokenized dataset (~20GB):
```bash
python data_pipeline.py download
```

Or tokenize from scratch (slower):
```bash
python data_pipeline.py prepare
```

## 2. Train

**Single GPU:**
```bash
python training.py
```

**Multi-GPU (8x):**
```bash
torchrun --standalone --nproc_per_node=8 training.py
```

Checkpoints save to `checkpoints/` every 2000 steps.

## 3. Evaluate

```bash
python eval_hellaswag.py --checkpoint checkpoints/checkpoint_step_2000.pt
```

## 4. Chat

```bash
python chat.py checkpoints/checkpoint_step_2000.pt
```
Commands: `/bench`, `/compare`, `/quit`


You can find my training metrics in "final_metrics.json". 
Plotted metrics:
![Final metrics](final_metrics.png)

Initially I had a few optimization issues and bugs with DDP and my training loop, which is why there was unstable token throughput and MFU early on. I was able to fix a few bugs which can be seen from step 4000 and onwards. After step 4000, the main culprit was checkpointing, which often had reduced token throughput for 1-2 steps after a checkpoint.

Hellaswag evaluation steadily increased overtime, however there may be some scrutiny in the implementation or dataset having leakage.


Credits go to Andrej Karpathy for all his educational content.