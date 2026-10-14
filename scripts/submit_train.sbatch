#!/bin/bash
#SBATCH --job-name=orthrus-qwen3_5-4b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.out

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-.cache/torchinductor}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

STEPS=${STEPS:-0}   # 0 = the full --epochs (paper: 2)
OUT_DIR="${OUT_DIR:-checkpoints}"
mkdir -p "$OUT_DIR"

echo "node=$(hostname) steps=$STEPS out=$OUT_DIR"

# Paper setup (Table 4) on 4 GPUs: accumulation 32 keeps its global batch of
# 1 x 32 x 4 = 128 sequences (the paper: 1 x 16 x 8).
srun torchrun --nproc_per_node=4 -m src.train.train \
  --model-dir pretrained-models/orthrus-qwen3_5-4b-init \
  --packed-cache-path data/processed/orthrus-packed-2048-asst \
  --grad-accum-steps 32 \
  --output-dir "$OUT_DIR" \
  --max-steps "$STEPS" \
  --log-every 25 \
  --save-every 100 \
  --keep-last-checkpoints 3 \
  --num-workers 8 \
  --fsdp2 \
  --auto-resume
