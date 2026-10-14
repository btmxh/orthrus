#!/bin/bash
#SBATCH --job-name=orthrus-regenerate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.out

# Step 2/3 of the paper's data pipeline: the frozen AR model regenerates every response.
# One vLLM batch worker per GPU, each on its own request shard (scripts/data/sample_nemotron.py
# writes one shard per GPU). vLLM pins its own torch, so it lives in a separate env:
#   VLLM=/path/to/vllm-env/bin/vllm MODEL=Qwen/Qwen3.5-4B sbatch scripts/data/regenerate.sbatch
# Shards whose responses file exists are skipped, so a requeued job resumes.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VLLM=${VLLM:-vllm}
MODEL=${MODEL:?set MODEL to the frozen AR base, e.g. Qwen/Qwen3.5-4B}
DATA_DIR=${DATA_DIR:-data/nemotron}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

shards=("$DATA_DIR"/requests-*.jsonl)
gpus=${SLURM_GPUS_ON_NODE:-4}
echo "node=$(hostname) model=$MODEL shards=${#shards[@]} gpus=$gpus"

pids=()
for i in "${!shards[@]}"; do
  input=${shards[$i]}
  output=${input/requests-/responses-}
  if [ -s "$output" ]; then
    echo "skip $output (exists)"
    continue
  fi
  CUDA_VISIBLE_DEVICES=$((i % gpus)) "$VLLM" run_batch \
    -i "$input" -o "$output.tmp" \
    --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" \
    && mv "$output.tmp" "$output" &
  pids+=($!)
  # More shards than GPUs: wait for this wave before reusing a GPU.
  if (( ${#pids[@]} == gpus )); then wait "${pids[@]}"; pids=(); fi
done
wait
