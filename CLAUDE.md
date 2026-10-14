# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Orthrus is a research implementation that bolts a parallel "diffusion view" onto a frozen autoregressive (AR) LLM. Each attention or mixer layer gets a trainable twin (`*_diff` params, about 16% of the model). The diffusion view drafts a whole block of tokens in one pass, reading the AR view's KV cache or recurrent state. The AR view then verifies the draft, so output is lossless (greedy prefix match, or speculative-style rejection sampling when temperature > 0).

The repo has no unit tests, linter config or CI. After touching anything under `src/models`, `src/train` or `src/utils`, run the smoke test (about 1.5 min, fits a 6 GB GPU):

```bash
python scripts/smoke_test.py            # single GPU
python scripts/smoke_test.py --nproc 8  # torchrun + FSDP2 (H100 node; the 1-GPU run can't exercise FSDP2)
```

It builds an 8-layer truncated Qwen3.5-0.8B init and a small packed cache on first run. It checks:
- The diffusion view reproduces the AR view at init (block_size=1, with an anchor±1 control), so off-by-ones in RoPE, masks or delta-net state fail here.
- Training loss decreases.
- `--auto-resume` and pruning work, including with a planted partial checkpoint.
- Twins are saved in fp32 and updated, and frozen weights are bit-identical to init.
- Inference runs on the final checkpoint.

Real training runs on H100s; this machine is for development.

## Setup

Dev environment is Nix + direnv + uv. `flake.nix` provides Python 3.12 and uv. `.envrc` runs `uv sync --extra train` and activates `.venv`, so `cd` into the repo is enough. The flake also sets the NixOS-specific paths that PyPI torch/triton wheels need (`LD_LIBRARY_PATH` with libstdc++/zlib, and `/run/opengl-driver/lib` for libcuda via `TRITON_LIBCUDA_PATH`). If imports fail with missing `.so` files, add the library there rather than patching wheels. Add dependencies with `uv add`, which updates `pyproject.toml` and `uv.lock`. Flakes only see git-tracked files, so a new file `flake.nix` depends on must be at least `git add`ed.

`flash-attn` is not installed locally. With the `kernels` dev dependency, transformers swaps `flash_attention_2` for the prebuilt hub kernel `kernels-community/flash-attn2`, so `config._attn_implementation` may read as that name (`is_flash_attention()` accepts it). Qwen3.5 training and generation require flash attention. On GPUs with under 160 KB of shared memory per block (consumer Ampere/Ada), `fused_flex_attention` shrinks the flex backward tiles, because head_dim=256 does not fit the defaults. This is a no-op on A100/H100.

All imports are package-absolute (`from src.models...`), so run modules from the repo root with `python -m`.

## Three independent backends

They share concepts but no code:

- **`src/model.py`**: Qwen3 (pure attention) PyTorch model and inference. This is the remote-code model behind the released HF checkpoints (`chiennv/Orthrus-Qwen3-*`, loaded with `trust_remote_code=True`). `OrthrusLM.generate(use_diffusion_mode=True)` runs the draft → verify loop over a `DynamicCache`. After each step it calls `past_key_values.crop(start_idx)` to roll back rejected tokens. `use_diffusion_mode=False` falls back to the HF `GenerationMixin`. Note: `model.py` aliases `Qwen3Config as OrthrusConfig` and does not use `src/configuration.py`.
- **`src/model_mlx.py`**: standalone MLX port of the Qwen3 model for Apple Silicon (`load_model_and_tokenizer`, `mlx_generate`). Tested with `mlx==0.31.2` and `mlx-lm==0.31.3`.
- **`src/models/`**: Qwen3.5 hybrid backbone (full attention plus gated delta-net linear attention). This is the only backbone with training support (`src/train/`). It is a separate design, not an extension of `model.py`.

## Qwen3.5 architecture (`src/models/modeling_orthrus_qwen3_5.py`)

- **Twins**: `DIFF_PARAM_SUFFIXES` is the source of truth for what is trainable. `freeze_to_diffusion` matches parameter names against it. `copy_diff_from_ar` warm-starts every twin from its AR counterpart. `init_model` uses the same list to decide which missing keys are acceptable. If you add a diffusion-side parameter, add its suffix to this list and to `copy_diff_from_ar`. Only `init_model` runs `copy_diff_from_ar`; the trainer never does, so a trained `--model-dir` keeps its twins. The trainer loads in fp32 and casts only frozen params to `--dtype`, so twins keep fp32 master weights and checkpoints store them in fp32.
- **Two-pass forward**: `diffusion_train_forward` first runs a frozen AR pass under `no_grad`. During that pass each layer's mixer has `_capture_ar=True` and stashes context in `_ar_cache`: K/V for attention layers, and per-anchor recurrent states plus conv left-context for delta-net layers. Then one batched diffusion pass runs over every anchor block at once.
- **Anchors**: training samples many block starts ("anchors") per sequence (`src/utils/init_model.py:sample_batch_anchors`, restricted to assistant tokens via `assistant_mask`). Each block is `[anchor token, MASK × (block_size-1)]`, and position t predicts token t+1 (`src/train/loss.py:block_local_labels`). Anchors must be ascending, because delta-net prefix states are chained segment by segment.
- **Masking**: diffusion queries attend to a strict prefix of AR keys before their own anchor, plus their own block only (block-diagonal). This uses flex attention (`build_diffusion_block_mask`, compiled). The AR pass must use flash attention (`FLASH_ATTENTION_IMPLS`); eager and sdpa are rejected for both training and generation.
- **Inference API**: `model.generate(input_ids, max_new_tokens=..., temperature=..., use_diffusion_mode=True)` (batch 1, `src/models/generation_qwen3_5.py`). It is lossless draft-and-verify with the acceptance logic of `model.py`, over a `HybridCache`: attention K/V plus, per delta-net layer, the recurrent state and conv left-context of the committed prefix. A rejected draft must never reach the recurrent state, so verification keeps the block's post-conv inputs pending and `commit(n)` replays only the accepted prefix. `use_diffusion_mode=False` runs the same loop one token at a time (the AR baseline). Report decoding efficiency as **TPF** (`return_stats=True` → `tpf`): generated tokens / (draft passes + verify passes). Only drafting and verifying count as forward passes, not prefill or the MTP head, so a draft+verify cycle is 2 passes and the AR baseline is about 1. Greedy outputs of the two modes can differ at bf16 near-ties, so check losslessness as "each token is a full-recompute argmax up to ties", not as token equality. The uncached `forward(diffusion_train=False, is_diffusion_pass=True)` / `diffusion_block_forward` remain as references.
- **Fixed-state GDN** (`config.fixed_state_layers`, `init_model --fixed-state-layers all|0,1,2`): those layers' twins never write the block into the AR prefix state, only read it (gated delta rule with beta=0), so they have q-only projection/conv and no k/v/beta. The readout must stay kernel-exact (fp32 l2norm, `1/sqrt(dk)`, cumulative decay). Prefix readouts mostly sit near `norm_diff`'s eps, so the decay acts as an amplitude gate and is not normalized away. These layers fail the block_size=1 AR equivalence by design, so the smoke test checks their readout against `torch_chunk_gated_delta_rule` with beta=0 instead (`--fixed-state-layers`).
- **Kernels**: delta-net uses fla's `causal_conv1d` and `torch_chunk_gated_delta_rule` (which should resolve to fla). The trainer logs a WARNING if either falls back to torch, because the fallback is slower and numerically different. Liger patches RMSNorm (gemma casting, offset 1.0), SwiGLU and fused linear cross-entropy.

## Qwen3.5 training pipeline

1. **Init checkpoint**: load base weights, remap multimodal keys, drop vision/MTP weights, and copy twins from AR.
   ```bash
   python -m src.utils.init_model --base Qwen/Qwen3.5-4B --out pretrained-models/orthrus-qwen3_5-4b-init
   ```
   `block_size` and `mask_token_id` are stored in `config.json`, and training refuses to run without them.
2. **Packed data cache**: takes rows with a `messages` column, tokenizes them with the chat template, builds an `assistant_mask`, and packs fixed-length rows.
   ```bash
   python scripts/build_packed_cache.py --dataset-path data/sft.jsonl --out data/processed/orthrus-packed-2048-asst --max-samples 2000  # smoke build
   ```
   The tokenizer must match the training tokenizer, because `assistant_token_mask` walks `<|im_start|>assistant … <|im_end|>` and skips `<think>` blocks. The script fails if fewer than 5% of tokens are assistant tokens. `--seq-len` must match between cache and trainer.
3. **Train** (FSDP2 at decoder-layer granularity, or DDP without `--fsdp2`):
   ```bash
   torchrun --nproc_per_node=4 -m src.train.train --model-dir pretrained-models/orthrus-qwen3_5-4b-init \
     --packed-cache-path data/processed/orthrus-packed-2048-asst --output-dir checkpoints --fsdp2 --auto-resume \
     --grad-accum-steps 32
   ```
   For a quick check, add `--max-steps N --save-every 0`. SLURM launcher: `sbatch scripts/submit_train.sbatch` (env overrides: `STEPS`, `OUT_DIR`).
   `<output-dir>/step-N/` holds only the trainable twins (`diff_weights.safetensors`, fp32) plus `trainer_state.pt`. Resuming rebuilds the frozen backbone from `--model-dir`, so pass the same `--model-dir` (the twins file records it, and a mismatch logs a warning). `final/` is a full HF model. `--auto-resume` picks the newest complete `step-*`. If `world_size` or `grad_accum` changed since the checkpoint, the run restarts the epoch from batch 0 but keeps step and LR.

Defaults follow the paper (arXiv 2605.12825, Appendix A, Table 4): L=2048, 256 anchors, K=32, 2 epochs, peak LR 2e-4 with cosine schedule, 5% warmup, clipping 1.0, micro-batch 1 and global batch 128. `submit_train.sbatch` targets 4×H100, so it uses accumulation 32 to keep the global batch at 128. Known deviations from the paper:
- The twins keep fp32 master weights; the paper optimizes them in bf16.
- The loss is CE on dataset tokens. The paper uses forward KL to the frozen AR distribution, and its CE ablation loses TPF (6.35 → 5.86 on Qwen3-8B).
- The paper's data is Nemotron-Post-Training-Dataset-v2 (chat:math:code 1:1:1) with responses regenerated by the frozen AR model. The repo has no pipeline for that.

`data/`, `pretrained-models/`, `checkpoints/`, `wandb/` and `slurm-logs/` are gitignored.
