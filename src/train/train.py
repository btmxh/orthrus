"""Train the Orthrus diffusion view on the frozen Qwen3.5 hybrid backbone (FSDP2, data-parallel).

    torchrun --nproc_per_node=8 -m src.train.train --model-dir ... --packed-cache-path ... --fsdp2
"""

from __future__ import annotations

import argparse
import os

# BEFORE torch initializes its CUDA allocator: the diffusion pass fragments the pool
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import time
import traceback
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

import src.utils.distributed as dist_utils
from src.models.config_orthrus_qwen3_5 import OrthrusQwen3_5Config
from src.models.modeling_orthrus_qwen3_5 import (
    FLASH_ATTENTION_IMPLS,
    OrthrusQwen3_5DecoderLayer,
    OrthrusQwen3_5ForCausalLM,
    freeze_to_diffusion,
)
from src.train.checkpoint import (
    latest_checkpoint_dir,
    load_diff_weights,
    load_trainer_state,
    resume_batch_idx_is_valid,
    save_checkpoint,
    save_final,
)
from src.train.loss import build_loss_fn
from src.utils.data_utils import create_packed_dataloader
from src.utils.init_model import sample_batch_anchors
from src.utils.logging import (
    configure_run_logger,
    format_human_count,
    get_cosine_lr_scale,
    log_main,
    mix_seed,
)
from src.utils.patching import apply_liger, kernel_report

try:
    import wandb
except ImportError:
    wandb = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


@record
def train(args):
    distributed, rank, local_rank, world_size = dist_utils.setup_distributed(
        args.dist_timeout_minutes
    )
    is_main = rank == 0
    log_file = None
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, "train.log")
    configure_run_logger(is_main, log_file)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    resume_dir = args.resume_from or (
        latest_checkpoint_dir(args.output_dir) if args.auto_resume else None
    )
    if resume_dir is not None and not os.path.isdir(resume_dir):
        raise FileNotFoundError(f"--resume-from does not exist: {resume_dir}")
    if resume_dir:
        log_main(is_main, f"Resuming from {resume_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    cfg = OrthrusQwen3_5Config.from_pretrained(args.model_dir)
    cfg.use_cache = False
    cfg._attn_implementation = args.attn_implementation
    if cfg.block_size is None or cfg.mask_token_id is None:
        raise ValueError(
            f"{args.model_dir}/config.json is missing block_size or mask_token_id; build the "
            f"init checkpoint with `python -m src.utils.init_model`."
        )
    # Load in fp32 so the trainable twins keep fp32 master weights (bf16 rounds away small
    # AdamW updates); only the frozen params drop to --dtype. Twins are warm-started from
    # the AR weights by init_model, so a trained --model-dir keeps its twins.
    model = OrthrusQwen3_5ForCausalLM.from_pretrained(
        args.model_dir, config=cfg, dtype=torch.float32
    )
    n_train, n_total = freeze_to_diffusion(model)
    for param in model.parameters():
        if not param.requires_grad:
            param.data = param.data.to(dtype)
    if resume_dir is not None:
        # step-* checkpoints hold only the twins; the frozen backbone is --model-dir.
        load_diff_weights(resume_dir, model, args.model_dir, is_main)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    if args.liger:
        apply_liger(model, rms_norm=True, swiglu=True, is_main=is_main)

    loss_fn = build_loss_fn(kind=args.loss, use_liger=args.liger)
    model.loss_fn = loss_fn

    model = model.to(device)

    if args.fsdp2:
        if not distributed:
            raise ValueError("--fsdp2 requires a torchrun launch (WORLD_SIZE > 1).")
        dist_utils.apply_fsdp2(
            model, modules_to_shard=[OrthrusQwen3_5DecoderLayer],
            activation_checkpointing=args.activation_checkpointing,
        )
        log_main(is_main, "FSDP2 initialized (fully_shard @ decoder-layer granularity).")
    elif distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    base_model = model.module if (distributed and not args.fsdp2) else model

    wandb_run = None
    if args.wandb and is_main:
        if wandb is None:
            raise ImportError("wandb is not installed; `pip install wandb` or drop --wandb.")
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                               config=vars(args))

    kernels = kernel_report()
    block_size = cfg.block_size
    log_main(is_main, "=" * 88)
    log_main(is_main, f"Orthrus-Qwen3.5 training | model_dir={args.model_dir} | "
                      f"world_size={world_size} | dtype={args.dtype} | "
                      f"attn={args.attn_implementation}")
    log_main(is_main, f"Config | seq_len={args.seq_len} | block_size={block_size} | "
                      f"micro_bsz={args.micro_batch_size} | grad_accum={args.grad_accum_steps} | "
                      f"anchors/seq={args.num_anchor_blocks} | lr={args.lr:.2e}")
    log_main(is_main, f"Supervised positions/row = {args.num_anchor_blocks * (block_size - 1):,} "
                      f"(num_anchor_blocks x (block_size - 1))")
    tokens_per_step = (args.seq_len * args.micro_batch_size * args.grad_accum_steps * world_size)
    log_main(is_main, f"Global tokens/step = {tokens_per_step:,}")
    log_main(is_main, f"Params | total={format_human_count(total_params)} | "
                      f"trainable={format_human_count(trainable_params)} "
                      f"({100.0 * trainable_params / max(1, total_params):.2f}%) | "
                      f"{n_train}/{n_total} tensors")
    log_main(is_main, f"Kernels | gated_delta_rule={kernels['gated_delta_rule']} | "
                      f"causal_conv1d={kernels['causal_conv1d']} | "
                      f"loss={loss_fn.kind} fused={loss_fn.is_fused}")
    if not kernels["gated_delta_rule_is_fla"]:
        log_main(is_main, "WARNING: gated-delta-rule resolved to the TORCH fallback instead of fla. Correct, but far slower and numerically different.")
    if not kernels["causal_conv1d_is_kernel"]:
        log_main(is_main, "WARNING: the causal conv is not fla's kernel. Expected fla.modules.conv; correct but slower and numerically different.")
    log_main(is_main, "=" * 88)

    loader = create_packed_dataloader(
        cache_path=args.packed_cache_path, seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size, num_workers=args.num_workers,
        distributed=distributed, shuffle=True, seed=args.seed,
        require_assistant_mask=not args.allow_missing_assistant_mask,
    )
    sampler = loader.sampler

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    updates_per_epoch = max(1, len(loader) // args.grad_accum_steps)
    total_steps = updates_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda cur: get_cosine_lr_scale(cur, warmup_steps, total_steps),
    )
    log_main(is_main, f"LR schedule | cosine | warmup={warmup_steps} | total={total_steps}")

    step, start_epoch, resume_batch_idx = 0, 0, -1
    if resume_dir is not None:
        step, start_epoch, resume_batch_idx, saved_world, saved_ga = load_trainer_state(
            resume_dir, model, optimizer, scheduler, args.fsdp2, distributed
        )
        log_main(is_main, f"Restored | step={step} epoch={start_epoch + 1} "
                          f"after batch_idx={resume_batch_idx}")
        if not resume_batch_idx_is_valid(saved_world, saved_ga, world_size, args.grad_accum_steps):
            log_main(is_main,
                     f"WARNING: checkpoint was written at world_size={saved_world or '?'} "
                     f"grad_accum={saved_ga or '?'} but this run is world_size={world_size} "
                     f"grad_accum={args.grad_accum_steps}. The per-rank batch_idx does not "
                     f"transfer; restarting this epoch from batch 0 (step/LR preserved).")
            resume_batch_idx = -1

    if args.seq_len < block_size + 1:
        raise ValueError(f"seq_len {args.seq_len} too short for block_size {block_size}.")

    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    amp_context = (lambda: torch.autocast(device_type="cuda", dtype=dtype)) if use_amp \
        else nullcontext
    anchor_gen = torch.Generator(device=device)

    train_start = last_log = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        epoch_iter = (tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
                      if (tqdm is not None and is_main) else loader)

        for batch_idx, batch in enumerate(epoch_iter):
            if epoch == start_epoch and batch_idx <= resume_batch_idx:
                continue
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            assistant_mask = batch.get("assistant_mask")
            if assistant_mask is not None:
                assistant_mask = assistant_mask.to(device, non_blocking=True)

            with amp_context():
                anchor_gen.manual_seed(mix_seed(args.seed, epoch, rank, batch_idx))
                anchors, anchor_valid = sample_batch_anchors(
                    input_ids, block_size, args.num_anchor_blocks,
                    generator=anchor_gen, assistant_mask=assistant_mask,
                )

                loss, _hidden, _labels = model(
                    input_ids=input_ids, anchors=anchors, anchor_valid=anchor_valid,
                    supervise_mask=assistant_mask,
                )
                loss = loss / args.grad_accum_steps

            is_sync_step = (batch_idx + 1) % args.grad_accum_steps == 0
            if distributed and args.fsdp2:
                model.set_requires_gradient_sync(is_sync_step)
                sync_ctx = nullcontext()
            elif distributed:
                sync_ctx = nullcontext() if is_sync_step else model.no_sync()
            else:
                sync_ctx = nullcontext()
            with sync_ctx:
                loss.backward()

            if not is_sync_step:
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_main and step % args.log_every == 0:
                now = time.perf_counter()
                elapsed = max(now - last_log, 1e-6)
                last_log = now
                tokens_logged = (args.seq_len * args.micro_batch_size * args.grad_accum_steps
                                 * world_size * args.log_every)
                loss_value = loss.item() * args.grad_accum_steps
                frac_valid = anchor_valid.float().mean().item()
                log_main(is_main, f"[train] epoch={epoch + 1}/{args.epochs} step={step} "
                                  f"loss={loss_value:.4f} "
                                  f"lr={optimizer.param_groups[0]['lr']:.2e} "
                                  f"tok/s={tokens_logged / elapsed:,.0f} "
                                  f"anchor_valid={frac_valid:.2f}")
                if wandb_run is not None:
                    wandb_run.log({"train/loss": loss_value,
                                   "train/lr": optimizer.param_groups[0]["lr"],
                                   "train/tokens_per_sec": tokens_logged / elapsed,
                                   "train/anchor_valid": frac_valid,
                                   "train/epoch": epoch + 1}, step=step)

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(
                    os.path.join(args.output_dir, f"step-{step}"),
                    model,
                    base_model,
                    optimizer,
                    scheduler,
                    step,
                    epoch,
                    batch_idx,
                    args,
                    distributed,
                    is_main,
                )

            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break

    final_dir = save_final(
        args.output_dir,
        model,
        base_model,
        tokenizer,
        args.fsdp2,
        distributed,
        is_main
    )
    
    if is_main:
        mins = (time.perf_counter() - train_start) / 60.0
        log_main(is_main, "=" * 88)
        log_main(is_main, f"Training complete | steps={step} | {mins:.2f} min | {final_dir}")
        log_main(is_main, "=" * 88)
        if wandb_run is not None:
            wandb_run.finish()
    dist_utils.cleanup_distributed(distributed)


def parse_args():
    # Defaults follow the paper's training setup (arXiv 2605.12825, Appendix A, Table 4).
    p = argparse.ArgumentParser(description="Train Orthrus-Qwen3.5 (dual-view diffusion).")
    # Data
    p.add_argument("--packed-cache-path", type=str, required=True,
                   help="Prebuilt packed dataset (input_ids + assistant_mask).")
    p.add_argument("--allow-missing-assistant-mask", action="store_true",
                   help="Accept a cache without assistant_mask. Anchors then land anywhere "
                        "in the row, so the diffusion view also trains on prompt tokens.")
    p.add_argument("--num-workers", type=int, default=8)
    # Model / IO
    p.add_argument("--model-dir", type=str, required=True,
                   help="Orthrus init checkpoint (see src/utils/init_model.py).")
    p.add_argument("--output-dir", type=str, default="checkpoints/orthrus-qwen3_5-4b")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--auto-resume", action="store_true",
                   help="Resume from the newest step-* in --output-dir (for preempted jobs).")
    p.add_argument("--keep-last-checkpoints", type=int, default=5)
    p.add_argument("--attn-implementation", type=str, default="flash_attention_2",
                   choices=list(FLASH_ATTENTION_IMPLS),
                   help="Flash attention ONLY, it builds the AR K/V cache the diffusion. "
                        "The diffusion pass uses flex attention regardless.")
    # Training
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=16,
                   help="Paper: 16 x 8 GPUs x micro-batch 1 = global batch 128.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=0, help="0 = run full epochs")
    p.add_argument("--num-anchor-blocks", type=int, default=256,
                   help="Draft-block anchors sampled per sequence. Bounded by "
                        "seq_len - block_size; supervised positions per row are "
                        "num_anchor_blocks x (block_size - 1).")
    p.add_argument("--loss", choices=["kl", "ce"], default="kl",
                   help="kl: forward KL to the frozen AR view's distribution (paper). "
                        "ce: cross-entropy on the data tokens (the paper's ablation).")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="0 disables clipping.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    # Kernels / parallelism
    p.add_argument("--liger", action="store_true", default=True,
                   help="Apply Liger RMSNorm/SwiGLU + the fused linear loss (KL or CE).")
    p.add_argument("--no-liger", dest="liger", action="store_false")
    p.add_argument("--fsdp2", action="store_true")
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--dist-timeout-minutes", type=int, default=180)
    # Logging
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="orthrus-qwen3_5")
    p.add_argument("--wandb-run-name", type=str, default=None)

    args = p.parse_args()
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0.0, 1.0).")
    if args.num_anchor_blocks <= 0:
        raise ValueError("--num-anchor-blocks must be > 0.")
    if args.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be > 0.")
    return args


if __name__ == "__main__":
    try:
        train(parse_args())
    except Exception:
        print(f"[rank{os.environ.get('RANK', '0')}] Unhandled training exception:")
        traceback.print_exc()
        raise
