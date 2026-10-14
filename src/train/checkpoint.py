"""Checkpoint save/load/prune.
"""

from __future__ import annotations

import glob
import os
import re
import shutil

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from safetensors import safe_open
from safetensors.torch import save_file

from src.utils.logging import log_main

TRAINER_STATE_FILE = "trainer_state.pt"
# step-* checkpoints hold only the trainable twins; the frozen backbone is --model-dir.
DIFF_WEIGHTS_FILE = "diff_weights.safetensors"


def _step_dirs(output_dir: str) -> tuple[list[tuple[int, str]], list[str]]:
    """(sorted complete step-* dirs, incomplete ones). trainer_state.pt is written last,
    so a save preempted midway leaves a dir without it."""
    complete, incomplete = [], []
    for name in os.listdir(output_dir):
        match = re.fullmatch(r"step-(\d+)", name)
        full = os.path.join(output_dir, name)
        if not (match and os.path.isdir(full)):
            continue
        if os.path.isfile(os.path.join(full, TRAINER_STATE_FILE)):
            complete.append((int(match.group(1)), full))
        else:
            incomplete.append(full)
    return sorted(complete), incomplete


def latest_checkpoint_dir(output_dir: str) -> str | None:
    """Newest complete step-*; preempted partial saves are skipped."""
    if not os.path.isdir(output_dir):
        return None
    complete, _ = _step_dirs(output_dir)
    return complete[-1][1] if complete else None


def prune_old_checkpoints(output_dir: str, keep_last: int) -> list[str]:
    """Keep the newest `keep_last` complete checkpoints. Partial saves are stale by the
    time this runs (called after a save finishes), so they are removed rather than
    counted -- otherwise a stale high-numbered one would evict every real checkpoint."""
    if keep_last <= 0:
        return []
    complete, incomplete = _step_dirs(output_dir)
    stale = [path for _, path in complete[:-keep_last]] + incomplete
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)
    return stale


def resume_batch_idx_is_valid(saved_world, saved_grad_accum, world_size, grad_accum_steps):
    return saved_world == world_size and saved_grad_accum == grad_accum_steps


def save_trainer_state(
    save_dir,
    model,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    batch_idx: int,
    fsdp2: bool,
    distributed: bool,
    is_main: bool,
    grad_accum_steps: int,
):
    # Under FSDP2 the optimizer state is sharded, gather to rank 0 (cpu_offload bounds host memory).
    if fsdp2 and distributed:
        optim_state = get_optimizer_state_dict(
            model, optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    else:
        optim_state = optimizer.state_dict()
    if not is_main:
        return
    # Write-then-rename so a preempted save never leaves a truncated trainer_state.pt.
    state_path = os.path.join(save_dir, TRAINER_STATE_FILE)
    tmp_path = state_path + ".tmp"
    torch.save(
        {
            "optimizer": optim_state,
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "world_size": dist.get_world_size() if distributed else 1,
            "grad_accum_steps": grad_accum_steps,
        },
        tmp_path,
    )
    os.replace(tmp_path, state_path)


def load_trainer_state(
    checkpoint_dir,
    model,
    optimizer,
    scheduler,
    fsdp2: bool,
    distributed: bool,
):
    state_path = os.path.join(checkpoint_dir, TRAINER_STATE_FILE)
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"{checkpoint_dir} has no {TRAINER_STATE_FILE}.")

    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if fsdp2 and distributed:
        set_optimizer_state_dict(
            model, optimizer, optim_state_dict=payload["optimizer"],
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    else:
        optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return (
        int(payload["step"]),
        int(payload["epoch"]),
        int(payload["batch_idx"]),
        int(payload.get("world_size", 0)),
        int(payload.get("grad_accum_steps", 0)),
    )


def save_checkpoint(
    save_dir,
    model,
    base_model,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    batch_idx: int,
    args,
    distributed: bool,
    is_main: bool,
):
    if args.fsdp2 and distributed:
        # Frozen params are dropped before the all-gather, so only the twins are gathered.
        diff_state = get_model_state_dict(
            model, options=StateDictOptions(
                full_state_dict=True, cpu_offload=True, ignore_frozen_params=True
            ),
        )
    else:
        diff_state = {
            name: param.detach().cpu()
            for name, param in base_model.named_parameters() if param.requires_grad
        }
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        save_file(
            {name: tensor.contiguous() for name, tensor in diff_state.items()},
            os.path.join(save_dir, DIFF_WEIGHTS_FILE),
            metadata={"model_dir": os.path.abspath(args.model_dir)},
        )
    save_trainer_state(save_dir, model, optimizer, scheduler, step, epoch, batch_idx,
                       fsdp2=args.fsdp2, distributed=distributed, is_main=is_main,
                       grad_accum_steps=args.grad_accum_steps)
    if is_main:
        log_main(is_main, f"[checkpoint] saved to {save_dir}")
        for removed in prune_old_checkpoints(args.output_dir, args.keep_last_checkpoints):
            log_main(is_main, f"[checkpoint] pruned {removed}")


def load_diff_weights(checkpoint_dir, model, model_dir: str, is_main: bool) -> None:
    """Load a step-* checkpoint's twins onto a model built from --model-dir (frozen params
    already set). Legacy full-model checkpoints also work: only their twin tensors are read."""
    twin_names = {name for name, param in model.named_parameters() if param.requires_grad}
    shards = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors in {checkpoint_dir}")
    state = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as reader:
            saved_model_dir = (reader.metadata() or {}).get("model_dir")
            if saved_model_dir and saved_model_dir != os.path.abspath(model_dir):
                log_main(is_main, f"WARNING: {checkpoint_dir} was trained on --model-dir "
                                  f"{saved_model_dir}, resuming on {model_dir}. The frozen "
                                  f"backbone must be the same weights.")
            for key in reader.keys():
                if key in twin_names:
                    state[key] = reader.get_tensor(key)
    missing = twin_names - state.keys()
    if missing:
        raise RuntimeError(
            f"{checkpoint_dir} lacks {len(missing)} twin tensors, e.g. {sorted(missing)[:3]}"
        )
    model.load_state_dict(state, strict=False)


def save_final(
    output_dir,
    model,
    base_model,
    tokenizer,
    fsdp2: bool,
    distributed: bool,
    is_main: bool,
):
    final_dir = os.path.join(output_dir, "final")
    if fsdp2 and distributed:
        final_state = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        if is_main:
            os.makedirs(final_dir, exist_ok=True)
            base_model.save_pretrained(final_dir, state_dict=final_state)
            tokenizer.save_pretrained(final_dir)
    elif is_main:
        os.makedirs(final_dir, exist_ok=True)
        (model.module if distributed else model).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
    return final_dir
