"""Build an Orthrus-Qwen3.5 init checkpoint, and sample training anchors.

    python -m src.utils.init_model --base Qwen/Qwen3.5-4B --out <dir> --block-size 32
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from src.models.config_orthrus_qwen3_5 import OrthrusQwen3_5Config
from src.models.modeling_orthrus_qwen3_5 import (
    DIFF_PARAM_SUFFIXES,
    OrthrusQwen3_5ForCausalLM,
    copy_diff_from_ar,
)


def _load_base_state_dict(model_name_or_path: str) -> dict:
    """Raw base state dict from local safetensors, downloading the repo if needed."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = model_name_or_path
    if not os.path.isdir(path):
        path = snapshot_download(model_name_or_path, allow_patterns=["*.safetensors", "*.json"])
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors found under {path}")
    state_dict = {}
    for shard in shards:
        state_dict.update(load_file(shard))
    return state_dict


def _parse_fixed_state_layers(spec: str | None, layer_types: list[str]) -> list[int]:
    """'all' -> every linear-attention layer; '0,1,2' -> those layer indices."""
    linear = [i for i, kind in enumerate(layer_types) if kind == "linear_attention"]
    if not spec:
        return []
    if spec == "all":
        return linear
    layers = sorted({int(index) for index in spec.split(",")})
    not_linear = [i for i in layers if i not in linear]
    if not_linear:
        raise ValueError(
            f"--fixed-state-layers {not_linear} are not linear-attention layers; "
            f"those are {linear}."
        )
    return layers


def init_orthrus(
    base_model_name_or_path: str,
    out_dir: str,
    block_size: int = 32,
    mask_token_id: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    force: bool = False,
    num_hidden_layers: int | None = None,
    fixed_state_layers: str | None = None,
):
    """Base Qwen3.5 checkpoint -> Orthrus init checkpoint.
    """
    save_dir = Path(out_dir)
    if save_dir.exists() and (save_dir / "config.json").exists() and not force:
        raise ValueError(f"{out_dir} already holds a checkpoint; pass --force to overwrite.")
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    base_cfg = AutoConfig.from_pretrained(base_model_name_or_path)
    text_cfg = getattr(base_cfg, "text_config", base_cfg)  # multimodal -> text tower
    cfg = OrthrusQwen3_5Config.from_dict(text_cfg.to_dict())
    cfg.block_size = block_size
    if num_hidden_layers is not None:
        # Keep only the first N decoder layers (smoke tests); later layers' weights are dropped.
        cfg.num_hidden_layers = num_hidden_layers
        cfg.layer_types = cfg.layer_types[:num_hidden_layers]
    cfg.fixed_state_layers = _parse_fixed_state_layers(fixed_state_layers, cfg.layer_types)
    cfg.use_cache = False

    if mask_token_id is None:
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = len(tokenizer)
    if not mask_token_id < cfg.vocab_size:
        raise ValueError(
            f"mask_token_id {mask_token_id} >= vocab_size {cfg.vocab_size}; "
            f"pick a reserved id or widen the vocab."
        )
    cfg.mask_token_id = mask_token_id

    model = OrthrusQwen3_5ForCausalLM(cfg).to(dtype)

    base_sd = _load_base_state_dict(base_model_name_or_path)
    remapped = {}
    for key, value in base_sd.items():
        if key.startswith("model.language_model."):
            remapped[key.replace("model.language_model.", "model.", 1)] = value
        elif key.startswith("language_model."):
            remapped["model." + key[len("language_model.") :]] = value
        elif key.startswith(("model.visual.", "visual.", "mtp", "model.mtp")):
            continue  # vision tower / multi-token-prediction head: not the text decoder
        else:
            remapped[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(remapped, strict=False)
    # The only acceptable missing keys are the freshly-created diffusion twins.
    non_diff_missing = [
        key for key in missing_keys
        if not any(suffix in key for suffix in DIFF_PARAM_SUFFIXES) and key != "lm_head.weight"
    ]
    if non_diff_missing:
        raise RuntimeError(
            f"Base text weights did not fully load: {len(non_diff_missing)} non-diff keys "
            f"missing, e.g. {non_diff_missing[:5]}. Check the checkpoint key layout."
        )

    # Warm-start every twin from its AR counterpart.
    copy_diff_from_ar(model)

    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(
        param.numel() for name, param in model.named_parameters()
        if any(suffix in name for suffix in DIFF_PARAM_SUFFIXES)
    )
    print(f"[init] saved to {save_dir}")
    print(f"[init] block_size={block_size} mask_token_id={mask_token_id} "
          f"vocab_size={cfg.vocab_size}")
    print(f"[init] layer_types: {cfg.layer_types.count('linear_attention')} linear_attention + "
          f"{cfg.layer_types.count('full_attention')} full_attention")
    print(f"[init] fixed_state_layers: {cfg.fixed_state_layers}")
    print(f"[init] params: total={total/1e9:.2f}B trainable(diff)={trainable/1e9:.2f}B "
          f"({100.0 * trainable / total:.1f}%)")
    print(f"[init] unexpected keys dropped: {len(unexpected_keys)}")
    return model, tokenizer


def assistant_anchor_candidates(assistant_mask, block_size, seq_len, min_anchor=1):
    """Anchor positions that start inside an assistant run
    """
    mask = assistant_mask.to(torch.int8).flatten()
    if mask.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=assistant_mask.device)
    # Run boundaries via a first difference on a zero-padded copy.
    zero = torch.zeros(1, dtype=torch.int8, device=mask.device)
    boundary = torch.diff(torch.cat([zero, mask, zero]).to(torch.int16))
    run_starts = (boundary == 1).nonzero(as_tuple=True)[0]
    run_ends = (boundary == -1).nonzero(as_tuple=True)[0]  # exclusive
    if run_starts.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=mask.device)
    candidates = torch.cat([
        torch.arange(int(run_start), int(run_end), device=mask.device)
        for run_start, run_end in zip(run_starts.tolist(), run_ends.tolist())
    ])
    fits_in_row = candidates <= seq_len - block_size
    return candidates[(candidates >= min_anchor) & fits_in_row]


def sample_anchors(
    seq_len: int,
    block_size: int,
    num_anchors: int,
    generator=None,
    device=None,
    assistant_mask=None,
):
    """Sorted block starts for ONE sequence, `1 <= anchor <= seq_len - block_size`.
    """
    if seq_len - block_size < 1:
        raise ValueError(f"seq_len {seq_len} too short for block_size {block_size}.")

    if assistant_mask is not None:
        candidates = assistant_anchor_candidates(assistant_mask[:seq_len], block_size, seq_len)
        if candidates.numel() == 0:
            return candidates
        num_to_draw = min(num_anchors, candidates.numel())
        picked = torch.randperm(
            candidates.numel(), generator=generator, device=candidates.device
        )[:num_to_draw]
        return torch.sort(candidates[picked]).values

    num_candidates = seq_len - block_size  # positions [1 .. seq_len - block_size]
    num_to_draw = min(num_anchors, num_candidates)
    picked = torch.randperm(num_candidates, generator=generator, device=device)[:num_to_draw]
    return torch.sort(picked + 1).values


def sample_batch_anchors(
    input_ids,
    block_size: int,
    num_anchors: int,
    generator=None,
    assistant_mask=None,
):
    batch_size, seq_len = input_ids.shape
    num_slots = min(num_anchors, seq_len - block_size)
    device = input_ids.device
    anchors = torch.zeros((batch_size, num_slots), dtype=torch.long, device=device)
    anchor_valid = torch.zeros((batch_size, num_slots), dtype=torch.bool, device=device)

    for batch in range(batch_size):
        row_mask = None if assistant_mask is None else assistant_mask[batch]
        row_anchors = sample_anchors(
            seq_len, block_size, num_slots,
            generator=generator, device=device, assistant_mask=row_mask,
        )
        if row_anchors.numel() == 0:
            # No assistant text in this row:
            row_anchors = sample_anchors(
                seq_len, block_size, num_slots, generator=generator, device=device
            )
            anchors[batch, : row_anchors.numel()] = row_anchors
            continue

        num_found = row_anchors.numel()
        anchors[batch, :num_found] = row_anchors
        anchor_valid[batch, :num_found] = True
        if num_found < num_slots:
            anchors[batch, num_found:] = row_anchors[-1]
    return anchors, anchor_valid


def parse_args():
    parser = argparse.ArgumentParser(description="Build an Orthrus-Qwen3.5 init checkpoint.")
    parser.add_argument("--base", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=32,
                        help="Parallel block size K; the paper uses 32 at every scale.")
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num-hidden-layers", type=int, default=None,
                        help="Truncate to the first N decoder layers (smoke tests only).")
    parser.add_argument("--fixed-state-layers", type=str, default=None,
                        help="Linear-attention layers whose twin only reads the AR prefix "
                             "state: 'all' or comma-separated layer indices.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_orthrus(
        args.base, args.out, block_size=args.block_size, mask_token_id=args.mask_token_id,
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }[args.dtype],
        force=args.force,
        num_hidden_layers=args.num_hidden_layers,
        fixed_state_layers=args.fixed_state_layers,
    )
