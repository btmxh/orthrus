"""End-to-end smoke test of the Orthrus-Qwen3.5 pipeline on a layer-truncated Qwen3.5-0.8B.

    python scripts/smoke_test.py               # one GPU (fits a 6GB card)
    python scripts/smoke_test.py --nproc 8     # torchrun + FSDP2, e.g. on the H100 node

Builds what it needs if missing (truncated init checkpoint, small packed cache), then checks:
  1. Equivalence: at init the twins equal the AR weights, so with block_size=1 the
     diffusion view must reproduce the AR hidden state at every anchor -- on the training
     path (flex mask, chained delta-net states) and the generation path. A control at
     anchor+-1 must NOT match, so RoPE/mask/conv/state off-by-ones are caught.
  2. Training: loss finite and decreasing.
  3. Resume: a planted partial step-* dir is skipped by --auto-resume and pruned.
  4. Checkpoint: twins saved in fp32 and all updated; frozen weights bit-identical to init.
  5. Inference: the final checkpoint loads and runs a full diffusion block.
"""

from __future__ import annotations

import argparse
import glob
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET_REPO = "HuggingFaceH4/ultrachat_200k"
DATASET_FILE = "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"
SEQ_LEN = 1024


def log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def run(cmd: list[str], log_path: Path) -> None:
    """Run quietly into log_path; dump its tail on failure."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as out:
        result = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = log_path.read_text(errors="replace").splitlines()[-40:]
        print("\n".join(tail))
        raise SystemExit(f"[smoke] FAILED: {' '.join(cmd[:4])} ... (full log: {log_path})")


def ensure_prereqs(args) -> None:
    if not (args.init_dir / "config.json").exists():
        log(f"building {args.num_layers}-layer init checkpoint at {args.init_dir}")
        run([sys.executable, "-m", "src.utils.init_model", "--base", args.base,
             "--out", str(args.init_dir), "--block-size", "16",
             "--num-hidden-layers", str(args.num_layers)],
            args.out_dir.parent / "smoke-init.log")
    if not args.cache_dir.exists():
        from huggingface_hub import hf_hub_download

        log(f"building packed cache at {args.cache_dir}")
        parquet = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
        run([sys.executable, "scripts/build_packed_cache.py", "--dataset-path", parquet,
             "--out", str(args.cache_dir), "--model-dir", str(args.init_dir),
             "--seq-len", str(SEQ_LEN), "--max-samples", "1000", "--num-proc", "8"],
            args.out_dir.parent / "smoke-cache.log")


def check_equivalence(init_dir: str) -> None:
    """Runs in a spawned process so its GPU memory is released before training."""
    import torch
    import torch.nn.functional as F

    from src.models.modeling_orthrus_qwen3_5 import OrthrusQwen3_5ForCausalLM

    model = OrthrusQwen3_5ForCausalLM.from_pretrained(
        init_dir, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    inner = model.model
    gen = torch.Generator(device="cuda").manual_seed(0)
    input_ids = torch.randint(0, 150000, (2, 512), device="cuda", generator=gen)

    def min_cos(a, b):
        return F.cosine_similarity(a.float(), b.float(), dim=-1).min().item()

    def at(hidden, positions):
        return torch.gather(hidden, 1, positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))

    failures = []
    with torch.no_grad():
        ar_hidden = inner(input_ids=input_ids).last_hidden_state

        # >= 128 queries keeps flex off its decode kernel.
        inner.block_size = 1
        anchors = torch.stack([
            torch.sort(torch.randperm(510, device="cuda", generator=gen)[:256] + 1).values
            for _ in range(2)
        ])
        diff_hidden = inner.diffusion_train_forward(input_ids, anchors)[:, :, 0]
        cos = min_cos(diff_hidden, at(ar_hidden, anchors))
        log(f"  train path: min cos vs AR = {cos:.5f}")
        if cos < 0.999:
            failures.append(f"train path diverges from AR (min cos {cos:.5f})")
        for shift in (-1, 1):
            shifted = (anchors + shift).clamp(0, input_ids.shape[1] - 1)
            cos = min_cos(diff_hidden, at(ar_hidden, shifted))
            log(f"  control anchor{shift:+d}: min cos = {cos:.5f} (must be low)")
            if cos > 0.9:
                failures.append(f"control at anchor{shift:+d} matches; check has no teeth")

        # Short prefixes exercise the zero-padded conv left-context.
        for prefix_len in (1, 2, 3, 4, 37, 300):
            gen_hidden = inner.diffusion_block_forward(input_ids[:1, :prefix_len], diff_len=1)
            cos = min_cos(gen_hidden[:, 0], ar_hidden[:1, prefix_len - 1])
            log(f"  gen path prefix_len={prefix_len:3d}: min cos vs AR = {cos:.5f}")
            if cos < 0.999:
                failures.append(f"gen path diverges from AR at prefix_len={prefix_len}")
    if failures:
        raise AssertionError("; ".join(failures))


def train_losses(log_text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\[train\] .*? loss=(\S+)", log_text)]


def check_training(args) -> None:
    out = args.out_dir
    shutil.rmtree(out, ignore_errors=True)
    if args.nproc > 1:
        base = [sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={args.nproc}", "-m", "src.train.train", "--fsdp2"]
    else:
        base = [sys.executable, "-m", "src.train.train"]
    save_every = args.steps // 2
    common = ["--model-dir", str(args.init_dir), "--packed-cache-path", str(args.cache_dir),
              "--output-dir", str(out), "--seq-len", str(SEQ_LEN),
              "--num-anchor-blocks", "16", "--grad-accum-steps", "1", "--log-every", "1",
              "--save-every", str(save_every), "--keep-last-checkpoints", "1",
              "--num-workers", "2"]

    log(f"training {args.steps} steps ({'FSDP2 x' + str(args.nproc) if args.nproc > 1 else '1 GPU'})")
    run(base + common + ["--max-steps", str(args.steps)], out / "run1.out")
    train_log = out / "train.log"
    run1_text = train_log.read_text()
    losses = train_losses(run1_text)
    log(f"  losses: {' '.join(f'{x:.2f}' for x in losses)}")
    assert len(losses) == args.steps, f"expected {args.steps} logged steps, got {len(losses)}"
    assert all(math.isfinite(x) for x in losses), "non-finite loss"
    k = max(1, len(losses) // 4)
    assert sum(losses[-k:]) / k < sum(losses[:k]) / k, "loss did not decrease"
    assert (out / "final").is_dir() and (out / f"step-{args.steps}").is_dir()

    # A save preempted midway leaves a step dir without trainer_state.pt.
    stale = out / f"step-{args.steps * 100}"
    stale.mkdir()
    (stale / "config.json").write_text("{}")
    resumed_steps = args.steps + save_every
    log(f"resuming to step {resumed_steps} past a planted partial checkpoint")
    run(base + common + ["--max-steps", str(resumed_steps), "--auto-resume"], out / "run2.out")
    run2_text = train_log.read_text()[len(run1_text):]
    assert f"Restored | step={args.steps} " in run2_text, \
        f"did not resume from step-{args.steps}:\n{run2_text[:2000]}"
    assert len(train_losses(run2_text)) == save_every, "resumed run logged the wrong step count"
    assert not stale.exists(), "partial checkpoint was not pruned"
    assert (out / f"step-{resumed_steps}" / "trainer_state.pt").is_file()


def load_safetensors(directory: Path) -> dict:
    from safetensors.torch import load_file

    tensors = {}
    for shard in sorted(glob.glob(str(directory / "*.safetensors"))):
        tensors.update(load_file(shard))
    return tensors


def check_checkpoint(args) -> None:
    import torch

    from src.models.modeling_orthrus_qwen3_5 import DIFF_PARAM_SUFFIXES

    init = load_safetensors(args.init_dir)
    final = load_safetensors(args.out_dir / "final")
    assert init.keys() == final.keys(), f"key mismatch: {init.keys() ^ final.keys()}"
    twins = [k for k in final if any(s in k for s in DIFF_PARAM_SUFFIXES)]
    frozen = [k for k in final if k not in twins]
    bad_dtype = [k for k in twins if final[k].dtype != torch.float32]
    unchanged = [k for k in twins if torch.equal(final[k], init[k].float())]
    drifted = [k for k in frozen if not torch.equal(final[k], init[k])]
    log(f"  {len(twins)} twin tensors (fp32, all updated), {len(frozen)} frozen (bit-identical)")
    assert not bad_dtype, f"twins not saved in fp32: {bad_dtype[:5]}"
    assert not unchanged, f"twins never updated (no gradient?): {unchanged[:5]}"
    assert not drifted, f"frozen weights changed: {drifted[:5]}"


def check_inference(final_dir: str) -> None:
    import torch
    from transformers import AutoTokenizer

    from src.models.modeling_orthrus_qwen3_5 import OrthrusQwen3_5ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(final_dir)
    model = OrthrusQwen3_5ForCausalLM.from_pretrained(
        final_dir, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=True, add_generation_prompt=True, return_dict=True,
    )["input_ids"]
    input_ids = torch.tensor([prompt], device="cuda")
    with torch.no_grad():
        ar_logits = model(input_ids=input_ids, diffusion_train=False)
        diff_logits = model(input_ids=input_ids, diffusion_train=False, is_diffusion_pass=True)
    block_size = model.config.block_size
    assert ar_logits.shape[:2] == (1, input_ids.shape[1])
    assert diff_logits.shape[:2] == (1, block_size), diff_logits.shape
    assert torch.isfinite(ar_logits).all() and torch.isfinite(diff_logits).all()
    log(f"  AR logits {tuple(ar_logits.shape)}, diffusion block {tuple(diff_logits.shape)}")


def run_in_subprocess(fn, *fn_args) -> None:
    proc = multiprocessing.get_context("spawn").Process(target=fn, args=fn_args)
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise SystemExit(f"[smoke] FAILED in {fn.__name__}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--num-layers", type=int, default=8,
                   help="Keep the first N layers. 8 = 6 delta-net + 2 full attention.")
    p.add_argument("--init-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=Path("data/processed/smoke-packed-1024"))
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/smoke"))
    p.add_argument("--nproc", type=int, default=1, help=">1 launches torchrun with --fsdp2.")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--keep", action="store_true", help="Keep checkpoints/logs after success.")
    args = p.parse_args()
    if args.init_dir is None:
        name = args.base.split("/")[-1].lower()
        args.init_dir = Path(f"pretrained-models/orthrus-{name}-L{args.num_layers}-init")
    if args.steps < 2:
        p.error("--steps must be >= 2")
    return args


def main():
    os.chdir(ROOT)
    args = parse_args()
    ensure_prereqs(args)

    log("1/5 equivalence at init (diffusion view == AR view with block_size=1)")
    run_in_subprocess(check_equivalence, str(args.init_dir))
    log("2-3/5 training + resume")
    check_training(args)
    log("4/5 checkpoint contents")
    check_checkpoint(args)
    log("5/5 inference from the final checkpoint")
    run_in_subprocess(check_inference, str(args.out_dir / "final"))

    if not args.keep:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    log("PASSED")


if __name__ == "__main__":
    main()
