"""End-to-end smoke test of the Orthrus-Qwen3.5 pipeline on a layer-truncated Qwen3.5-0.8B.

    python scripts/smoke_test.py               # one GPU (fits a 6GB card)
    python scripts/smoke_test.py --nproc 8     # torchrun + FSDP2, e.g. on the H100 node
    python scripts/smoke_test.py --mtp-seed-tokens 3   # MTP pre-drafted blocks

Builds what it needs if missing (truncated init checkpoint, small packed cache), then checks:
  1. Init checks. Equivalence: at init the twins equal the AR weights, so with block_size=1
     the diffusion view must reproduce the AR hidden state at every anchor -- on the training
     path (flex mask, chained delta-net states) and the generation path. A control at
     anchor+-1 must NOT match, so RoPE/mask/conv/state off-by-ones are caught. Then the
     train and generation paths must agree on full blocks. With MTP seeding, the batched
     multi-anchor draft chain must equal sequential MTP decoding of each anchor (fp32).
  2. Training: loss finite and decreasing.
  3. Resume: a planted partial step-* dir is skipped by --auto-resume and pruned.
  4. Checkpoint: step-* dirs hold only the twins (equal to final's at the same step); final
     has twins in fp32, all updated, and frozen weights bit-identical to init.
  5. Inference on the final checkpoint: the generation cache matches a full recompute
     (also across a partial commit) and the uncached diffusion draft; greedy generate()
     is lossless (each token a full-recompute argmax up to bf16 ties); sampling runs.
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
             "--num-hidden-layers", str(args.num_layers),
             "--mtp-seed-tokens", str(args.mtp_seed_tokens)],
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
    block_size = inner.block_size
    gen = torch.Generator(device="cuda").manual_seed(0)
    input_ids = torch.randint(0, 150000, (2, 512), device="cuda", generator=gen)
    failures = []

    def min_cos(a, b):
        return F.cosine_similarity(a.float(), b.float(), dim=-1).min().item()

    def expect_match(name, cos):
        log(f"  {name}: min cos = {cos:.5f}")
        if cos < 0.999:
            failures.append(f"{name} (min cos {cos:.5f})")

    def at(hidden, positions):
        return torch.gather(hidden, 1, positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))

    def sorted_anchors(count, high):
        return torch.stack([
            torch.sort(torch.randperm(high, device="cuda", generator=gen)[:count] + 1).values
            for _ in range(input_ids.shape[0])
        ])

    with torch.no_grad():
        # Twins == AR weights at init, so a 1-token block reproduces the AR state.
        ar_hidden = inner(input_ids=input_ids).last_hidden_state
        inner.block_size = 1
        anchors = sorted_anchors(256, 510)   # >= 128 queries keeps flex off its decode kernel
        diff_hidden = inner.diffusion_train_forward(input_ids, anchors)[:, :, 0]
        expect_match("train path vs AR", min_cos(diff_hidden, at(ar_hidden, anchors)))
        for shift in (-1, 1):
            shifted = (anchors + shift).clamp(0, input_ids.shape[1] - 1)
            cos = min_cos(diff_hidden, at(ar_hidden, shifted))
            log(f"  control anchor{shift:+d}: min cos = {cos:.5f} (must be low)")
            if cos > 0.9:
                failures.append(f"control at anchor{shift:+d} matches; check has no teeth")
        # Short prefixes exercise the zero-padded conv left-context.
        for prefix_len in (1, 2, 3, 4, 37, 300):
            gen_hidden = inner.diffusion_block_forward(input_ids[:1, :prefix_len], diff_len=1)
            expect_match(f"gen path vs AR, prefix_len={prefix_len}",
                         min_cos(gen_hidden[:, 0], ar_hidden[:1, prefix_len - 1]))
        inner.block_size = block_size

        # The batched training path and the generation path must agree on full blocks.
        anchors = sorted_anchors(16, input_ids.shape[1] - block_size - 1)
        anchors[0, 0] = 1                      # shortest prefix: zero-padded conv lefts
        train_hidden = inner.diffusion_train_forward(input_ids, anchors)
        for row, slot in ((0, 0), (0, 7), (1, 3), (1, 15)):
            anchor = int(anchors[row, slot])
            gen_hidden = inner.diffusion_block_forward(input_ids[row : row + 1, : anchor + 1])
            expect_match(f"train vs gen path, full block at anchor {anchor}",
                         min_cos(train_hidden[row, slot], gen_hidden[0]))

        if model.mtp is not None:
            check_mtp_chain(model, gen, failures)
    if failures:
        raise AssertionError("; ".join(failures))


def check_mtp_chain(model, gen, failures) -> None:
    """Batched seed_tokens (many anchors, masked chain) vs plain causal MTP decoding of one
    anchor at a time, compared on each step's hidden state (fp32, random inputs). Tokens
    alone are too coarse: an extra visible key rarely flips the argmax."""
    import torch

    model.float()
    inner, mtp = model.model, model.mtp
    num_seeds = min(model.config.mtp_seed_tokens, inner.block_size - 1)
    seq_len = 256
    input_ids = torch.randint(0, 150000, (1, seq_len), device="cuda", generator=gen)
    ar_hidden = torch.randn(1, seq_len, inner.config.hidden_size, device="cuda", generator=gen)
    anchors = torch.tensor([[1, 2, 9, 120, 121, 250]], device="cuda")

    step_hidden = []                               # what seed_tokens feeds the lm_head

    def recording_lm_head(hidden):
        step_hidden.append(hidden)
        return model.lm_head(hidden)

    batched = mtp.seed_tokens(ar_hidden, input_ids, anchors, num_seeds,
                              inner.embed_tokens, recording_lm_head, inner._rope)
    batched_hidden = torch.stack(step_hidden, dim=2)[0]           # (A, steps, H)

    worst = 0.0
    for slot, anchor in enumerate(anchors[0].tolist()):
        # Step-1 inputs (h_p, x_{p+1}) for p < anchor, then each draft appended in turn.
        embeds = inner.embed_tokens(input_ids[:, 1 : anchor + 1])
        hidden = ar_hidden[:, :anchor]
        drafts, outs = [], []
        for _ in range(num_seeds):
            positions = torch.arange(embeds.shape[1], device="cuda").unsqueeze(0)
            out, _ = mtp.step(embeds, hidden, inner._rope(embeds, positions))
            draft = model.lm_head(out[:, -1:]).argmax(-1)
            drafts.append(int(draft))
            outs.append(out[0, -1])
            embeds = torch.cat([embeds, inner.embed_tokens(draft)], dim=1)
            hidden = torch.cat([hidden, out[:, -1:]], dim=1)
        reference = torch.stack(outs)                               # (steps, H)
        rel_err = ((batched_hidden[slot] - reference).norm(dim=-1) / reference.norm(dim=-1)).max()
        worst = max(worst, rel_err.item())
        if batched[0, slot].tolist() != drafts or rel_err > 1e-4:
            failures.append(f"MTP chain mismatch at anchor {anchor}: tokens "
                            f"{batched[0, slot].tolist()} vs {drafts}, rel err {rel_err:.2e}")
    log(f"  MTP chain ({len(anchors[0])} anchors x {num_seeds} drafts) vs sequential decoding: "
        f"max rel err = {worst:.2e}")


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

    # The last step-* (same step as final) holds exactly the twins, nothing frozen.
    step_dir = args.out_dir / f"step-{args.steps + args.steps // 2}"
    files = sorted(os.listdir(step_dir))
    assert files == ["diff_weights.safetensors", "trainer_state.pt"], files
    step_twins = load_safetensors(step_dir)
    assert step_twins.keys() == set(twins), f"step twins differ: {step_twins.keys() ^ set(twins)}"
    mismatched = [k for k in twins if not torch.equal(step_twins[k], final[k])]
    assert not mismatched, f"step-* twins != final twins: {mismatched[:5]}"
    size_mb = (step_dir / "diff_weights.safetensors").stat().st_size / 2**20
    log(f"  {step_dir.name}: twins only ({size_mb:.0f} MB), identical to final")


def check_inference(final_dir: str, cache_dir: str, init_dir: str) -> None:
    import datasets
    import torch
    from transformers import AutoTokenizer

    from src.models.generation_qwen3_5 import HybridCache, MTPCache
    from src.models.modeling_orthrus_qwen3_5 import OrthrusQwen3_5ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(final_dir)
    model = OrthrusQwen3_5ForCausalLM.from_pretrained(
        final_dir, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    inner = model.model
    block_size = model.config.block_size
    failures = []

    def rel_err(got, ref):
        return ((got.float() - ref.float()).norm() / ref.float().norm()).item()

    def expect_close(name, err, tol=0.05):
        log(f"  {name}: rel err = {err:.4f}")
        if err > tol:
            failures.append(f"{name} (rel err {err:.4f})")

    def full_logits(seq):
        return model(input_ids=seq, diffusion_train=False)

    text = torch.tensor([datasets.load_from_disk(cache_dir)[0]["input_ids"][:400]], device="cuda")
    with torch.inference_mode():
        # Generation cache vs full recompute, across a partial commit (rejected drafts must
        # not reach the delta-net state or the attention K/V).
        # The MTP cache is driven as generate() drives it: extended with each commit's
        # post-norm AR states and the token after each committed position.
        mtp_cache = MTPCache(model) if model.mtp is not None else None
        cache = HybridCache(model)
        hidden = inner.forward_cached(text[:, :300], cache)
        cache.commit(300)
        expect_close("cache: prefill vs full",
                     rel_err(model.lm_head(hidden), full_logits(text[:, :300])))
        if mtp_cache:
            mtp_cache.extend(hidden, text[:, 1:301])
        hidden = inner.forward_cached(text[:, 300:316], cache)
        cache.commit(5)                                        # 11 tokens rolled back
        if mtp_cache:
            mtp_cache.extend(hidden[:, :5], torch.cat([text[:, 301:305], text[:, 340:341]], 1))
        hidden = inner.forward_cached(text[:, 340:356], cache)
        cache.commit(16)
        if mtp_cache:
            mtp_cache.extend(hidden, torch.cat([text[:, 341:356], text[:, 360:361]], 1))
        ref = full_logits(torch.cat([text[:, :305], text[:, 340:356]], dim=1))[:, 305:]
        expect_close("cache: after partial commit vs full", rel_err(model.lm_head(hidden), ref))
        prefix = torch.cat([text[:, :305], text[:, 340:356], text[:, 360:361]], dim=1)
        seeds, seed_fn = None, None
        if mtp_cache:
            # Step-1 state at the last committed position vs a full uncached MTP pass.
            post_norm = inner(input_ids=prefix).last_hidden_state
            embeds = inner.embed_tokens(prefix[:, 1:])
            positions = torch.arange(prefix.shape[1] - 1, device="cuda").unsqueeze(0)
            full_step1, _ = model.mtp.step(embeds, post_norm[:, :-1], inner._rope(embeds, positions))
            expect_close("MTP cache: step-1 state vs full",
                         rel_err(mtp_cache.last_hidden, full_step1[:, -1:]))
            num_seeds = min(model.config.mtp_seed_tokens, block_size - 1)
            seeds = mtp_cache.seeds(num_seeds)
            anchor_at = torch.tensor([[prefix.shape[1] - 1]], device="cuda")
            ref_seeds = model.mtp.seed_tokens(post_norm, prefix, anchor_at, num_seeds,
                                              inner.embed_tokens, model.lm_head, inner._rope)
            log(f"  MTP cache: seeds {seeds[0, 0].tolist()} vs uncached {ref_seeds[0, 0].tolist()}")
            if not torch.equal(seeds, ref_seeds):
                failures.append("MTP cache seeds differ from uncached seed_tokens")
            seed_fn = model._mtp_seeds
        expect_close("cache: diffusion draft vs uncached",
                     rel_err(inner.diffusion_draft(text[:, 360:361], cache, block_size,
                                                   seed_tokens=seeds),
                             inner.diffusion_block_forward(prefix, seed_fn=seed_fn)))

        # Lossless greedy decoding: every generated token must be the argmax of a full
        # recompute up to bf16 ties (exact equality with plain AR decoding is too strict:
        # block-shaped verification flips near-ties). The init checkpoint (twins == AR)
        # accepts drafts, so partially accepted blocks are covered too.
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )["input_ids"].cuda()
        init_model = OrthrusQwen3_5ForCausalLM.from_pretrained(
            init_dir, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).cuda().eval()
        for name, candidate in (("final", model), ("init", init_model)):
            for use_diffusion in (False, True):
                out, stats = candidate.generate(prompt, max_new_tokens=96, return_stats=True,
                                                use_diffusion_mode=use_diffusion)
                hidden = candidate.model(input_ids=out).last_hidden_state.float()
                pred = (hidden @ candidate.lm_head.weight.float().T)[0, prompt.shape[1] - 1 : -1]
                chosen = out[0, prompt.shape[1]:]
                gap = (pred.max(-1).values - pred.gather(-1, chosen[:, None])[:, 0]).max().item()
                tokens_per_step = len(chosen) / (len(stats["accepted"]) + 1)
                log(f"  greedy generate ({name}, diffusion={use_diffusion}): {len(chosen)} "
                    f"tokens, {tokens_per_step:.2f} tok/step, max argmax gap {gap:.3f}")
                if gap > 0.5:
                    failures.append(f"{name}: generated token {gap:.3f} logits below argmax")
        del init_model
        out = model.generate(prompt, max_new_tokens=64, temperature=0.8)
        assert out.shape[1] > prompt.shape[1] and (out[0] < model.config.vocab_size).all()
        log(f"  sampled generate (T=0.8): {out.shape[1] - prompt.shape[1]} tokens")

        diff_logits, block_tokens = model(input_ids=prompt, diffusion_train=False,
                                          is_diffusion_pass=True, return_block_tokens=True)
        assert diff_logits.shape[:2] == (1, block_size), diff_logits.shape
        assert torch.isfinite(diff_logits).all()
        num_seeds = model.config.mtp_seed_tokens
        assert int(block_tokens[0, 0]) == int(prompt[0, -1])
        assert (block_tokens[0, 1 + num_seeds :] == model.config.mask_token_id).all()
        if num_seeds:
            assert (block_tokens[0, 1 : 1 + num_seeds] != model.config.mask_token_id).all()
        log(f"  diffusion block {tuple(diff_logits.shape)}: anchor + {num_seeds} MTP drafts "
            f"+ masks")
    if failures:
        raise AssertionError("; ".join(failures))


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
    p.add_argument("--mtp-seed-tokens", type=int, default=0,
                   help="Passed to init_model: MTP-pre-drafted block slots.")
    args = p.parse_args()
    if args.init_dir is None:
        name = args.base.split("/")[-1].lower()
        variant = f"-mtp{args.mtp_seed_tokens}" if args.mtp_seed_tokens else ""
        args.init_dir = Path(f"pretrained-models/orthrus-{name}-L{args.num_layers}{variant}-init")
    if args.steps < 2:
        p.error("--steps must be >= 2")
    return args


def main():
    os.chdir(ROOT)
    args = parse_args()
    ensure_prereqs(args)

    log("1/5 init checks (AR equivalence, train-gen parity)")
    run_in_subprocess(check_equivalence, str(args.init_dir))
    log("2-3/5 training + resume")
    check_training(args)
    log("4/5 checkpoint contents")
    check_checkpoint(args)
    log("5/5 inference from the final checkpoint")
    run_in_subprocess(check_inference, str(args.out_dir / "final"), str(args.cache_dir),
                      str(args.init_dir))

    if not args.keep:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    log("PASSED")


if __name__ == "__main__":
    main()
