"""End-to-end smoke test of the Orthrus-Qwen3.5 pipeline on a layer-truncated Qwen3.5-0.8B.

    python scripts/smoke_test.py               # one GPU (fits a 6GB card)
    python scripts/smoke_test.py --nproc 8     # torchrun + FSDP2, e.g. on the H100 node

Builds what it needs if missing (truncated init checkpoint, small packed cache), then checks:
  1. Equivalence: at init the twins equal the AR weights, so with block_size=1 the
     diffusion view must reproduce the AR hidden state at every anchor -- on the training
     path (flex mask, chained delta-net states) and the generation path. A control at
     anchor+-1 must NOT match, so RoPE/mask/conv/state off-by-ones are caught. Loss
     implementations agree (KL fused/chunked/naive, CE fused/torch) and the KL teacher
     is position-aligned (slot 0's KL is ~0 at init).
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

        check_losses(model, input_ids, gen, failures)
    if failures:
        raise AssertionError("; ".join(failures))


def check_losses(model, input_ids, gen, failures) -> None:
    """KL/CE loss implementations agree (fused vs chunked vs naive), and the KL teacher is
    aligned: at init, block slot 0 equals the AR view at the anchor, so its KL is ~0."""
    import torch
    import torch.nn.functional as F

    from src.train.loss import IGNORE_INDEX, build_loss_fn, kl_chunked

    rows, hidden_size, vocab = 700, 128, 5000
    weight = (torch.randn(vocab, hidden_size, device="cuda", generator=gen) * 0.05).bfloat16()
    teacher = torch.randn(rows, hidden_size, device="cuda", generator=gen).bfloat16()
    student = (teacher.float() + 0.5 * torch.randn(rows, hidden_size, device="cuda",
                                                   generator=gen)).bfloat16()
    labels = torch.randint(0, vocab, (rows,), device="cuda", generator=gen)
    labels[torch.rand(rows, device="cuda", generator=gen) < 0.3] = IGNORE_INDEX
    keep = labels != IGNORE_INDEX

    def naive_kl(s):
        s_logp = F.log_softmax(s[keep].float() @ weight.float().T, -1)
        t_logp = F.log_softmax(teacher[keep].float() @ weight.float().T, -1)
        return (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()

    variants = {
        "naive": naive_kl,
        "chunked": lambda s: kl_chunked(s[keep], teacher[keep], weight, chunk_size=128),
        "fused": lambda s: build_loss_fn("kl", use_liger=True)(weight, s, labels, teacher),
        "ce fused": lambda s: build_loss_fn("ce", use_liger=True)(weight, s, labels),
        "ce torch": lambda s: F.cross_entropy(s.float() @ weight.float().T, labels,
                                              ignore_index=IGNORE_INDEX),
    }
    out = {}
    with torch.enable_grad():
        for name, fn in variants.items():
            s = student.clone().requires_grad_()
            loss = fn(s)
            loss.backward()
            out[name] = (loss.item(), s.grad.float())
    for name, ref in (("chunked", "naive"), ("fused", "naive"), ("ce fused", "ce torch")):
        loss_err = abs(out[name][0] - out[ref][0]) / abs(out[ref][0])
        grad_err = ((out[name][1] - out[ref][1]).norm() / out[ref][1].norm()).item()
        log(f"  loss {name} vs {ref}: loss rel err {loss_err:.1e}, grad rel err {grad_err:.1e}")
        if loss_err > 1e-3 or grad_err > 2e-2:
            failures.append(f"loss {name} disagrees with {ref}")

    if getattr(model.config, "fixed_state_layers", None) or getattr(model.config, "mtp_seed_tokens", 0):
        return                                  # slot 0 != AR at init in these variants
    inner = model.model
    inner.block_size = block_size = model.config.block_size
    anchors = torch.stack([
        torch.sort(torch.randperm(input_ids.shape[1] - block_size - 1, device="cuda",
                                  generator=gen)[:16] + 1).values
        for _ in range(input_ids.shape[0])
    ])
    seen = {}
    kl_fn = build_loss_fn("kl", use_liger=True)

    def recording(weight, hidden, labels, teacher_hidden):
        seen.update(hidden=hidden, teacher=teacher_hidden)
        return kl_fn(weight, hidden, labels, teacher_hidden)

    recording.needs_teacher = True
    model.loss_fn = recording
    model(input_ids=input_ids, anchors=anchors)
    model.loss_fn = None
    lm_head = model.lm_head.weight.float()

    def slot_kl(student_hidden, teacher_hidden):
        s_logp = F.log_softmax(student_hidden.float().reshape(-1, lm_head.shape[1]) @ lm_head.T, -1)
        t_logp = F.log_softmax(teacher_hidden.float().reshape(-1, lm_head.shape[1]) @ lm_head.T, -1)
        return (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean().item()

    slot0 = slot_kl(seen["hidden"][:, :, 0], seen["teacher"][:, :, 0])
    shifted = slot_kl(seen["hidden"][:, :, 0], seen["teacher"][:, :, 1])
    log(f"  KL teacher alignment at init: slot 0 = {slot0:.4f} (must be ~0), "
        f"vs teacher at a+1 = {shifted:.4f}")
    if slot0 > 0.05 or shifted < 0.5:
        failures.append(f"KL teacher misaligned (slot 0 {slot0:.4f}, shifted {shifted:.4f})")


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

    from src.models.generation_qwen3_5 import HybridCache
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
        cache = HybridCache(model)
        logits = model.lm_head(inner.forward_cached(text[:, :300], cache))
        cache.commit(300)
        expect_close("cache: prefill vs full", rel_err(logits, full_logits(text[:, :300])))
        inner.forward_cached(text[:, 300:316], cache)
        cache.commit(5)                                        # 11 tokens rolled back
        logits = model.lm_head(inner.forward_cached(text[:, 340:356], cache))
        cache.commit(16)
        ref = full_logits(torch.cat([text[:, :305], text[:, 340:356]], dim=1))[:, 305:]
        expect_close("cache: after partial commit vs full", rel_err(logits, ref))
        prefix = torch.cat([text[:, :305], text[:, 340:356], text[:, 360:361]], dim=1)
        expect_close("cache: diffusion draft vs uncached",
                     rel_err(inner.diffusion_draft(text[:, 360:361], cache, block_size),
                             inner.diffusion_block_forward(prefix)))

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
                log(f"  greedy generate ({name}, diffusion={use_diffusion}): {len(chosen)} "
                    f"tokens, TPF {stats['tpf']:.2f}, max argmax gap {gap:.3f}")
                if gap > 0.5:
                    failures.append(f"{name}: generated token {gap:.3f} logits below argmax")
        del init_model
        out = model.generate(prompt, max_new_tokens=64, temperature=0.8)
        assert out.shape[1] > prompt.shape[1] and (out[0] < model.config.vocab_size).all()
        log(f"  sampled generate (T=0.8): {out.shape[1] - prompt.shape[1]} tokens")

        diff_logits = model(input_ids=prompt, diffusion_train=False, is_diffusion_pass=True)
        assert diff_logits.shape[:2] == (1, block_size), diff_logits.shape
        assert torch.isfinite(diff_logits).all()
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
    run_in_subprocess(check_inference, str(args.out_dir / "final"), str(args.cache_dir),
                      str(args.init_dir))

    if not args.keep:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    log("PASSED")


if __name__ == "__main__":
    main()
