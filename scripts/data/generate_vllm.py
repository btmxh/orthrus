"""Step 2/3 of the paper's data pipeline: the frozen AR model regenerates every response,
through vLLM's offline engine (no HTTP server). Runs in the isolated vLLM env:

    uv sync --project scripts/data/vllm-env                      # once, on the GPU node
    scripts/data/vllm-env/.venv/bin/python scripts/data/generate_vllm.py \
        --model Qwen/Qwen3.5-4B --dir data/nemotron

Quick benchmark instead (no sampling step, no effect on the real run): streams --bench
prompts per domain into a fresh data/bench/<timestamp>/ and generates them there.

    scripts/data/vllm-env/.venv/bin/python scripts/data/generate_vllm.py \
        --model Qwen/Qwen3.5-4B --bench 1024 [--chunk-size 512]

Starts one worker per GPU, each owning one requests-*.jsonl shard (sample_nemotron.py
writes one per GPU). A worker loads the model once and feeds LLM.chat chunks of
--chunk-size requests, so vLLM's continuous batching keeps the GPU full; after each
chunk it appends the responses to responses-*.jsonl and fsyncs. A rerun skips requests
already answered, so an interrupted run loses at most one chunk per GPU. Output records
use vLLM's batch output format, which merge_responses.py reads.

Performance metrics go to <dir>/metrics/: worker-*.jsonl (config, model load, one line
per chunk) and summary.json (aggregated; also printed). Per chunk: throughput, output
lengths, finish reasons, prefix-cache hits, preemptions (KV-cache pressure), batch
concurrency rebuilt from vLLM's per-request timestamps, the tail (90% -> 100% of
requests finished, when the batch drains), and nvidia-smi GPU utilization.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import uuid


def load_done(path: str) -> set[str]:
    """custom_ids already answered. A torn last line (killed mid-write) is cut off."""
    done = set()
    if not os.path.exists(path):
        return done
    good_bytes = 0
    with open(path, "rb") as f:
        for line in f:
            try:
                done.add(json.loads(line)["custom_id"])
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                break
            good_bytes += len(line)
    if good_bytes < os.path.getsize(path):
        with open(path, "r+b") as f:
            f.truncate(good_bytes)
    return done


def record(custom_id, status, body=None, error=None) -> str:
    return json.dumps({
        "id": f"vllm-{uuid.uuid4().hex}", "custom_id": custom_id,
        "response": {"status_code": status, "request_id": f"vllm-{uuid.uuid4().hex}",
                     "body": body},
        "error": error,
    }, ensure_ascii=False) + "\n"


def percentiles(values) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"mean": round(statistics.fmean(ordered), 1), "p50": pick(0.5), "p90": pick(0.9),
            "p99": pick(0.99), "max": ordered[-1]}


class GpuSampler:
    """Polls nvidia-smi once a second for this worker's GPU; no-op if unavailable."""

    def __init__(self, gpu: str):
        self.gpu, self.samples, self._stop = gpu, [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        command = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                   "--format=csv,noheader,nounits", "-i", self.gpu]
        while not self._stop.is_set():
            try:
                util, memory = subprocess.check_output(command, text=True, timeout=10).split(",")
                self.samples.append((time.time(), float(util), float(memory)))
            except (OSError, subprocess.SubprocessError, ValueError):
                return
            self._stop.wait(1.0)

    def window(self, start: float, end: float) -> dict:
        inside = [s for s in self.samples if start <= s[0] <= end]
        if not inside:
            return {}
        last = [s for s in inside if s[0] >= end - 0.1 * (end - start)] or inside[-1:]
        return {"util_mean": round(statistics.fmean(s[1] for s in inside), 1),
                "util_last10pct": round(statistics.fmean(s[1] for s in last), 1),
                "mem_used_mib_max": max(s[2] for s in inside)}

    def stop(self):
        self._stop.set()


def batch_metrics(results) -> dict:
    """Batch occupancy rebuilt from vLLM's per-request engine timestamps (monotonic):
    each request is running from scheduled_ts to last_token_ts."""
    spans = [(r.metrics.scheduled_ts, r.metrics.last_token_ts) for r in results
             if getattr(r, "metrics", None) is not None
             and r.metrics.scheduled_ts > 0 and r.metrics.last_token_ts > 0]
    if not spans:
        return {}
    events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
    start, end = events[0][0], events[-1][0]
    running = peak = 0
    area = low_time = 0.0
    previous = start
    timeline = []
    for t, delta in events:
        timeline.append((previous, t, running))
        previous = t
        running += delta
        peak = max(peak, running)
    for t0, t1, level in timeline:
        area += level * (t1 - t0)
    for t0, t1, level in timeline:
        if level < 0.25 * peak:
            low_time += t1 - t0
    span = max(end - start, 1e-9)
    finishes = sorted(e for _, e in spans)
    tail = finishes[-1] - finishes[int(0.9 * (len(finishes) - 1))]
    return {"engine_span_s": round(span, 2), "concurrency_peak": peak,
            "concurrency_mean": round(area / span, 1),
            "frac_time_below_25pct_peak": round(low_time / span, 3),
            "tail_90_to_100pct_s": round(tail, 2), "tail_frac": round(tail / span, 3),
            "preemptions": sum(r.metrics.num_preemptions for r in results
                               if getattr(r, "metrics", None) is not None)}


def run_worker(args) -> None:
    from vllm import LLM, SamplingParams, __version__ as vllm_version

    output = args.shard.replace("requests-", "responses-")
    shard_name = os.path.basename(args.shard).removesuffix(".jsonl")
    metrics_dir = os.path.join(os.path.dirname(args.shard), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = open(os.path.join(metrics_dir, f"worker-{shard_name}.jsonl"), "a")

    def log_metrics(entry):
        metrics_file.write(json.dumps(entry) + "\n")
        metrics_file.flush()

    done = load_done(output)
    with open(args.shard) as f:
        pending = [r for r in map(json.loads, f) if r["custom_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{shard_name}: {len(done):,} done, {len(pending):,} to generate", flush=True)
    if not pending:
        return

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    sampler = GpuSampler(gpu)
    log_metrics({"type": "config", "time": time.time(), "shard": shard_name, "gpu": gpu,
                 "model": args.model, "vllm": vllm_version, "chunk_size": args.chunk_size,
                 "max_model_len": args.max_model_len, "limit": args.limit,
                 "gpu_memory_utilization": args.gpu_memory_utilization,
                 "pending": len(pending), "already_done": len(done)})
    load_start = time.time()
    # disable_log_stats=False: per-request timestamps for the batch metrics.
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
              disable_log_stats=False)
    tokenizer = llm.get_tokenizer()
    log_metrics({"type": "load", "load_s": round(time.time() - load_start, 1)})
    work_start = time.time()

    with open(output, "a") as out:
        for index, start in enumerate(range(0, len(pending), args.chunk_size)):
            chunk = pending[start : start + args.chunk_size]
            chunk_start = time.time()
            lines, batch = [], []
            for request in chunk:
                body = request["body"]
                template_kwargs = body.get("chat_template_kwargs") or {}
                prompt = tokenizer.apply_chat_template(
                    body["messages"], add_generation_prompt=True, tokenize=False,
                    **template_kwargs,
                )
                prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                max_tokens = body["max_completion_tokens"]
                # One over-long prompt would abort the whole LLM.chat call: fail it here.
                if prompt_len + max_tokens > args.max_model_len:
                    lines.append(record(request["custom_id"], 400, error=(
                        f"prompt {prompt_len} + max_tokens {max_tokens} tokens exceeds "
                        f"max_model_len {args.max_model_len}")))
                    continue
                params = SamplingParams(
                    temperature=body.get("temperature", 0.0), top_p=body.get("top_p", 1.0),
                    top_k=body.get("top_k", -1), max_tokens=max_tokens,
                )
                batch.append((request, template_kwargs, params))
            prep_s = time.time() - chunk_start

            # LLM.chat takes one chat_template_kwargs per call; requests normally share one.
            groups: dict[str, list] = {}
            for item in batch:
                groups.setdefault(json.dumps(item[1], sort_keys=True), []).append(item)
            all_results = []
            generate_start = time.time()
            for items in groups.values():
                results = llm.chat(
                    [request["body"]["messages"] for request, _, _ in items],
                    [params for _, _, params in items],
                    chat_template_kwargs=items[0][1], use_tqdm=False,
                )
                all_results += results
                for (request, _, _), result in zip(items, results):
                    completion = result.outputs[0]
                    body = {"object": "chat.completion", "model": args.model, "choices": [{
                        "index": 0, "finish_reason": completion.finish_reason,
                        "message": {"role": "assistant", "content": completion.text},
                    }]}
                    lines.append(record(request["custom_id"], 200, body=body))
            generate_end = time.time()
            out.writelines(lines)
            out.flush()
            os.fsync(out.fileno())

            out_lens = [len(r.outputs[0].token_ids) for r in all_results]
            prompt_tokens = sum(len(r.prompt_token_ids or []) for r in all_results)
            generate_s = max(generate_end - generate_start, 1e-9)
            finish = {}
            for r in all_results:
                finish[r.outputs[0].finish_reason] = finish.get(r.outputs[0].finish_reason, 0) + 1
            entry = {
                "type": "chunk", "chunk": index, "requests": len(chunk),
                "rejected_too_long": len(chunk) - len(all_results),
                "prep_s": round(prep_s, 2), "generate_s": round(generate_s, 2),
                "write_s": round(time.time() - generate_end, 2),
                "prompt_tokens": prompt_tokens,
                "cached_prompt_tokens": sum(r.num_cached_tokens or 0 for r in all_results),
                "generated_tokens": sum(out_lens),
                "gen_tok_per_s": round(sum(out_lens) / generate_s, 1),
                "req_per_s": round(len(all_results) / generate_s, 2),
                "output_tokens": percentiles(out_lens), "finish_reasons": finish,
                **batch_metrics(all_results),
                **sampler.window(generate_start, generate_end),
            }
            log_metrics(entry)
            print(f"{shard_name} chunk {index}: {entry['requests']} req, "
                  f"{entry['gen_tok_per_s']:,.0f} gen tok/s, "
                  f"conc {entry.get('concurrency_mean', '?')}/{entry.get('concurrency_peak', '?')}, "
                  f"tail {entry.get('tail_frac', '?')}, gpu {entry.get('util_mean', '?')}%, "
                  f"preempt {entry.get('preemptions', '?')}", flush=True)

    sampler.stop()
    log_metrics({"type": "done", "time": time.time(), "work_s": round(time.time() - work_start, 1)})
    metrics_file.close()


def summarize(metrics_dir: str, wall_s: float) -> dict:
    workers = {}
    for path in sorted(glob.glob(os.path.join(metrics_dir, "worker-*.jsonl"))):
        entries = [json.loads(line) for line in open(path)]
        runs = [i for i, e in enumerate(entries) if e["type"] == "config"]
        latest = entries[runs[-1]:] if runs else entries    # the most recent run only
        chunks = [e for e in latest if e["type"] == "chunk"]
        if not chunks:
            continue
        config = latest[0]
        generate_s = sum(c["generate_s"] for c in chunks)
        generated = sum(c["generated_tokens"] for c in chunks)
        workers[config["shard"]] = {
            "gpu": config["gpu"], "requests": sum(c["requests"] for c in chunks),
            "load_s": next((e["load_s"] for e in latest if e["type"] == "load"), None),
            "work_s": next((e["work_s"] for e in latest if e["type"] == "done"), None),
            "generated_tokens": generated,
            "gen_tok_per_s": round(generated / max(generate_s, 1e-9), 1),
            "overhead_frac": round(sum(c["prep_s"] + c["write_s"] for c in chunks)
                                   / max(sum(c["prep_s"] + c["generate_s"] + c["write_s"]
                                             for c in chunks), 1e-9), 3),
            "concurrency_mean": round(statistics.fmean(c.get("concurrency_mean", 0) for c in chunks), 1),
            "tail_frac_mean": round(statistics.fmean(c.get("tail_frac", 0) for c in chunks), 3),
            "gpu_util_mean": round(statistics.fmean(c.get("util_mean", 0) for c in chunks), 1),
            "preemptions": sum(c.get("preemptions", 0) for c in chunks),
            "length_finishes": sum(c["finish_reasons"].get("length", 0) for c in chunks),
            "config": {k: config[k] for k in ("model", "vllm", "chunk_size", "max_model_len",
                                              "gpu_memory_utilization")},
        }
    work = [w["work_s"] for w in workers.values() if w["work_s"]]
    total_tokens = sum(w["generated_tokens"] for w in workers.values())
    return {
        "wall_s": round(wall_s, 1), "workers": workers,
        "aggregate_gen_tok_per_s": round(total_tokens / max(wall_s, 1e-9), 1),
        "worker_finish_spread_s": round(max(work) - min(work), 1) if work else None,
        "total_requests": sum(w["requests"] for w in workers.values()),
        "total_generated_tokens": total_tokens,
    }


def gpu_count(args) -> int:
    return args.gpus or len(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(","))


def bench(args) -> None:
    """Stream --bench prompts per domain into a fresh run dir (one shard per GPU), then
    generate them like a real run: same request format, workers and metrics."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sample_nemotron import make_request, stream_prompts, write_shards

    run_dir = os.path.join(args.dir or "data/bench", time.strftime("%Y%m%d-%H%M%S"))
    requests = []
    for domain in args.domains.split(","):
        prompts = stream_prompts(args.source, domain, args.bench, args.seed)
        print(f"{domain}: streamed {len(prompts):,} prompts", flush=True)
        requests += [make_request(domain, i, p, args.model) for i, p in enumerate(prompts)]
    write_shards(requests, run_dir, gpu_count(args), args.seed)
    args.dir = run_dir
    launch(args)


def launch(args) -> None:
    shards = sorted(glob.glob(os.path.join(args.dir, "requests-*.jsonl")))
    if not shards:
        raise FileNotFoundError(f"no requests-*.jsonl in {args.dir}")
    gpus = gpu_count(args)
    if len(shards) > gpus:
        print(f"WARNING: {len(shards)} shards on {gpus} GPUs; extra shards run in later waves")
    worker_args = ["--model", args.model, "--chunk-size", str(args.chunk_size),
                   "--max-model-len", str(args.max_model_len),
                   "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                   "--seed", str(args.seed), "--limit", str(args.limit)]
    started = time.time()
    failed = []
    for wave in range(0, len(shards), gpus):
        workers = []
        for gpu, shard in enumerate(shards[wave : wave + gpus]):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            command = [sys.executable, __file__, "--shard", shard, *worker_args]
            workers.append((shard, subprocess.Popen(command, env=env)))
        for shard, worker in workers:
            if worker.wait() != 0:
                failed.append(shard)
    metrics_dir = os.path.join(args.dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)       # workers may all fail before creating it
    summary = summarize(metrics_dir, time.time() - started)
    with open(os.path.join(metrics_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(f"workers failed for {failed}; rerun to resume")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--dir", help="Holds requests-*.jsonl; starts one worker per GPU. "
                                 "With --bench: parent of the run dir (default data/bench).")
    p.add_argument("--shard", help="Worker mode: generate one requests-*.jsonl shard.")
    p.add_argument("--gpus", type=int, default=None,
                   help="Default: count of CUDA_VISIBLE_DEVICES, else 4.")
    p.add_argument("--chunk-size", type=int, default=8192,
                   help="Requests per LLM.chat call; also the resume granularity.")
    p.add_argument("--bench", type=int, default=0,
                   help="Quick benchmark: stream this many prompts per domain into a fresh "
                        "run dir and generate them (sample_nemotron.py not needed).")
    p.add_argument("--source", default="nvidia/Nemotron-Post-Training-Dataset-v2")
    p.add_argument("--domains", default="chat,math,code")
    p.add_argument("--limit", type=int, default=0,
                   help="Only the first N pending requests per shard.")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.bench:
        if args.shard:
            p.error("--bench starts its own workers; drop --shard")
        bench(args)
    elif (args.dir is None) == (args.shard is None):
        p.error("pass exactly one of --dir (launch) / --shard (worker), or --bench N")
    else:
        run_worker(args) if args.shard else launch(args)


if __name__ == "__main__":
    main()
