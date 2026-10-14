"""Step 1/3 of the paper's data pipeline (Appendix A): sample prompts from
Nemotron-Post-Training-Dataset-v2 at chat:math:code = 1:1:1 and write them as vLLM batch
requests, so the frozen AR model regenerates every response. Each domain contributes as
many prompts as the smallest one has usable (code, ~175K), unless --per-domain caps it.

    python scripts/data/sample_nemotron.py --model Qwen/Qwen3.5-4B --out data/nemotron
    scripts/data/vllm-env/.venv/bin/python scripts/data/generate_vllm.py \
        --model Qwen/Qwen3.5-4B --dir data/nemotron              # step 2: vLLM, one shard per GPU
    python scripts/data/merge_responses.py --dir data/nemotron  # step 3: -> messages JSONL

Prompts are single-turn (leading system messages + the first user turn), so every
supervised assistant token is the model's own output. The dataset is gated: accept its
terms on the Hub and log in (`hf auth login`) first.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import datasets

# Nemotron's reasoning switches mean nothing to other models' chat templates.
REASONING_TOGGLES = {"detailed thinking on", "detailed thinking off", "/think", "/no_think"}


def load_domain(source: str, domain: str):
    """A Hub dataset id, or a local directory laid out like the repo (data/<domain>-*.parquet).
    data_files restricts the download to this split: load_dataset(repo, split=...) alone
    prepares every split, including ~38 GB of multilingual data."""
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "data", f"{domain}-*.parquet")))
        if not files:
            raise FileNotFoundError(f"no data/{domain}-*.parquet under {source}")
        return datasets.load_dataset("parquet", data_files=files, split="train")
    return datasets.load_dataset(
        source, data_files={domain: f"data/{domain}-*.parquet"}, split=domain
    )


def stream_prompts(source: str, domain: str, count: int, seed: int,
                   buffer_size: int = 2000) -> list[list[dict]]:
    """`count` usable prompts from a streamed, buffer-shuffled split: reads only the first
    row groups instead of downloading the split (for quick benchmarks)."""
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "data", f"{domain}-*.parquet")))
        stream = datasets.load_dataset("parquet", data_files=files, split="train", streaming=True)
    else:
        stream = datasets.load_dataset(source, data_files={domain: f"data/{domain}-*.parquet"},
                                       split=domain, streaming=True)
    prompts = []
    for row in stream.shuffle(seed=seed, buffer_size=buffer_size):
        prompt = single_turn_prompt(row["messages"])
        if prompt:
            prompts.append(prompt)
            if len(prompts) == count:
                break
    return prompts


def make_request(domain: str, index: int, prompt: list[dict], model: str,
                 max_tokens: int = 4096, temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = -1, enable_thinking: bool = False) -> dict:
    """An OpenAI-style chat request (vLLM batch format); generate_vllm.py reads the body.
    Defaults: greedy, matching the greedy decoding the drafter is evaluated on, no thinking."""
    body = {
        "model": model, "messages": prompt, "max_completion_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p, "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    return {"custom_id": f"{domain}-{index:07d}", "method": "POST",
            "url": "/v1/chat/completions", "body": body}


def write_shards(requests: list[dict], out: str, num_shards: int, seed: int) -> None:
    """Shuffle across domains so every shard (GPU) gets the same mix and length profile."""
    random.Random(seed).shuffle(requests)
    os.makedirs(out, exist_ok=True)
    for shard in range(num_shards):
        path = os.path.join(out, f"requests-{shard:05d}-of-{num_shards:05d}.jsonl")
        with open(path, "w") as f:
            for request in requests[shard :: num_shards]:
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"wrote {len(requests):,} requests to {num_shards} shards in {out}")


def single_turn_prompt(messages) -> list[dict] | None:
    """Leading system messages (minus reasoning toggles) + the first user turn."""
    prompt = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role == "system":
            if content and content.strip().lower() not in REASONING_TOGGLES:
                prompt.append({"role": "system", "content": content})
        elif role == "user":
            if not content or not content.strip():
                return None
            prompt.append({"role": "user", "content": content})
            return prompt
        else:
            return None                    # conversation does not open with a user turn
    return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="nvidia/Nemotron-Post-Training-Dataset-v2",
                   help="Hub dataset id, or a local dir with data/<domain>-*.parquet.")
    p.add_argument("--domains", default="chat,math,code")
    p.add_argument("--per-domain", type=int, default=None,
                   help="Cap on prompts per domain. Default: the smallest domain's count.")
    p.add_argument("--model", required=True,
                   help="The frozen AR base, as vLLM will serve it (e.g. Qwen/Qwen3.5-4B).")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy, matching the greedy decoding the drafter is evaluated on.")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--enable-thinking", action="store_true",
                   help="Let the model think (Qwen chat_template_kwargs). Off by default.")
    p.add_argument("--num-shards", type=int, default=4, help="One per GPU for generate_vllm.py.")
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    # Usable prompts per domain, in a seeded random order; every domain then contributes
    # the same number -- the smallest domain's count -- so the mix is exactly 1:1:1.
    def extract(batch):
        found = (single_turn_prompt(messages) for messages in batch["messages"])
        return {"prompt": [json.dumps(p, ensure_ascii=False) if p else "" for p in found]}

    prompts = {}
    for domain in args.domains.split(","):
        dataset = load_domain(args.source, domain).select_columns(["messages"])
        # Arrow-side and parallel: only the prompts (not the long responses) reach Python.
        dataset = dataset.shuffle(seed=args.seed).map(
            extract, batched=True, remove_columns=["messages"], num_proc=args.num_proc,
            desc=f"Extracting {domain} prompts",
        )
        prompts[domain] = [json.loads(p) for p in dataset["prompt"] if p]
        print(f"{domain}: {len(prompts[domain]):,} usable prompts of {len(dataset):,} rows")
    per_domain = min(len(found) for found in prompts.values())
    if args.per_domain is not None:
        per_domain = min(per_domain, args.per_domain)
    print(f"taking {per_domain:,} prompts per domain")

    requests = [
        make_request(domain, index, prompt, args.model, args.max_tokens, args.temperature,
                     args.top_p, args.top_k, args.enable_thinking)
        for domain, found in prompts.items() for index, prompt in enumerate(found[:per_domain])
    ]
    write_shards(requests, args.out, args.num_shards, args.seed)


if __name__ == "__main__":
    main()
