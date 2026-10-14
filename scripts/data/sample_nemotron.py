"""Step 1/3 of the paper's data pipeline (Appendix A): sample prompts from
Nemotron-Post-Training-Dataset-v2 at chat:math:code = 1:1:1 and write them as vLLM batch
requests, so the frozen AR model regenerates every response.

    python scripts/data/sample_nemotron.py --model Qwen/Qwen3.5-4B --out data/nemotron
    sbatch scripts/data/regenerate.sbatch                      # step 2: vLLM, one shard per GPU
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
    """A Hub dataset id, or a local directory laid out like the repo (data/<domain>-*.parquet)."""
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "data", f"{domain}-*.parquet")))
        if not files:
            raise FileNotFoundError(f"no data/{domain}-*.parquet under {source}")
        return datasets.load_dataset("parquet", data_files=files, split="train")
    return datasets.load_dataset(source, split=domain)


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
    p.add_argument("--per-domain", type=int, default=200_000,
                   help="Prompts per domain (paper: 600K examples, 1:1:1).")
    p.add_argument("--model", required=True,
                   help="The frozen AR base, as vLLM will serve it (e.g. Qwen/Qwen3.5-4B).")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy, matching the greedy decoding the drafter is evaluated on.")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--enable-thinking", action="store_true",
                   help="Let the model think (Qwen chat_template_kwargs). Off by default.")
    p.add_argument("--num-shards", type=int, default=4, help="One per GPU for regenerate.sbatch.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    requests = []
    for domain in args.domains.split(","):
        dataset = load_domain(args.source, domain).shuffle(seed=args.seed)
        kept = skipped = 0
        for row in dataset:
            prompt = single_turn_prompt(row["messages"])
            if prompt is None:
                skipped += 1
                continue
            body = {
                "model": args.model,
                "messages": prompt,
                "max_completion_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
            }
            requests.append({"custom_id": f"{domain}-{kept:07d}", "method": "POST",
                             "url": "/v1/chat/completions", "body": body})
            kept += 1
            if kept == args.per_domain:
                break
        print(f"{domain}: {kept:,} prompts ({skipped:,} rows skipped, {len(dataset):,} in split)")
        if kept < args.per_domain:
            print(f"WARNING: {domain} has only {kept:,} usable prompts; domains are unbalanced.")

    # Shuffle across domains so every shard (GPU) gets the same mix and length profile.
    random.Random(args.seed).shuffle(requests)
    os.makedirs(args.out, exist_ok=True)
    for shard in range(args.num_shards):
        path = os.path.join(args.out, f"requests-{shard:05d}-of-{args.num_shards:05d}.jsonl")
        with open(path, "w") as f:
            for request in requests[shard :: args.num_shards]:
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"wrote {len(requests):,} requests to {args.num_shards} shards in {args.out}")


if __name__ == "__main__":
    main()
