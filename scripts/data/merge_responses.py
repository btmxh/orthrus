"""Step 3/3 of the paper's data pipeline: join vLLM batch outputs with their requests into
`messages` JSONL for scripts/build_packed_cache.py.

    python scripts/data/merge_responses.py --dir data/nemotron
    python scripts/build_packed_cache.py --dataset-path data/nemotron/regenerated.jsonl \
        --model-dir pretrained-models/orthrus-qwen3_5-4b-init --out data/processed/orthrus-packed-2048-asst

Failed requests are dropped, and so are responses cut off by max_tokens (unless
--keep-truncated): the chat template would close them with a turn end the model never
produced.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os


def read_jsonl(paths):
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", required=True, help="Holds requests-*.jsonl and responses-*.jsonl.")
    p.add_argument("--out", default=None, help="Default: <dir>/regenerated.jsonl")
    p.add_argument("--keep-truncated", action="store_true")
    args = p.parse_args()
    out_path = args.out or os.path.join(args.dir, "regenerated.jsonl")

    request_files = sorted(glob.glob(os.path.join(args.dir, "requests-*.jsonl")))
    response_files = sorted(glob.glob(os.path.join(args.dir, "responses-*.jsonl")))
    if not request_files or not response_files:
        raise FileNotFoundError(f"need requests-*.jsonl and responses-*.jsonl in {args.dir}")
    prompts = {r["custom_id"]: r["body"]["messages"] for r in read_jsonl(request_files)}

    stats = collections.defaultdict(collections.Counter)
    seen = set()
    with open(out_path, "w") as out:
        for record in read_jsonl(response_files):
            custom_id = record["custom_id"]
            domain = custom_id.rsplit("-", 1)[0]
            seen.add(custom_id)
            response = record.get("response") or {}
            body = response.get("body")
            if record.get("error") or response.get("status_code") != 200 or not body:
                stats[domain]["failed"] += 1
                continue
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length" and not args.keep_truncated:
                stats[domain]["truncated"] += 1
                continue
            message = choice["message"]
            assistant = {"role": "assistant", "content": message.get("content") or ""}
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                assistant["reasoning_content"] = reasoning
            out.write(json.dumps({"messages": prompts[custom_id] + [assistant], "domain": domain},
                                 ensure_ascii=False) + "\n")
            stats[domain]["kept"] += 1

    missing = len(prompts.keys() - seen)
    for domain, counts in sorted(stats.items()):
        print(f"{domain}: kept {counts['kept']:,}, truncated {counts['truncated']:,}, "
              f"failed {counts['failed']:,}")
    if missing:
        print(f"WARNING: {missing:,} requests have no response (unfinished shard?)")
    kept = [counts["kept"] for counts in stats.values()]
    if kept and max(kept) > 1.1 * min(kept):
        print("WARNING: domains differ by >10% after filtering; the mix is no longer 1:1:1.")
    print(f"wrote {sum(kept):,} conversations to {out_path}")


if __name__ == "__main__":
    main()
