"""Build the packed training cache: `messages` rows -> `input_ids` + `assistant_mask`.

    python scripts/build_packed_cache.py --dataset-path data/sft.jsonl \
      --out data/processed/orthrus-packed-2048-asst
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, ".")

from transformers import AutoTokenizer

from src.utils.data_utils import build_packed_cache, load_packed_dataset

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="JSONL/parquet file, save_to_disk dir, or hub name. Needs a "
                             "`messages` column.")
    parser.add_argument("--out", type=str, required=True)
    # The tokenizer MUST be the one training uses: `assistant_token_mask` walks
    # <|im_start|>assistant ... <|im_end|> spans, so a mismatch silently mislabels every row.
    parser.add_argument("--model-dir", type=str,
                        default="pretrained-models/orthrus-qwen3_5-4b-init",
                        help="Init checkpoint to take the tokenizer from.")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Packed row length; the paper uses L=2048.")
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap input conversations. Use a few thousand for a smoke build.")
    parser.add_argument("--group-batch-size", type=int, default=4096,
                        help="Conversations concatenated before slicing into rows.")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Keep source order. Shuffling is on by default so packed rows do "
                             "not group same-source conversations together.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --out.")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.out):
        if not args.force:
            raise ValueError(f"{args.out} already exists; pass --force to overwrite.")
        print(f"--force: removing {args.out}")
        shutil.rmtree(args.out)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"tokenizer={args.model_dir} (len={len(tokenizer)}) | seq_len={args.seq_len} | "
          f"num_proc={args.num_proc}")
    print(f"source={args.dataset_path}")

    build_packed_cache(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        save_path=args.out,
        seq_len=args.seq_len,
        num_proc=args.num_proc,
        max_samples=args.max_samples,
        group_batch_size=args.group_batch_size,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )

    dataset = load_packed_dataset(args.out, args.seq_len)
    row = dataset[0]
    if len(row["assistant_mask"]) != len(row["input_ids"]):
        raise ValueError("assistant_mask and input_ids differ in length -- packing is broken.")

    sample = dataset.select(range(min(200, len(dataset))))
    assistant_frac = sum(sum(r) for r in sample["assistant_mask"]) / (
        len(sample) * args.seq_len
    )
    print(f"\nrows={len(dataset):,} x {args.seq_len} = {len(dataset) * args.seq_len:,} tokens")
    print(f"columns={dataset.column_names}")
    print(f"assistant tokens ~{100.0 * assistant_frac:.1f}% (first {len(sample)} rows)")
    if assistant_frac < 0.05:
        raise ValueError(
            f"Only {100.0 * assistant_frac:.2f}% of tokens are marked assistant. The tokenizer "
            f"almost certainly does not match the data's chat template, which would train the "
            f"diffusion view on prompt tokens."
        )
    print(f"\nOK -- train with --packed-cache-path {args.out} --seq-len {args.seq_len}")


if __name__ == "__main__":
    main()