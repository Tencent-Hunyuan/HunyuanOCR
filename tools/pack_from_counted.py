#!/usr/bin/env python3
"""
Pack pre-counted JSONL files into training bins.

This script assumes all input JSONL files already contain the 'num_tokens' field
(i.e., token counting has already been done). It directly performs:
  1. Read all *_count.jsonl files from an input directory (or specified files).
  2. Filter out samples with num_tokens <= 0 or num_tokens > pack_length.
  3. Shuffle all samples.
  4. Pack into bins of --pack-length tokens using binpacking.
  5. Save as a single JSONL file (one packed group per line).

Usage examples:

  # Pack all counted JSONL files from a directory:
  python tools/pack_from_counted.py \
      --input-dir /path/to/output_jsonl_count_std \
      --pack-output ./all_parsing_packed.jsonl \
      --pack-length 20480

  # Pack specific files:
  python tools/pack_from_counted.py \
      --input /path/to/file1_count.jsonl /path/to/file2_count.jsonl \
      --pack-output ./all_parsing_packed.jsonl \
      --pack-length 20480

  # Use a txt list of file paths:
  python tools/pack_from_counted.py \
      --input-list /path/to/file_list.txt \
      --pack-output ./all_parsing_packed.jsonl \
      --pack-length 20480
"""

import argparse
import json
import random
import time
from pathlib import Path

import binpacking


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
def pack_data(data_list: list, pack_length: int, batch_size: int = 1024) -> list:
    """
    Pack samples into bins of at most pack_length tokens.
    Process in batches to keep memory bounded.
    """
    all_packed = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i : i + batch_size]
        lengths = [d["num_tokens"] for d in batch]
        grouped = binpacking.to_constant_volume(list(enumerate(lengths)), pack_length, weight_pos=1)
        for group in grouped:
            group_data = []
            for idx, _ in group:
                item = batch[idx]
                group_data.append(
                    {
                        "image": item["image"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "num_tokens": item["num_tokens"],
                    }
                )
            all_packed.append(group_data)
    return all_packed


# ---------------------------------------------------------------------------
# Read input list
# ---------------------------------------------------------------------------
def read_input_list(txt_path: Path) -> list:
    """Read a txt file containing one JSONL file path per line.
    Empty lines and lines starting with '#' are ignored."""
    paths = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line))
    return paths


# ---------------------------------------------------------------------------
# Collect samples from JSONL files
# ---------------------------------------------------------------------------
def collect_samples(jsonl_paths: list, pack_length: int) -> list:
    """Read all JSONL files and collect valid samples."""
    all_samples = []
    skipped_zero = 0
    skipped_too_long = 0

    for jsonl_path in jsonl_paths:
        file_count = 0
        file_skipped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON decode error in {jsonl_path}: {e}")
                    continue

                num_tokens = item.get("num_tokens", 0)
                if num_tokens <= 0:
                    skipped_zero += 1
                    file_skipped += 1
                    continue
                if num_tokens > pack_length:
                    skipped_too_long += 1
                    file_skipped += 1
                    continue

                all_samples.append(item)
                file_count += 1

        print(
            f"  {Path(jsonl_path).name}: {file_count} samples loaded"
            + (f" ({file_skipped} skipped)" if file_skipped > 0 else "")
        )

    if skipped_zero > 0:
        print(f"\n  Total skipped (num_tokens <= 0): {skipped_zero}")
    if skipped_too_long > 0:
        print(f"  Total skipped (num_tokens > {pack_length}): {skipped_too_long}")

    return all_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pack pre-counted JSONL files into training bins")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, nargs="+", help="One or more pre-counted JSONL file paths")
    input_group.add_argument("--input-dir", type=str, help="Directory containing *_count.jsonl files")
    input_group.add_argument(
        "--input-list", type=str, help="Path to a txt file containing JSONL file paths (one per line)"
    )

    parser.add_argument("--pack-output", type=str, required=True, help="Output path for the packed JSONL file")
    parser.add_argument("--pack-length", type=int, default=20480, help="Max tokens per packed bin (default: 20480)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for binpacking (default: 1024)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    parser.add_argument(
        "--suffix",
        type=str,
        default="_count.jsonl",
        help="File suffix filter when using --input-dir (default: '_count.jsonl')",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Collect JSONL paths
    # ------------------------------------------------------------------
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"Error: {args.input_dir} is not a valid directory.")
            return
        jsonl_paths = sorted(str(p) for p in input_dir.iterdir() if p.is_file() and p.name.endswith(args.suffix))
        print(f"Found {len(jsonl_paths)} files matching '*{args.suffix}' in {input_dir}")
    elif args.input_list:
        jsonl_paths = [str(p) for p in read_input_list(Path(args.input_list))]
    else:
        jsonl_paths = args.input

    # Validate paths
    valid_paths = []
    for p in jsonl_paths:
        if Path(p).exists():
            valid_paths.append(p)
        else:
            print(f"Warning: File not found, skipping: {p}")
    if not valid_paths:
        print("Error: No valid JSONL files found.")
        return

    print(f"\nJSONL files to pack: {len(valid_paths)}")
    print(f"Pack output: {args.pack_output}")
    print(f"Pack length: {args.pack_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2. Collect all samples
    # ------------------------------------------------------------------
    print("\n[Phase 1] Collecting all counted samples ...")
    start_time = time.time()
    all_samples = collect_samples(valid_paths, args.pack_length)

    print(f"\nTotal samples collected: {len(all_samples)}")

    if not all_samples:
        print("Error: No valid samples found. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Shuffle
    # ------------------------------------------------------------------
    random.seed(args.seed)
    random.shuffle(all_samples)
    print(f"Shuffled with seed={args.seed}")

    # ------------------------------------------------------------------
    # 4. Pack
    # ------------------------------------------------------------------
    print(f"\n[Phase 2] Packing with pack_length={args.pack_length} ...")
    pack_start = time.time()
    packed = pack_data(all_samples, args.pack_length, args.batch_size)
    pack_elapsed = time.time() - pack_start

    total_items = sum(len(group) for group in packed)
    avg_per_group = total_items / len(packed) if packed else 0

    # Token statistics
    total_tokens = sum(item["num_tokens"] for group in packed for item in group)
    avg_tokens_per_group = total_tokens / len(packed) if packed else 0

    print(f"Packing done in {pack_elapsed:.2f}s")
    print(f"  Packed groups: {len(packed)}")
    print(f"  Total items in groups: {total_items}")
    print(f"  Avg items per group: {avg_per_group:.1f}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg tokens per group: {avg_tokens_per_group:.0f} / {args.pack_length}")
    print(f"  Token utilization: {avg_tokens_per_group / args.pack_length * 100:.1f}%")

    # ------------------------------------------------------------------
    # 5. Save packed result as JSONL (one group per line)
    # ------------------------------------------------------------------
    pack_output_path = Path(args.pack_output)
    pack_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pack_output_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(group, ensure_ascii=False) + "\n" for group in packed)

    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total_elapsed:.1f}s!")
    print(f"   - Packed JSONL: {pack_output_path}")
    print(f"   - {len(packed)} groups, {total_items} total samples")
    print(f"   - File size: {pack_output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
