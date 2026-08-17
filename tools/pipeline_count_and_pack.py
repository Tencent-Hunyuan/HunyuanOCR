#!/usr/bin/env python3
"""
Integrated pipeline: JSONL -> token counting -> packing.

Steps:
  1. Read JSONL file paths from a txt list (one path per line).
  2. For each JSONL file, read every line, extract question/answer/image,
     calculate token count using HunYuanVLProcessor, and save a new JSONL
     with an added 'num_tokens' field to the output directory.
  3. Collect all counted samples, shuffle, pack into bins of --pack-length
     tokens, and save as a single JSONL file (one packed group per line).

Uses multiprocessing (32 processes) + threading (8 threads per process)
for token counting.
"""

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool
from pathlib import Path

import binpacking
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# from hunyuan_vl.processing_hunyuan_vl import HunYuanVLProcessor
# from hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor

# from transformers import HunYuanVLProcessor
# from transformers.models.hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor


# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Data arguments for image processor
# ---------------------------------------------------------------------------
class DataArguments:
    def __init__(self, max_pixels=2048 * 2048, min_pixels=512 * 512):
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels


# ---------------------------------------------------------------------------
# Token calculation
# ---------------------------------------------------------------------------
def calculate_tokens(image_path: str, question: str, answer: str, system_prompt: str, hunyuan_processor) -> int:
    """
    Calculate the total number of tokens for a single sample
    (text + image tokens).
    """
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": answer},
        ]

        text = hunyuan_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        if not os.path.exists(image_path):
            # Fallback: text-only token count
            text_tokens = hunyuan_processor.tokenizer(text, return_tensors="pt")["input_ids"]
            return len(text_tokens[0])

        image = Image.open(image_path).convert("RGB")
        inputs = hunyuan_processor(text=[text], images=image, padding=False, return_tensors="pt")
        return len(inputs["input_ids"][0])

    except Exception as e:
        print(f"Error calculating tokens for {image_path}: {e}")
        return 0


# ---------------------------------------------------------------------------
# Per-file worker (runs in a subprocess)
# ---------------------------------------------------------------------------
def count_tokens_for_file(
    jsonl_path: str,
    output_dir: str,
    model_path: str,
    max_pixels: int,
    min_pixels: int,
    threads_per_process: int,
    system_prompt: str,
    result_queue: Queue,
    allow_empty_answer: bool = False,
):
    """
    Worker: load one JSONL file, compute num_tokens for each sample,
    save a new JSONL with num_tokens to output_dir, and report back.
    """
    try:
        src = Path(jsonl_path)
        out_path = Path(output_dir) / (src.stem + "_count.jsonl")

        # If count file already exists, skip computation
        if out_path.exists():
            print(f"[{src.name}] Found existing count file, skipping: {out_path}")
            result_queue.put((str(src), str(out_path), True))
            return

        print(f"[{src.name}] Loading JSONL ...")
        raw_lines = []
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_lines.append(line)
        print(f"[{src.name}] {len(raw_lines)} samples to process")

        """
        # Initialize processor inside subprocess (avoid pickling)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
        base_image_processor = HunYuanVLImageProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        processor = HunYuanVLProcessor(
            image_processor=base_image_processor, tokenizer=tokenizer
        )
        """

        # Initialize processor inside subprocess (avoid pickling).
        # Use AutoProcessor so that special tokens (image/video tokens) are
        # properly registered — manual construction skips that step and
        # crashes with `TokenizersBackend has no attribute video_token`.
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )

        def _process_line(line_str):
            """Parse one JSONL line, calculate tokens, return result dict."""
            try:
                data = json.loads(line_str)

                # Extract question/answer from conv
                if not data.get("conv") or len(data["conv"]) == 0:
                    return None
                conv = data["conv"][0]
                question = conv.get("question", "")
                answer = conv.get("answer", "")
                if not question:
                    return None
                # An empty answer is a deliberate label for a blank page — the
                # model must learn to emit nothing. Dropping it teaches the
                # opposite, so it is only skipped when the caller asks.
                if not answer and not allow_empty_answer:
                    return None

                # Extract image path
                img_path = data.get("img_path_sh") or data.get("img_path_cq")
                if not img_path:
                    return None

                # Calculate tokens
                num_tokens = calculate_tokens(img_path, question, answer, system_prompt, processor)

                return {
                    "image": img_path,
                    "question": question,
                    "answer": answer,
                    "num_tokens": num_tokens,
                }
            except Exception as e:
                print(f"[{src.name}] Error processing line: {e}")
                return None

        # Use ThreadPool for image-level parallelism
        results = []
        pool = ThreadPool(threads_per_process)
        for result in tqdm(
            pool.imap_unordered(_process_line, raw_lines),
            total=len(raw_lines),
            desc=src.name,
        ):
            if result is not None:
                results.append(result)
        pool.close()
        pool.join()

        # Save counted JSONL
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in results)

        print(f"[{src.name}] Saved: {out_path} ({len(results)} samples)")
        result_queue.put((str(src), str(out_path), True))

    except Exception as e:
        print(f"[{jsonl_path}] Fatal error: {e}")
        import traceback

        traceback.print_exc()
        result_queue.put((str(jsonl_path), None, False))


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
def read_input_list(txt_path: Path) -> list[Path]:
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Integrated pipeline: JSONL -> token counting -> packing")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, nargs="+", help="One or more JSONL file paths")
    input_group.add_argument(
        "--input-list", type=str, help="Path to a txt file containing JSONL file paths (one per line)"
    )

    parser.add_argument(
        "--model-path", type=str, default="./HunyuanOCR", help="Path to HunyuanVL model (default: ./HunyuanOCR)"
    )
    parser.add_argument("--count-output-dir", type=str, required=True, help="Output directory for counted JSONL files")
    parser.add_argument("--pack-output", type=str, required=True, help="Output path for the packed JSONL file")
    parser.add_argument("--pack-length", type=int, default=20480, help="Max tokens per packed bin (default: 20480)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for binpacking (default: 1024)")
    parser.add_argument("--max-pixels", type=int, default=2048 * 2048, help="Max image pixels (default: 2048*2048)")
    parser.add_argument("--min-pixels", type=int, default=512 * 512, help="Min image pixels (default: 512*512)")
    parser.add_argument("--num-processes", type=int, default=32, help="Number of parallel processes (default: 32)")
    parser.add_argument(
        "--threads-per-process", type=int, default=8, help="Threads per process for image processing (default: 8)"
    )
    parser.add_argument("--system-prompt", type=str, default="", help="System prompt to use (default: empty)")
    parser.add_argument(
        "--allow-empty-answer",
        action="store_true",
        help="Keep samples whose answer is empty (blank-page labels). Off by default.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Collect JSONL paths
    # ------------------------------------------------------------------
    if args.input_list:
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

    count_output_dir = Path(args.count_output_dir)
    count_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"JSONL files to process: {len(valid_paths)}")
    print(f"Model path: {args.model_path}")
    print(f"Count output dir: {count_output_dir}")
    print(f"Pack output: {args.pack_output}")
    print(f"Pack length: {args.pack_length}")
    print(f"Processes: {args.num_processes}, Threads/process: {args.threads_per_process}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2. Count tokens (multi-process, batched by num_processes)
    # ------------------------------------------------------------------
    print("\n[Phase 1] Token counting ...")
    start_time = time.time()

    count_files = {}  # src_path -> count_path
    num_processes = min(args.num_processes, len(valid_paths))

    # Process files in batches of num_processes
    for batch_start in range(0, len(valid_paths), num_processes):
        batch_paths = valid_paths[batch_start : batch_start + num_processes]
        result_queue = Queue()
        processes = []

        for jsonl_path in batch_paths:
            p = Process(
                target=count_tokens_for_file,
                args=(
                    jsonl_path,
                    str(count_output_dir),
                    args.model_path,
                    args.max_pixels,
                    args.min_pixels,
                    args.threads_per_process,
                    args.system_prompt,
                    result_queue,
                    args.allow_empty_answer,
                ),
            )
            p.start()
            processes.append(p)
            print(f"  Started worker for {Path(jsonl_path).name} (PID: {p.pid})")

        # Collect results for this batch
        for _ in range(len(processes)):
            src_path, count_path, success = result_queue.get()
            if success and count_path:
                count_files[src_path] = count_path
                print(f"  Done: {Path(src_path).name} -> {Path(count_path).name}")
            else:
                print(f"  Failed: {Path(src_path).name}")

        for p in processes:
            p.join()

        print(f"  Batch [{batch_start + 1}-{batch_start + len(batch_paths)}/{len(valid_paths)}] done")

    elapsed = time.time() - start_time
    print(f"\nToken counting complete in {elapsed:.1f}s: {len(count_files)}/{len(valid_paths)} files succeeded")

    # ------------------------------------------------------------------
    # 3. Collect all samples, shuffle
    # ------------------------------------------------------------------
    print("\n[Phase 2] Collecting all counted samples ...")
    all_samples = []
    for src_path in valid_paths:
        if src_path not in count_files:
            continue
        count_path = count_files[src_path]
        with open(count_path, "r", encoding="utf-8") as f:
            file_count = 0
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if item.get("num_tokens", 0) > 0:
                        all_samples.append(item)
                        file_count += 1
        print(f"  {Path(src_path).name}: {file_count} samples")

    print(f"Total samples collected: {len(all_samples)}")

    random.seed(args.seed)
    random.shuffle(all_samples)
    print(f"Shuffled with seed={args.seed}")

    # ------------------------------------------------------------------
    # 4. Pack
    # ------------------------------------------------------------------
    print(f"\n[Phase 3] Packing with pack_length={args.pack_length} ...")
    pack_start = time.time()
    packed = pack_data(all_samples, args.pack_length, args.batch_size)
    pack_elapsed = time.time() - pack_start

    total_items = sum(len(group) for group in packed)
    avg_per_group = total_items / len(packed) if packed else 0
    print(f"Packing done in {pack_elapsed:.2f}s")
    print(f"  Packed groups: {len(packed)}")
    print(f"  Total items in groups: {total_items}")
    print(f"  Avg items per group: {avg_per_group:.1f}")

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
    print(f"   - Counted JSONL dir: {count_output_dir}")
    print(f"   - Packed JSONL: {pack_output_path}")
    print(f"   - {len(packed)} groups, {total_items} total samples")


if __name__ == "__main__":
    main()

# Usage examples:
#
# Test with a single file:
#   python tools/pipeline_count_and_pack.py \
#       --input /path/to/single.jsonl \
#       --model-path ./HunyuanOCR \
#       --count-output-dir ./all_parsing_jsonl_count \
#       --pack-output ./all_parsing_packed.jsonl \
#       --num-processes 1 --threads-per-process 8
#
# Full run with all files:
#   python tools/pipeline_count_and_pack.py \
#       --input-list /path/to/all_parsing.txt \
#       --model-path ./HunyuanOCR \
#       --count-output-dir ./all_parsing_jsonl_count \
#       --pack-output ./all_parsing_packed.jsonl \
#       --num-processes 32 --threads-per-process 8 \
#       --pack-length 20480
