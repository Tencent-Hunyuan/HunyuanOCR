"""
Multi-GPU HuggingFace transformers inference for HunyuanOCR-1.5
(AR baseline, no vLLM, no DFlash).

Runs 8 model replicas — one per GPU — with `torch.multiprocessing.spawn`,
splits the input JSONL evenly across them, and generates with sampling
settings **strictly aligned** with the vLLM client
(`inference/vLLM/infer_vllm_client.py`):

    temperature        = 0.0        → do_sample=False  (greedy)
    top_p              = 1.0        (n/a for greedy)
    top_k              = -1         (n/a for greedy)
    repetition_penalty = 1.08
    skip_special_tokens= True
    max_new_tokens     = 32768

Model loading:
    HunYuanVLForConditionalGeneration + AutoProcessor,
    dtype=bfloat16, attn_implementation=eager,
    with the video-token backfill patch for older tokenizer snapshots.

A `StoppingCriteria` implements the same tail-repetition early-stop as the
vLLM streaming client (see `has_tail_repetition`), and `clean_repeated_substrings`
is applied to the final decoded text as the last safety net — matching the
vLLM path 1:1 in observable behavior.

Usage
-----
    python inference/transformers/infer_hf_8gpu.py \
        --model  "your/path/to/HunyuanOCR" \
        --input  "your/path/to/input.jsonl" \
        --output "./results/hf_out" \
        --merge

Input JSONL — each line must contain:
  * an image field  (default key: "image_path"; str or list[str])
  * a prompt field  (default key: "问题"; if missing, --prompt is used)
"""

import argparse
import importlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.output_utils import normalize_doc_parse_markdown
from utils.tasks import (
    DEFAULT_TASK,
    TASK_DESCRIPTIONS,
    TASK_PROMPTS,
    get_prompt,
)

# ============================================================================
# Defaults — aligned with inference/vLLM/infer_vllm_client.py
# ============================================================================
DEFAULT_PROMPT = TASK_PROMPTS[DEFAULT_TASK]


# ============================================================================
# Tail-repetition helpers (mirror inference/vLLM/infer_vllm_client.py)
# ============================================================================
def has_tail_repetition(text: str, min_repeats: int = 8, max_unit: int = 256) -> bool:
    """Detect whether the tail of `text` is stuck in a small repeated unit."""
    n = len(text)
    if n < min_repeats * 2:
        return False
    upper = min(max_unit, n // min_repeats)
    for length in range(1, upper + 1):
        unit = text[-length:]
        if not unit.strip():
            continue
        ok = True
        for k in range(2, min_repeats + 1):
            if text[-length * k : -length * (k - 1)] != unit:
                ok = False
                break
        if ok:
            return True
    return False


def clean_repeated_substrings(text: str, min_repeats: int = 10) -> str:
    """Trim long repeated suffixes as a final safety net."""
    n = len(text)
    if n < 2000:
        return text
    for length in range(2, n // min_repeats + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length
        if count >= min_repeats:
            return text[: n - length * (count - 1)]
    return text


# ============================================================================
# I/O helpers
# ============================================================================
def split_jsonl(input_path: str, num_parts: int) -> list[list[dict]]:
    """Read a JSONL file and split it evenly into `num_parts` chunks."""
    with open(input_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    if not data:
        return [[] for _ in range(num_parts)]
    chunk_size = math.ceil(len(data) / num_parts)
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    while len(chunks) < num_parts:
        chunks.append([])
    return chunks


def unwrap_field(data: dict, key: str, default=None):
    """Fetch data[key], auto-unwrap single-element lists."""
    raw = data.get(key)
    if raw is None:
        return default
    if isinstance(raw, list):
        if len(raw) == 0:
            return default
        return raw[0]
    return raw


# ============================================================================
# Tokenizer / processor loading
# ============================================================================
def _patch_hunyuan_tokenizer_special_tokens(tokenizer) -> None:
    """Backfill missing special-token attributes on older HunyuanOCR tokenizers."""
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    extra_tokens = init_kwargs.get("extra_special_tokens", {}) or {}

    defaults = {
        "image_token": "<｜hy_place▁holder▁no▁102｜>",
        "image_start_token": "<｜hy_place▁holder▁no▁100｜>",
        "image_end_token": "<｜hy_place▁holder▁no▁101｜>",
        "video_token": "<｜hy_place▁holder▁no▁103｜>",
        "video_start_token": "<｜hy_place▁holder▁no▁104｜>",
        "video_end_token": "<｜hy_place▁holder▁no▁105｜>",
    }
    for name, default_value in defaults.items():
        if hasattr(tokenizer, name):
            continue
        value = extra_tokens.get(name)
        if value is None and name == "video_token":
            value = extra_tokens.get("image_token")
        setattr(tokenizer, name, value or default_value)


def _load_processor_with_patch(model_path: str):
    """Fallback processor loader for older tokenizer snapshots missing video_token."""
    from transformers import AutoImageProcessor, AutoTokenizer

    processor_module = importlib.import_module("transformers.models.hunyuan_vl.processing_hunyuan_vl")
    HunYuanVLProcessor = processor_module.HunYuanVLProcessor

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    _patch_hunyuan_tokenizer_special_tokens(tokenizer)
    image_processor = AutoImageProcessor.from_pretrained(model_path)

    video_processor = None
    try:
        from transformers import AutoVideoProcessor

        video_processor = AutoVideoProcessor.from_pretrained(model_path)
    except Exception:
        video_processor = None

    try:
        return HunYuanVLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
        )
    except TypeError:
        return HunYuanVLProcessor(image_processor, tokenizer, video_processor)


def load_model_and_processor(model_path: str, dtype_name: str, attn_implementation: str, device: str):
    import torch
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

    dtype = getattr(torch, dtype_name)
    try:
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    except AttributeError as e:
        if "video_token" not in str(e):
            raise
        print(
            "[warn] AutoProcessor tokenizer lacks video_token; retrying with patched Hunyuan tokenizer.",
            file=sys.stderr,
        )
        processor = _load_processor_with_patch(model_path)

    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )
    model = model.to(device)
    model.eval()
    return model, processor


# ============================================================================
# Tail-repetition StoppingCriteria (equivalent of vLLM streaming early-stop)
# ============================================================================
def build_stopping_criteria(
    processor,
    prompt_len: int,
    min_repeats: int,
    check_start_chars: int = 4000,
    check_step_chars: int = 1000,
    token_probe_step: int = 64,
):
    """Return a StoppingCriteriaList that stops when the tail of the decoded
    text collapses into a small repeated unit.

    Strictly mirrors `inference/vLLM/infer_vllm_client.py::infer_stream`:
      * gate on cumulative *decoded character length* (`acc_len`)
      * first check after `check_start_chars` (default 4000) characters
      * subsequent checks every `check_step_chars` (default 1000) characters
      * `has_tail_repetition` runs on the last 8000 characters of the *full*
        decoded generation, i.e. `"".join(parts)[-8000:]` in the vLLM version.

    Since HF's StoppingCriteria fires per-token, we only *decode* every
    `token_probe_step` tokens (a cheap gate to avoid decoding on every step);
    on each probe we recompute `acc_len` and apply the vLLM-identical policy.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    tokenizer = processor.tokenizer

    class TailRepetitionStop(StoppingCriteria):
        def __init__(self):
            self._next_check_at_chars = check_start_chars
            self._last_probe_tokens = 0
            self._triggered = False

        def __call__(self, input_ids, scores, **kwargs):
            if self._triggered:
                return True
            new_tokens = input_ids[0, prompt_len:]
            n_new = int(new_tokens.numel())
            if n_new - self._last_probe_tokens < token_probe_step:
                return False
            self._last_probe_tokens = n_new

            try:
                # Decode the *entire* generated suffix — this is the direct
                # analog of vLLM's `"".join(parts)`.
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            except Exception:
                return False

            acc_len = len(text)
            if acc_len < self._next_check_at_chars:
                return False
            self._next_check_at_chars = acc_len + check_step_chars

            if has_tail_repetition(text[-8000:], min_repeats=min_repeats):
                self._triggered = True
                return True
            return False

    return StoppingCriteriaList([TailRepetitionStop()])


# ============================================================================
# Per-GPU worker (spawned as a separate process, one CUDA device per worker)
# ============================================================================
def worker_main(
    gpu_id: int,
    world_size: int,
    args_dict: dict,
    chunk: list[dict],
):
    """Runs on one GPU. Loads model, iterates over `chunk`, writes one JSONL."""
    # Bind CUDA before importing torch to keep the child pinned to one card.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from PIL import Image
    from tqdm import tqdm

    torch.set_grad_enabled(False)
    device = "cuda:0"  # after CUDA_VISIBLE_DEVICES=gpu_id, this is the local card

    a = argparse.Namespace(**args_dict)

    print(f"[GPU {gpu_id}] loading model on {device} (dtype={a.dtype}, attn={a.attn_implementation}) ...", flush=True)
    t0 = time.time()
    model, processor = load_model_and_processor(
        a.model,
        a.dtype,
        a.attn_implementation,
        device,
    )
    print(f"[GPU {gpu_id}] model ready in {time.time() - t0:.1f}s", flush=True)

    tokenizer = processor.tokenizer
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id

    # ------------------------------------------------------------------
    # Task routing (aligned with the vLLM client):
    #   --task-type SET   → override every row's prompt with the official
    #                       TASK_PROMPTS[task_type]; normalize_doc_parse_markdown iff task_type
    #                       == "doc_parse".
    #   --task-type UNSET → per-row prompt from JSONL[prompt_key] (fallback
    #                       to --prompt); normalize_doc_parse_markdown only when the actual prompt
    #                       matches the official doc_parse wording (avoids
    #                       corrupting non-markdown outputs like spotting
    #                       JSON, formula LaTeX, table HTML — see
    #                       output_utils.py header comment).
    # ------------------------------------------------------------------
    forced_prompt = None
    force_doc_pp_gate = None  # None → decide per-sample; True/False → fixed
    if a.task_type:
        forced_prompt = get_prompt(a.task_type)
        force_doc_pp_gate = a.task_type == "doc_parse"
    doc_parse_prompt = TASK_PROMPTS["doc_parse"]

    # doc_parse markdown normalization gate:
    #   - --no-doc-postprocess          → disabled
    #   - --task-type is doc_parse      → always on
    #   - --task-type is non-doc_parse  → always off
    #   - --task-type unset             → decide per-sample by matching the prompt
    doc_pp_enabled = (not a.no_doc_postprocess) and (force_doc_pp_gate is not False)

    output_path = f"{a.output}_{gpu_id + 1}.jsonl"

    # Resume from previous run: skip already-written lines.
    start_idx = 0
    if not a.no_resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            start_idx = sum(1 for line in f if line.strip())
        print(f"[GPU {gpu_id}] resume from idx={start_idx} (out of {len(chunk)})", flush=True)

    mode = "a" if start_idx > 0 else "w"
    with open(output_path, mode, encoding="utf-8") as fout:
        pbar = tqdm(
            chunk[start_idx:],
            desc=f"GPU{gpu_id}",
            position=gpu_id,
            leave=True,
        )
        for data in pbar:
            try:
                img_path = unwrap_field(data, a.image_key)
                if img_path is None:
                    raise ValueError(f"missing image field '{a.image_key}'")
                if forced_prompt is not None:
                    prompt = forced_prompt
                else:
                    prompt = unwrap_field(data, a.prompt_key, default=a.prompt)
                    if not prompt:
                        prompt = a.prompt

                with Image.open(img_path) as raw:
                    image = raw.convert("RGB")

                messages = [
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=[text],
                    images=image,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)

                input_ids = inputs["input_ids"] if "input_ids" in inputs else inputs["inputs"]
                prompt_len = int(input_ids.shape[1])

                stopping_criteria = None
                if not a.no_stream:
                    stopping_criteria = build_stopping_criteria(
                        processor,
                        prompt_len=prompt_len,
                        min_repeats=a.repeat_min_repeats,
                    )

                gen_kwargs = {
                    "max_new_tokens": a.max_new_tokens,
                    "do_sample": False,  # temperature=0 → greedy
                    "repetition_penalty": a.repetition_penalty,
                    "use_cache": True,
                }
                if eos_token_id is not None:
                    gen_kwargs["eos_token_id"] = eos_token_id
                if pad_token_id is not None:
                    gen_kwargs["pad_token_id"] = pad_token_id
                if stopping_criteria is not None:
                    gen_kwargs["stopping_criteria"] = stopping_criteria

                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, **gen_kwargs)

                trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
                decoded = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                out_text = decoded[0] if decoded else ""
                out_text = clean_repeated_substrings(out_text)
                # doc_parse markdown 规整（与 vLLM client 同源）：只在 doc_parse
                # 样本上生效；--task-type 未指定时通过 prompt 精确匹配官方
                # doc_parse wording，避免污染 spotting / formula / table 等非
                # markdown 输出（见 inference/utils/output_utils.py 顶部注释）。
                if doc_pp_enabled and (force_doc_pp_gate is True or prompt == doc_parse_prompt):
                    try:
                        out_text, _ = normalize_doc_parse_markdown(out_text)
                    except Exception:  # noqa: S110
                        pass
                data[a.answer_key] = out_text

                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                fout.flush()

            except Exception as e:
                err = f"ERROR: {type(e).__name__}: {e!s}"
                print(f"[GPU {gpu_id}] {err}", flush=True)
                data[a.answer_key] = err
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"[GPU {gpu_id}] done → {output_path}", flush=True)


# ============================================================================
# Merge helper
# ============================================================================
def merge_outputs(output_base: str, num_gpus: int):
    merged_path = f"{output_base}.merged.jsonl"
    total = 0
    with open(merged_path, "w", encoding="utf-8") as fout:
        for gpu_id in range(1, num_gpus + 1):
            part = f"{output_base}_{gpu_id}.jsonl"
            if not os.path.exists(part):
                print(f"  [skip] {part} not found")
                continue
            with open(part, encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
                        total += 1
    print(f"\n[merge] {merged_path}  ({total} lines)")


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-GPU HuggingFace transformers inference for HunyuanOCR-1.5 (AR baseline, no vLLM)."
    )
    # Handle --list-tasks before requiring --output.
    if "--list-tasks" in sys.argv:
        print("Available task types (--task-type):")
        for key in TASK_PROMPTS:
            print(f"  {key:18s} {TASK_DESCRIPTIONS.get(key, '')}")
        sys.exit(0)
    p.add_argument(
        "--list-tasks",
        action="store_true",
        help="list all official task types and exit",
    )
    p.add_argument(
        "--task-type",
        default=None,
        choices=list(TASK_PROMPTS.keys()),
        help=(
            "force ALL rows to use the official prompt of this task (from "
            "inference/utils/tasks.py); this also gates doc_parse markdown "
            "normalization (only enabled for task_type='doc_parse'). "
            "If unset (default), the per-row --prompt-key field is used (falling "
            "back to --prompt), preserving the legacy JSONL-driven workflow."
        ),
    )
    p.add_argument("--model", default="your/path/to/HunyuanOCR", help="HunyuanOCR model directory")
    p.add_argument("--input", default="your/path/to/input.jsonl", help="input jsonl path")
    p.add_argument(
        "--output", required=True, help="output base path (no .jsonl suffix); files: <output>_<gpu_id>.jsonl"
    )
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument(
        "--gpu-ids",
        default=None,
        help="comma-separated physical GPU ids to use, e.g. '0,1,2,3'. Overrides --num-gpus if set.",
    )

    p.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="fallback prompt used when the input row has no prompt field"
    )
    p.add_argument("--prompt-key", default="问题", help="json key for per-sample prompt")
    p.add_argument("--image-key", default="image_path", help="json key for image path (str or list[str])")
    p.add_argument("--answer-key", default="hf_answer", help="json key to write model output into")

    # Model / dtype
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])

    # Sampling — aligned with inference/vLLM/infer_vllm_client.py
    #   temperature=0.0 → do_sample=False; top_p / top_k are ignored under greedy.
    p.add_argument("--max-new-tokens", type=int, default=32768)
    p.add_argument("--repetition-penalty", type=float, default=1.08)
    p.add_argument("--repeat-min-repeats", type=int, default=8, help="tail-repeat threshold that triggers early-stop")
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="disable tail-repetition StoppingCriteria (one-shot generation, no early-stop)",
    )
    p.add_argument(
        "--no-doc-postprocess",
        action="store_true",
        help="disable doc_parse markdown normalization (output_utils.normalize_doc_parse_markdown). "
        "Default: enabled, to match the vLLM client output convention.",
    )

    p.add_argument("--no-resume", action="store_true", help="disable resume (overwrite existing output files)")
    p.add_argument("--merge", action="store_true", help="merge per-GPU outputs into <output>.merged.jsonl after done")
    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    else:
        gpu_ids = list(range(args.num_gpus))
    world_size = len(gpu_ids)
    if world_size == 0:
        print("[error] no GPUs specified", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    chunks = split_jsonl(args.input, world_size)
    total = sum(len(c) for c in chunks)
    print(f"[info] input:    {args.input}")
    print(f"[info] total:    {total} samples")
    print(f"[info] gpus:     {gpu_ids}")
    print(f"[info] chunks:   {[len(c) for c in chunks]}")
    print(f"[info] model:    {args.model}")
    print(f"[info] output:   {args.output}_<gpu_id>.jsonl")
    print(f"[info] dtype:    {args.dtype}  attn={args.attn_implementation}")
    print(f"[info] sampling: greedy (t=0), rep_penalty={args.repetition_penalty}, max_new_tokens={args.max_new_tokens}")
    print()

    # Convert argparse.Namespace → plain dict for cross-process transfer.
    args_dict = vars(args)

    # Spawn one child process per GPU. Using multiprocessing.spawn (not
    # torch.multiprocessing.spawn) so we can pass arbitrary GPU IDs via the
    # child's CUDA_VISIBLE_DEVICES (torch.mp.spawn assumes contiguous ranks).
    import multiprocessing as mp

    ctx = mp.get_context("spawn")

    t0 = time.time()
    procs = []
    for local_rank, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=worker_main,
            args=(gpu_id, world_size, args_dict, chunks[local_rank]),
            daemon=False,
        )
        p.start()
        procs.append(p)

    exit_codes = []
    for p in procs:
        p.join()
        exit_codes.append(p.exitcode)

    elapsed = time.time() - t0
    print(f"\n[done] elapsed {elapsed:.1f}s  ({total / max(elapsed, 1):.2f} samples/s)")
    print(f"[done] exit codes: {exit_codes}")
    print(f"[done] outputs:    {args.output}_[1-{world_size}].jsonl")

    if args.merge:
        merge_outputs(args.output, world_size)

    # Propagate a non-zero exit if any worker failed.
    if any(code != 0 for code in exit_codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
