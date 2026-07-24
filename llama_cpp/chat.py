"""Minimal OpenAI-compatible client for the local vLLM / llama-server backend.

Two modes:

    # Batch mode (JSONL input; task-type routing + doc_parse post-processing,
    # aligned with inference/transformers/infer_hf_8gpu.py):
    python chat.py --input test_assets/data.jsonl [--task-type doc_parse] [...]

    # Smoke test on the bundled OCR images (no JSONL needed):
    python test_chat.py

`chat()` is exported as a library function so `test_chat.py` and other quick
scripts can reuse the same request pipeline.
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from datetime import datetime

from openai import OpenAI

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from inference.utils.hunyuan_tasks import (  # noqa: E402
    TASK_DESCRIPTIONS,
    TASK_PROMPTS,
    get_prompt,
)
from inference.utils.hunyuan_utils import process_one as _doc_pp  # noqa: E402

DOC_PARSE_PROMPT = TASK_PROMPTS["doc_parse"]

BASE_URL = "http://127.0.0.1:8080/v1"

_client = OpenAI(base_url=BASE_URL, api_key="empty")


class _Tee:
    """Write to multiple streams at once (e.g. terminal + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def _redirect_output_to_log() -> str:
    """Tee stdout/stderr to a timestamped log file in logs/. Returns the path."""
    log_dir = os.path.join(_HERE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"chat_{datetime.now():%Y%m%d_%H%M%S}.log")
    log_file = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path


def _image_part(path: str) -> dict:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def chat(
    prompt: str,
    image_paths: list = None,
    max_tokens: int = 4096,
    temperature: float = 0,
    top_p: float = 1,
    top_k: int = 1,
    repetition_penalty: float = 1,
) -> str:
    if image_paths:
        content = [*(_image_part(p) for p in image_paths), {"type": "text", "text": prompt}]
    else:
        content = prompt

    resp = _client.chat.completions.create(
        model=_client.models.list().data[0].id,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={"top_k": top_k, "repetition_penalty": repetition_penalty},
    )
    return resp.choices[0].message.content or ""


def maybe_postprocess(text: str, prompt: str, task_type: str, disable: bool) -> str:
    """Apply hunyuan_utils.process_one iff the row is a doc_parse task.

    Matches the gating logic in inference/transformers/infer_hf_8gpu.py:
      * --task-type set   -> apply iff task_type == "doc_parse"
      * --task-type unset -> apply iff `prompt` equals the official doc_parse
                             wording (the per-row prompt drove inference)
    """
    if disable:
        return text
    if task_type is not None:
        gate = task_type == "doc_parse"
    else:
        gate = prompt == DOC_PARSE_PROMPT
    if not gate:
        return text
    try:
        out, _ = _doc_pp(text)
        return out
    except Exception:
        return text


def _iter_jsonl(path: str):
    """Yield objects from a (possibly pretty-printed) JSONL file."""
    with open(path, "r") as f:
        text = f.read().strip()
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, idx = decoder.raw_decode(text, idx)
        yield obj


def _parse_type_limits(raw: str) -> dict:
    """Parse "ocr=1,layout=2" -> {"ocr": 1, "layout": 2}. Empty -> None."""
    if raw is None or raw == "":
        return None
    out = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(f"invalid --type-limits entry '{chunk}', expected key=int")
        k, v = chunk.split("=", 1)
        try:
            out[k.strip()] = int(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid int in --type-limits: '{v}'")
    return out


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Batch OCR client for the local llama.cpp / vLLM server, aligned "
            "with inference/transformers/infer_hf_8gpu.py "
            "(task-type routing + doc_parse markdown normalization)."
        )
    )
    # --list-tasks shortcut (parsed manually so we don't need --input).
    if "--list-tasks" in sys.argv:
        print("Available task types (--task-type):")
        for key in TASK_PROMPTS:
            print(f"  {key:18s} {TASK_DESCRIPTIONS.get(key, '')}")
        sys.exit(0)
    p.add_argument("--list-tasks", action="store_true", help="list all official task types and exit")
    p.add_argument(
        "--input",
        required=True,
        help="input JSONL path; each item must carry {type?, prompt?, image_paths?}",
    )
    p.add_argument(
        "--task-type",
        default=None,
        choices=list(TASK_PROMPTS.keys()),
        help=(
            "force ALL rows to use the official prompt of this task (from "
            "inference/utils/hunyuan_tasks.py); this also gates doc_parse markdown "
            "normalization (only enabled when task_type='doc_parse'). "
            "If unset, each row's 'prompt' field is used (post-processing only "
            "applies to rows whose prompt matches the official doc_parse wording)."
        ),
    )
    p.add_argument(
        "--no-doc-postprocess",
        action="store_true",
        help="disable doc_parse markdown normalization (hunyuan_utils.process_one).",
    )
    p.add_argument("--max-requests", type=int, default=None, help="cap total requests (default: unlimited)")
    p.add_argument(
        "--type-limits",
        type=_parse_type_limits,
        default=None,
        help='per-type cap, e.g. "ocr=1,layout=2"; unset means no filtering',
    )
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top-p", type=float, default=1)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--repetition-penalty", type=float, default=1)
    p.add_argument("--no-log", action="store_true", help="disable tee-to-log (logs/chat_<ts>.log)")
    return p.parse_args()


def _run_batch(args):
    base_dir = os.path.dirname(os.path.abspath(args.input))
    forced_prompt = get_prompt(args.task_type) if args.task_type else None

    total_start = time.perf_counter()
    count = 0
    type_counts = {}
    for item in _iter_jsonl(args.input):
        if args.max_requests is not None and count >= args.max_requests:
            break

        item_type = item.get("type", "unknown")

        if args.type_limits is not None:
            if item_type not in args.type_limits:
                continue
            limit = args.type_limits[item_type]
            if limit is not None and type_counts.get(item_type, 0) >= limit:
                continue

        prompt = forced_prompt if forced_prompt is not None else item.get("prompt", DOC_PARSE_PROMPT)
        image_paths = [os.path.join(base_dir, p) for p in item.get("image_paths", [])]
        label = ", ".join(item.get("image_paths", [])) or "(text-only)"

        print(f"=== [{item_type}] {label} ===")
        print(f"Prompt: {prompt}")
        item_start = time.perf_counter()
        out = chat(
            prompt,
            image_paths=image_paths or None,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        out = maybe_postprocess(out, prompt, args.task_type, args.no_doc_postprocess)
        print(out)
        print(f"[elapsed] {time.perf_counter() - item_start:.3f}s")
        print()
        count += 1
        type_counts[item_type] = type_counts.get(item_type, 0) + 1

    summary = ", ".join(f"{t}={n}" for t, n in type_counts.items()) or "none"
    print(f"[total] {count} items ({summary}), elapsed: {time.perf_counter() - total_start:.3f}s")


if __name__ == "__main__":
    args = _parse_args()
    if not args.no_log:
        _redirect_output_to_log()
    _run_batch(args)
