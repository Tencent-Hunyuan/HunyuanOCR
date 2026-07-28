"""
Minimal OpenAI-compatible client for a locally deployed HunyuanOCR-1.5 vLLM server.

Sends **one image + one prompt** to the server and prints the response and
per-request latency. Same sampling / streaming / tail-repetition early-stop /
tail-repeat cleanup as `inference/vLLM/batch_infer.py` and
`inference/transformers/infer_hf_8gpu.py`, so results are directly
comparable across AR, DFlash, and transformers paths — no matter which server
you start it against (`inference/vLLM/serve.sh` for AR, or
`inference/DFlash/serve_DFlash.sh` for DFlash speculative decoding).

Post-processing (tail-repetition detection / cleanup / streaming early-stop,
image encoding, and the doc_parse-only markdown normalization) lives in
`inference/utils/output_utils.py` and is imported here so the exact same logic
is shared across the single-image client, batch runners and eval pipelines.

Prompt selection is LOCKED to a fixed set of official task types via
`--task-type` (see `inference/utils/tasks.py`). Free-form prompt editing
is intentionally NOT exposed: hand-tweaked instructions were observed to
silently degrade model quality, so users pick a task, not a prompt.

Usage:
    # 1. start the server (in another terminal)
    MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/vLLM/serve.sh            # AR
    # or
    MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/DFlash/serve_DFlash.sh   # DFlash

    # 2. send a single-image inference request
    python inference/vLLM/infer_vllm_client.py \
        --image /path/to/document.png \
        --task-type doc_parse \
        --port 8000

    # list all task types
    python inference/vLLM/infer_vllm_client.py --list-tasks

Optional flags:
    --host / --port           default 127.0.0.1:8000
    --model                   default 'tencent/HunyuanOCR' (must match --served-model-name)
    --task-type               one of the official task types (default: doc_parse)
    --max-tokens              default 4096
    --repetition-penalty      default 1.08
    --no-stream               disable streaming + early-stop (one-shot generation)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Shared output utilities (streaming/early-stop + doc_parse markdown normalization).
from utils.output_utils import (
    clean_repeated_substrings,
    encode_image_as_data_url,
    infer_stream,
    normalize_doc_parse_markdown,
)
from utils.tasks import (
    DEFAULT_TASK,
    TASK_DESCRIPTIONS,
    TASK_PROMPTS,
    get_prompt,
)


def _print_task_list():
    print("Available task types (--task-type):")
    for key in TASK_PROMPTS:
        print(f"  {key:18s} {TASK_DESCRIPTIONS.get(key, '')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Path to input image")
    ap.add_argument(
        "--task-type",
        default=DEFAULT_TASK,
        choices=list(TASK_PROMPTS.keys()),
        metavar="TASK",
        help=f"official task type (use --list-tasks to see all); default: {DEFAULT_TASK}",
    )
    ap.add_argument("--list-tasks", action="store_true", help="print all task types and exit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="tencent/HunyuanOCR", help="Must match vLLM --served-model-name")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--repetition-penalty", type=float, default=1.08)
    ap.add_argument(
        "--repeat-min-repeats", type=int, default=8, help="tail-repeats threshold that triggers streaming early-stop"
    )
    ap.add_argument("--no-stream", action="store_true", help="disable streaming + early-stop (one-shot generation)")
    ap.add_argument(
        "--no-doc-postprocess",
        action="store_true",
        help="disable the doc_parse-only markdown normalization (output_utils.normalize_doc_parse_markdown)",
    )
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    if args.list_tasks:
        _print_task_list()
        return

    if not args.image:
        print("[ERROR] --image is required (or use --list-tasks)", file=sys.stderr)
        sys.exit(1)
    if not Path(args.image).is_file():
        print(f"[ERROR] image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    prompt = get_prompt(args.task_type)

    from openai import OpenAI

    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{args.host}:{args.port}/v1",
        timeout=args.timeout,
    )

    image_url = encode_image_as_data_url(args.image)
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    common_kwargs = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "extra_body": {
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "skip_special_tokens": True,
        },
    }

    print(
        f"[info] POST http://{args.host}:{args.port}/v1/chat/completions "
        f"(model={args.model}, task={args.task_type}, stream={not args.no_stream})"
    )
    t = time.time()

    early_stopped = False
    usage = None
    if args.no_stream:
        resp = client.chat.completions.create(stream=False, **common_kwargs)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
    else:
        text, early_stopped = infer_stream(
            client,
            common_kwargs,
            args.repeat_min_repeats,
        )

    text = clean_repeated_substrings(text)

    # doc_parse only: normalize markdown to OmniDocBench GT convention.
    doc_pp_stats = None
    if args.task_type == "doc_parse" and not args.no_doc_postprocess:
        text, doc_pp_stats = normalize_doc_parse_markdown(text)

    dt = time.time() - t

    print("=" * 60)
    print(f"Task             : {args.task_type}")
    print(f"Latency          : {dt:.2f}s")
    print(f"Output chars     : {len(text)}")
    if usage is not None:
        print(f"Prompt tokens    : {usage.prompt_tokens}")
        print(f"Completion tokens: {usage.completion_tokens}")
        if usage.completion_tokens:
            print(f"tok/s            : {usage.completion_tokens / dt:.1f}")
    if early_stopped:
        print("Early-stopped    : yes (tail repetition detected)")
    if doc_pp_stats:
        applied = {k: v for k, v in doc_pp_stats.items() if v}
        if applied:
            print(f"Doc-postprocess  : {applied}")
    print("=" * 60)
    print(text)


if __name__ == "__main__":
    main()
