"""Convert raw OCR JSONL files into a verl-compatible parquet dataset.

## Input JSONL

Each line is one training sample. Only two things are required:

  1. an **image path** (or a list of paths for multi-image samples);
  2. a **reference answer** (the ground truth the rollout will be scored against).

Optional fields:

  * ``prompt`` — either a raw user text string, or a chat-messages list
    ``[{"role": "user", "content": "<image>...text..."}, ...]``.
    If omitted, an empty prompt is used and the script inlines one ``<image>``
    placeholder per image at the front of a single user turn.
  * ``task_type`` — required at rollout time to route reward scoring (values:
    ``spotting`` / ``layout`` / ``parsing`` / ``chart_deplot`` /
    ``ie`` / ``translation`` / ``vqa``). If not present in the record, pass
    ``--task-type`` to stamp every row with the same value.
  * ``source_lang_text`` / ``target_lang`` — only used by translation task.

Anything else in the record is dropped: any of ``ability``, ``split``,
``index``, ``reward_func``, ``data_id``, ``resolutions``, ``ref_answer``
duplicates, etc. Only what verl and the reward function actually read is kept.

## Output parquet schema (per row)

```python
{
    "data_source":  str,        # reward-router key; defaults to task_type
    "prompt":       list[dict], # chat messages, ``content`` is str with <image> placeholders
    "images":       list[str],  # absolute image paths, count matches <image> count
    "reward_model": {"style": "rule", "ground_truth": str},
    "extra_info": {
        "task_type":        str,          # required; drives reward dispatch
        "img_path":         list[str],    # optional; kept for the RLLoggingBoard dashboard
        "source_lang_text": str,          # optional; translation only
        "target_lang":      str,          # optional; translation only
    },
}
```

## Usage

```bash
python prepare_data.py \\
    --inputs /path/to/a.jsonl /path/to/b.jsonl \\
    --output_train /path/to/train.parquet \\
    --output_val   /path/to/val.parquet \\
    --val_ratio 0.02 \\
    --task-type layout
```
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def _flatten_openai_content(content: Any) -> tuple[str, int]:
    """Normalise ``content`` to (verl_text, image_placeholder_count).

    * str        → returned as-is; ``<image>`` occurrences counted.
    * list[dict] → flattened to ``"<image>\\n...text..."``-style string
      (verl requires content to be a plain str with literal ``<image>`` markers,
      not OpenAI-style structured content).
    * anything else → ``str(content)`` (best effort).
    """
    if isinstance(content, str):
        return content, content.count("<image>")
    if not isinstance(content, list):
        return str(content), 0
    parts: list[str] = []
    n_img = 0
    for seg in content:
        if not isinstance(seg, dict):
            s = str(seg)
            if s:
                parts.append(s)
            continue
        seg_type = seg.get("type")
        if seg_type == "image" or "image" in seg or "image_url" in seg:
            parts.append("<image>")
            n_img += 1
        elif "text" in seg:
            t = seg.get("text", "")
            if t:
                parts.append(t)
    return "\n".join(parts), n_img


def _normalise_messages(prompt_field: Any, images: list, fallback_text: str) -> list[dict]:
    """Return a verl-ready messages list from various JSONL prompt layouts.

    Accepts:
      * list[{"role","content"}] — canonical.
      * str                      — raw user text; wrapped in a single turn.
      * anything else            — falls back to ``fallback_text``.
    Ensures the ``<image>`` placeholder count equals ``len(images)``.
    """
    n_expected = len(images)

    # Case A: already a messages list.
    if isinstance(prompt_field, list) and prompt_field and isinstance(prompt_field[0], dict):
        msgs: list[dict] = []
        total_img = 0
        for m in prompt_field:
            role = m.get("role", "user")
            text, n_img = _flatten_openai_content(m.get("content", ""))
            msgs.append({"role": role, "content": text})
            total_img += n_img
        if total_img != n_expected:
            # Patch the first user turn to have the correct number of placeholders.
            for msg in msgs:
                if msg["role"] == "user":
                    stripped = msg["content"].replace("<image>", "")
                    msg["content"] = ("<image>\n" * n_expected) + stripped
                    break
        return msgs

    # Case B: flat string prompt.
    if isinstance(prompt_field, str):
        return [{"role": "user", "content": ("<image>\n" * n_expected) + prompt_field}]

    # Case C: fallback.
    return [{"role": "user", "content": ("<image>\n" * n_expected) + (fallback_text or "")}]


def map_record(rec: dict, task_type_override: str | None, data_source_override: str | None) -> dict:
    """Convert a raw JSONL record to a verl-compatible parquet row."""
    # ---- images ----
    images = rec.get("images")
    if not images:
        img_path = rec.get("img_path")
        if isinstance(img_path, list):
            images = list(img_path)
        elif isinstance(img_path, str) and img_path:
            images = [img_path]
        else:
            images = []
    images = [str(p) for p in images]

    # ---- task_type ----
    task_type = task_type_override or rec.get("task_type") or rec.get("ocr_task_type") or ""
    if not task_type:
        raise ValueError(
            "task_type is required (either in the record as `task_type` / `ocr_task_type`, "
            "or via the --task-type CLI flag)"
        )

    # ---- prompt / messages ----
    prompt_msgs = _normalise_messages(rec.get("prompt"), images, rec.get("question") or "")

    # ---- reward_model ----
    ground_truth = rec.get("ref_answer") or ""
    if not ground_truth and isinstance(rec.get("reward_model"), dict):
        ground_truth = rec["reward_model"].get("ground_truth", "")
    reward_model = {"style": "rule", "ground_truth": ground_truth}

    # ---- extra_info (minimal) ----
    extra_info: dict = {"task_type": task_type, "question": rec.get("question") or ""}
    if images:
        extra_info["img_path"] = images
    for k in ("source_lang_text", "target_lang"):
        v = rec.get(k)
        if v:
            extra_info[k] = v

    # ---- data_source (reward-router key) ----
    data_source = data_source_override or rec.get("data_source") or task_type

    return {
        "data_source": data_source,
        "prompt": prompt_msgs,
        "images": images,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def read_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] skip malformed line in {path}: {e}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more .jsonl input files.")
    parser.add_argument("--output_train", required=True, help="Output train.parquet path.")
    parser.add_argument("--output_val", default=None, help="Output val.parquet path (optional).")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Fraction of samples routed to val split.")
    parser.add_argument(
        "--task-type",
        default=None,
        help="Override ``task_type`` for every record (drives reward dispatch). "
        "If unset, each record must carry its own ``task_type``.",
    )
    parser.add_argument(
        "--data_source",
        default=None,
        help="Override the ``data_source`` column. If unset, defaults to the "
        "record's own ``data_source``, then falls back to ``task_type``.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records: list[dict] = []
    for p in args.inputs:
        raw = read_jsonl(p)
        print(f"[info] {p}: {len(raw)} records")
        records.extend(raw)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = int(len(records) * args.val_ratio)
    val_records = records[:n_val]
    train_records = records[n_val:]

    train_rows = [map_record(r, args.task_type, args.data_source) for r in train_records]
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(args.output_train, index=False)
    print(f"[info] wrote {len(train_rows)} rows -> {args.output_train}")

    if args.output_val and val_records:
        val_rows = [map_record(r, args.task_type, args.data_source) for r in val_records]
        Path(args.output_val).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(val_rows).to_parquet(args.output_val, index=False)
        print(f"[info] wrote {len(val_rows)} rows -> {args.output_val}")


if __name__ == "__main__":
    main()
