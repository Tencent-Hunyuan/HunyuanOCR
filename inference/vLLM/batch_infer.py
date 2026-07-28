"""HunyuanOCR-1.5 vLLM 批量推理 (多端点并发)。
后处理逻辑与单图 infer_vllm_client.py 同源: 直接 import output_utils / tasks,
不再内嵌副本。prompt 锁定为 --task-type。doc_parse 时默认做 markdown 规整
(normalize_doc_parse_markdown), 可用 --no-doc-postprocess 关闭。
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.output_utils import (
    clean_repeated_substrings,
    encode_image_as_data_url,
    infer_stream,
    normalize_doc_parse_markdown,
)
from utils.tasks import DEFAULT_TASK, TASK_PROMPTS, get_prompt

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def run_one(client, model, img_path, prompt, max_tokens, rep_pen, repeat_min, do_doc_pp):
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image_as_data_url(img_path)}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    common = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "extra_body": {"top_k": -1, "repetition_penalty": rep_pen, "skip_special_tokens": True},
    }
    text, early = infer_stream(client, common, repeat_min)
    text = clean_repeated_substrings(text)
    pp_stats = None
    if do_doc_pp:
        text, pp_stats = normalize_doc_parse_markdown(text)
    return text, early, pp_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ports", default="8000")
    ap.add_argument("--model", default="tencent/HunyuanOCR")
    ap.add_argument("--task-type", default=DEFAULT_TASK, choices=list(TASK_PROMPTS.keys()))
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--repetition-penalty", type=float, default=1.08)
    ap.add_argument("--repeat-min-repeats", type=int, default=8)
    ap.add_argument("--no-doc-postprocess", action="store_true", help="doc_parse 时关闭 markdown 规整")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ports = [int(x) for x in args.ports.split(",") if x.strip()]
    clients = [OpenAI(api_key="EMPTY", base_url=f"http://{args.host}:{p}/v1", timeout=3600.0) for p in ports]
    prompt = get_prompt(args.task_type)
    do_doc_pp = (args.task_type == "doc_parse") and (not args.no_doc_postprocess)
    print(
        f"[info] task={args.task_type} doc_postprocess={do_doc_pp} "
        f"endpoints={ports} max_tokens={args.max_tokens} rep_pen={args.repetition_penalty}",
        flush=True,
    )

    imgs = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith(EXTS))
    if args.limit:
        imgs = imgs[: args.limit]
    todo = []
    for f in imgs:
        md = os.path.join(args.out_dir, os.path.splitext(f)[0] + ".md")
        if os.path.exists(md) and os.path.getsize(md) > 0:
            continue
        todo.append(f)
    print(f"[info] total={len(imgs)} skip={len(imgs) - len(todo)} todo={len(todo)}", flush=True)

    # Shared jsonl handle used by the nested worker under a thread pool; a
    # context manager would close it before the workers finish.
    jsonl = open(os.path.join(args.out_dir, "results.jsonl"), "a", encoding="utf-8")  # noqa: SIM115
    st = {"n": 0, "ok": 0, "err": 0, "early": 0, "pp": 0, "t0": time.time()}

    def work(item):
        idx, fname = item
        client = clients[idx % len(clients)]
        img_path = os.path.join(args.image_dir, fname)
        md = os.path.join(args.out_dir, os.path.splitext(fname)[0] + ".md")
        try:
            t = time.time()
            text, early, pp = run_one(
                client,
                args.model,
                img_path,
                prompt,
                args.max_tokens,
                args.repetition_penalty,
                args.repeat_min_repeats,
                do_doc_pp,
            )
            with open(md, "w", encoding="utf-8") as f:
                f.write(text)
            pp_applied = bool(pp and any(v for v in pp.values()))
            return {
                "image": fname,
                "chars": len(text),
                "early_stopped": early,
                "normalize_doc_parse_markdown": {k: v for k, v in (pp or {}).items() if v},
                "latency": round(time.time() - t, 2),
                "ok": True,
                "_pp_applied": pp_applied,
            }
        except Exception as e:
            return {"image": fname, "ok": False, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, it) for it in enumerate(todo)]
        for fut in as_completed(futs):
            r = fut.result()
            r.pop("_pp_applied", None) if False else None
            pp_applied = r.pop("_pp_applied", False)
            jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
            jsonl.flush()
            st["n"] += 1
            st["ok" if r["ok"] else "err"] += 1
            if r.get("early_stopped"):
                st["early"] += 1
            if pp_applied:
                st["pp"] += 1
            if not r["ok"] or st["n"] % 25 == 0:
                el = time.time() - st["t0"]
                rate = st["n"] / el if el else 0
                print(
                    f"[{st['n']}/{len(todo)}] ok={st['ok']} err={st['err']} "
                    f"early={st['early']} pp={st['pp']} {rate:.2f}img/s "
                    f"eta={(len(todo) - st['n']) / rate / 60:.1f}min"
                    + ("" if r["ok"] else f"  ERR {r['image']}: {r['error']}"),
                    flush=True,
                )
    jsonl.close()
    print(
        f"[done] ok={st['ok']} err={st['err']} early={st['early']} "
        f"doc_pp_applied={st['pp']} in {(time.time() - st['t0']) / 60:.1f}min",
        flush=True,
    )


if __name__ == "__main__":
    main()
