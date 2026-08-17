#!/usr/bin/env python3
"""
Convert the makedata HuggingFace imagefolder tree into HunyuanOCR raw JSONL.

Source layout (produced by makedata/build_hf_upload.py):

    <root>/<config>/<split>/metadata.jsonl   {"file_name": "000.jpg", "text": ...}
    <root>/<config>/<split>/000.jpg

Output — one line per sample, in the schema `tools/pipeline_count_and_pack.py`
actually reads (NOT the one in docs/data_format.md, which documents an
`image_path` / `conversations` schema that no code in this repo consumes):

    {"img_path_sh": "/abs/000.jpg", "conv": [{"question": <prompt>, "answer": <markdown>}]}

The prompt goes in `question`, not in a system message: the packer never
forwards a system prompt into the packed records, so anything put there is
counted at pack time and then dropped at train time.
"""
import argparse
import json
import os
import sys

CONFIGS = ("GCN", "Documents", "Others", "Handwriting")


def read_prompt(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def load_split(root, config, split):
    d = os.path.join(root, config, split)
    meta = os.path.join(d, "metadata.jsonl")
    if not os.path.isfile(meta):
        return None
    rows = []
    with open(meta, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    for r in rows:
        r["_abs"] = os.path.abspath(os.path.join(d, r["file_name"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="makedata/hf_upload directory")
    ap.add_argument("--prompt-file", required=True, help="file holding the OCR prompt")
    ap.add_argument("--out-dir", default="./data/raw", help="where to write the JSONLs")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--per-config", action="store_true",
                    help="one JSONL per config instead of one merged file per split")
    ap.add_argument("--drop-empty", action="store_true",
                    help="drop samples whose label is empty (default: keep them — "
                         "they are the blank-page samples that teach the model to "
                         "emit nothing)")
    ap.add_argument("--data-list", default=None,
                    help="also write a configs/data_list.txt pointing at the train JSONL(s)")
    args = ap.parse_args()

    prompt = read_prompt(args.prompt_file)
    os.makedirs(args.out_dir, exist_ok=True)
    written, report = {}, []

    for split in args.splits:
        buckets = {}
        for config in args.configs:
            rows = load_split(args.root, config, split)
            if rows is None:
                report.append((config, split, 0, 0, 0, "KHÔNG CÓ"))
                continue
            keep, empty, missing = [], 0, 0
            for r in rows:
                if not os.path.isfile(r["_abs"]):
                    missing += 1
                    continue
                if not r["text"].strip():
                    empty += 1
                    if args.drop_empty:
                        continue
                keep.append({"img_path_sh": r["_abs"],
                             "conv": [{"question": prompt, "answer": r["text"]}]})
            buckets[config] = keep
            report.append((config, split, len(rows), empty, missing, len(keep)))

        if args.per_config:
            for config, rows in buckets.items():
                p = os.path.join(args.out_dir, f"{config}_{split}.jsonl")
                with open(p, "w", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                written.setdefault(split, []).append(p)
        else:
            p = os.path.join(args.out_dir, f"{split}.jsonl")
            with open(p, "w", encoding="utf-8") as fh:
                for config in args.configs:
                    for r in buckets.get(config, []):
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            written.setdefault(split, []).append(p)

    print(f"{'config':13}{'split':12}{'rows':>7}{'rỗng':>7}{'thiếu ảnh':>11}{'ghi':>7}")
    for row in report:
        print(f"{row[0]:13}{row[1]:12}{row[2]:>7}{row[3]:>7}{row[4]:>11}{row[5]:>7}")
    print()
    for split, paths in written.items():
        for p in paths:
            n = sum(1 for _ in open(p, encoding="utf-8"))
            print(f"{p}: {n} dòng")

    if args.data_list:
        os.makedirs(os.path.dirname(args.data_list) or ".", exist_ok=True)
        with open(args.data_list, "w", encoding="utf-8") as fh:
            fh.write("# Sinh bởi tools/makedata_to_hyocr.py — không sửa tay.\n")
            for p in written.get("train", []):
                fh.write(os.path.abspath(p) + "\n")
        print(f"\n{args.data_list} đã ghi {len(written.get('train', []))} đường dẫn")

    total_empty = sum(r[3] for r in report if isinstance(r[3], int))
    if total_empty and not args.drop_empty:
        print(f"\n[chú ý] {total_empty} mẫu nhãn rỗng được giữ lại. "
              f"pipeline_count_and_pack.py sẽ loại chúng trừ khi chạy với "
              f"--allow-empty-answer.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
