"""Smoke-test the local llama.cpp / vLLM deployment on the bundled OCR images.

Iterates over every image under `llama_cpp/test_assets/ocr/`, sends it to the
server with the official doc_parse prompt, and prints the response together
with per-item elapsed time. Uses the same request pipeline as `chat.py`
(via the imported `chat()` function).

Usage:
    # 1. Start llama-server (see docs/llama_cpp.md §2)
    # 2. Run:
    python test_chat.py
"""

import os
import time

from chat import DOC_PARSE_PROMPT, chat, maybe_postprocess

# ---------------- knobs ----------------
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_assets", "ocr")
PROMPT = DOC_PARSE_PROMPT
# doc_parse markdown normalization on the response (output_utils.normalize_doc_parse_markdown).
DO_DOC_POSTPROCESS = True
# Cap number of images (None = all).
MAX_IMAGES = None
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
# ----------------------------------------


def _list_images(image_dir: str):
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS)]

    # Numeric sort when filenames look like "0.png", "10.jpg", ... else lexicographic.
    def _key(name: str):
        stem, _ = os.path.splitext(name)
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    return sorted(files, key=_key)


def main():
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"image dir not found: {IMAGE_DIR}")
    files = _list_images(IMAGE_DIR)
    if MAX_IMAGES is not None:
        files = files[:MAX_IMAGES]
    if not files:
        print(f"[warn] no images found under {IMAGE_DIR}")
        return

    print(f"=== running on {len(files)} image(s) from {IMAGE_DIR} ===")
    print(f"=== prompt: {PROMPT}")
    print()

    total_start = time.perf_counter()
    for i, name in enumerate(files, 1):
        path = os.path.join(IMAGE_DIR, name)
        print(f"--- [{i}/{len(files)}] {name} ---")
        item_start = time.perf_counter()
        out = chat(PROMPT, image_paths=[path])
        out = maybe_postprocess(
            out,
            prompt=PROMPT,
            task_type="doc_parse",
            disable=not DO_DOC_POSTPROCESS,
        )
        print(out)
        print(f"[elapsed] {time.perf_counter() - item_start:.3f}s")
        print()

    print(f"[total] {len(files)} images, elapsed: {time.perf_counter() - total_start:.3f}s")


if __name__ == "__main__":
    main()
