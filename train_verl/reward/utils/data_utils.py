"""Minimal image helpers used by the reward scorer."""

import base64
from io import BytesIO

import requests
from PIL import Image


def encode_image(image_path: str, max_pixels: int = 1024 * 1024) -> str:
    """Read an image (local path or http URL), optionally downscale so the
    total pixel count stays under ``max_pixels``, then return a base64 string."""
    if image_path.startswith("http"):
        content = requests.get(image_path, timeout=30).content
        img = Image.open(BytesIO(content))
    else:
        img = Image.open(image_path)

    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format=img.format or "PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
