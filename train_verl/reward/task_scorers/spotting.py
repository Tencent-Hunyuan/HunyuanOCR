"""Spotting task rule scorer.

Entry point: ``score_spotting(response, ref_answer) -> dict``.

Supports four wire formats (auto-detected):
  * format1     : ``<ref>text</ref><quad>(x1,y1),(x2,y2)</quad>``
  * format2     : ``<quad><pos_x1><pos_y1><pos_x2><pos_y2></quad><ref>text</ref>``
  * format3     : ``text(x1,y1),(x2,y2)``
  * format_json : ``[{"box": [xmin,ymin,xmax,ymax], "text": "..."}, ...]``

Scoring combines IoU-based greedy matching with per-pair edit-distance on the
text content; see ``_compare_spotting_results_soft``.
"""

from __future__ import annotations

import json
import re

from .text_metrics import levenshtein_distance, normalize_text, snap_to_nearest_integer

_SPOTTING_FORMATS = ("format1", "format2", "format3", "format_json")


# ---------------------------------------------------------------------------
# format detection + parsing
# ---------------------------------------------------------------------------


def _detect_spotting_format(text: str) -> str:
    """Detect a spotting response format. Returns one of ``_SPOTTING_FORMATS``
    or ``'unknown'``.

    JSON detection is tried first (strict list-of-dicts with ``box``/``bbox``
    and ``text`` fields) so it does not stick to prefix-matching XML tags.
    """
    stripped = text.strip() if isinstance(text, str) else ""
    if (
        stripped.startswith("[")
        and stripped.endswith("]")
        and ('"box"' in stripped or '"bbox"' in stripped)
        and '"text"' in stripped
    ):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict) and ("box" in obj[0] or "bbox" in obj[0]):
                return "format_json"
        except Exception:  # noqa: S110
            pass

    if "<ref>" in text and "<quad>(" in text:
        return "format1"
    elif "<quad><pos_" in text and "</quad><ref>" in text:
        return "format2"
    elif re.search(r"[^\(]+\(\d+,\d+\),\(\d+,\d+\)", text):
        return "format3"
    return "unknown"


def _parse_format1(text: str) -> list[tuple]:
    """``<ref>text</ref><quad>(x1,y1),(x2,y2)</quad>`` -> ``[(text, x1, y1, x2, y2), ...]``."""
    try:
        parts = text.split("</quad>")
        results = []
        for part in parts:
            if not part.strip():
                continue
            try:
                ref_match = re.search(r"<ref>(.*?)</ref>", part)
                coord_match = re.search(r"<quad>\((\d+),(\d+)\),\((\d+),(\d+)\)", part)
                if ref_match and coord_match:
                    content = ref_match.group(1).strip()
                    x1, y1, x2, y2 = map(int, coord_match.groups())
                    results.append((content, x1, y1, x2, y2))
            except (ValueError, IndexError) as e:
                print(f"Error parsing part {part}: {e}")
                continue
        return results
    except Exception as e:
        print(f"Error in _parse_format1: {e}")
        return []


def _parse_format2(text: str) -> list[tuple]:
    """``<quad><pos_x1><pos_y1><pos_x2><pos_y2></quad><ref>text</ref>``."""
    try:
        pattern = re.compile(r"<quad><pos_(\d+)><pos_(\d+)><pos_(\d+)><pos_(\d+)></quad><ref>(.*?)</ref>")
        matches = pattern.findall(text)
        if not matches:
            print(f"No matches found in text: {text[:100]}...")
            return []
        results = []
        for match in matches:
            try:
                x1, y1, x2, y2 = map(int, match[:4])
                content = match[4].strip()
                results.append((content, x1, y1, x2, y2))
            except (ValueError, IndexError) as e:
                print(f"Error parsing individual match: {e}")
                continue
        return results
    except Exception as e:
        print(f"Error in _parse_format2: {e}")
        return []


def _parse_format3(text: str) -> list[tuple]:
    """``text(x1,y1),(x2,y2)``."""
    try:
        items = re.split(r"(?<=\))\s*(?=[^\s\(\),])", text)
        results = []
        for item in items:
            match = re.match(r"([^\(]+)\((\d+),(\d+)\),\((\d+),(\d+)\)", item.strip())
            if match:
                content = match.group(1).strip()
                x1, y1 = int(match.group(2)), int(match.group(3))
                x2, y2 = int(match.group(4)), int(match.group(5))
                results.append((content, x1, y1, x2, y2))
        return results
    except BaseException:
        return []


def _parse_format_json(text: str) -> list[tuple]:
    """``[{"box": [xmin,ymin,xmax,ymax], "text": "..."}, ...]`` -> ``[(text, x1, y1, x2, y2), ...]``.

    Accepts ``bbox`` as an alias for ``box``.
    """
    try:
        obj = json.loads(text.strip())
    except Exception as e:
        print(f"Error in _parse_format_json (json.loads): {e}")
        return []
    if not isinstance(obj, list):
        return []

    results: list[tuple] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        box = item.get("box", item.get("bbox"))
        content = item.get("text", "")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (int(v) for v in box)
        except (TypeError, ValueError) as e:
            print(f"Error parsing box in _parse_format_json: {e}")
            continue
        if not isinstance(content, str):
            content = str(content)
        results.append((content.strip(), x1, y1, x2, y2))
    return results


def _parse_by_format(text: str, fmt: str) -> list[tuple]:
    """Dispatch ``text`` to the parser corresponding to ``fmt``."""
    if fmt == "format1":
        return _parse_format1(text)
    if fmt == "format2":
        return _parse_format2(text)
    if fmt == "format3":
        return _parse_format3(text)
    if fmt == "format_json":
        return _parse_format_json(text)
    return []


# ---------------------------------------------------------------------------
# geometry + comparison
# ---------------------------------------------------------------------------


def _calculate_iou(box1: tuple, box2: tuple) -> float:
    """IoU of two ``(text, x1, y1, x2, y2)`` axis-aligned boxes."""
    x1 = max(box1[1], box2[1])
    y1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[3] - box1[1]) * (box1[4] - box1[2])
    area2 = (box2[3] - box2[1]) * (box2[4] - box2[2])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def _compare_spotting_results_soft(boxes1, boxes2, iou_threshold: float = 0.5) -> tuple[bool, float, str]:
    """Greedy IoU matching + normalised text edit distance.

    Args:
        boxes1: prediction boxes ``[(text, x1, y1, x2, y2), ...]``.
        boxes2: reference boxes ``[(text, x1, y1, x2, y2), ...]``.
        iou_threshold: minimum IoU to accept a box match.

    Returns:
        ``(is_valid, reward, reason)``.
    """
    try:
        matched_boxes2: set[int] = set()
        total_norm_dist = 0.0
        counter = 0
        tp = 0
        matched_text_norm_dist = 0.0

        # Best-IoU match for every prediction box.
        for _, box1 in enumerate(boxes1):
            best_iou = -1
            best_match = None
            for jdx, box2 in enumerate(boxes2):
                if jdx in matched_boxes2:
                    continue
                iou = _calculate_iou(box1, box2)
                if iou > best_iou:
                    best_iou = iou
                    best_match = (jdx, box2)

            if best_iou >= iou_threshold and best_match is not None:
                jdx, box2 = best_match
                text1 = normalize_text(box1[0])
                text2 = normalize_text(box2[0])
                dist = levenshtein_distance(text1, text2)
                norm_dist = dist / max(len(text1), len(text2), 1e-6)
                total_norm_dist += norm_dist
                matched_text_norm_dist += norm_dist
                counter += 1
                tp += 1
                matched_boxes2.add(jdx)
            else:
                # Unmatched pred box: penalise by its full text length.
                text1 = normalize_text(box1[0])
                dist = len(text1)
                norm_dist = dist / max(len(text1), 1e-6)
                total_norm_dist += norm_dist
                counter += 1

        # Unmatched reference boxes: penalise by their full text length.
        for jdx, box2 in enumerate(boxes2):
            if jdx not in matched_boxes2:
                text2 = normalize_text(box2[0])
                dist = len(text2)
                norm_dist = dist / max(len(text2), 1e-6)
                total_norm_dist += norm_dist
                counter += 1

        reward = 1 - total_norm_dist / (counter + 1e-6)
        clamp_reward = snap_to_nearest_integer(max(min(reward, 1), 0))

        fp = len(boxes1) - tp
        fn = len(boxes2) - tp
        avg_matched_text_sim = 1 - matched_text_norm_dist / (tp + 1e-6) if tp > 0 else 0.0

        reason = (
            f"Total norm dist: {total_norm_dist:.3f}, counter: {counter}; "
            f"reward: {clamp_reward:.3f}; "
            f"pred cnt: {len(boxes1)}, ref cnt: {len(boxes2)}, "
            f"TP: {tp}, FP: {fp}, FN: {fn}; "
            f"matched_text_sim: {avg_matched_text_sim:.3f}"
        )
        return True, clamp_reward, reason
    except BaseException as e:
        return False, -1, f"Error in _compare_spotting_results_soft: {e}"


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def process_spotting_task(response: str, ref_answer: str) -> dict:
    """Score a spotting task locally.

    Args:
        response: rollout response (prediction).
        ref_answer: reference answer.

    Returns:
        ``{"analysis", "is_valid", "reward"}`` dict.
    """
    # <answer>...</answer> unwrap, used by some prompt templates.
    if "<answer>" in response and "</answer>" in response:
        response = re.findall(r"<answer>(.+?)</answer>", response, flags=re.DOTALL)[0].strip()
    if "<answer>" in ref_answer and "</answer>" in ref_answer:
        ref_answer = re.findall(r"<answer>(.+?)</answer>", ref_answer, flags=re.DOTALL)[0].strip()

    resp_norm = response.strip().replace(".", "").replace("。", "")
    ref_norm = ref_answer.strip().replace(".", "").replace("。", "")
    if resp_norm == ref_norm or ("没有文字" in resp_norm and ref_norm == "") or ("没有文字" in resp_norm and "没有文字" in ref_norm):
        return {"analysis": "Responses are identical.", "is_valid": True, "reward": 1.0}

    resp_format = _detect_spotting_format(response)
    ref_format = _detect_spotting_format(ref_answer)

    try:
        if resp_format in _SPOTTING_FORMATS:
            boxes_pred = _parse_by_format(response, resp_format)
        else:
            if "没有文字" in response or "无文字" in response or "no text" in response.lower():
                return {"analysis": "No text detected in response.", "is_valid": True, "reward": 0}
            return {
                "analysis": f"Unknown format for response: {resp_format}, reward set to 0",
                "is_valid": False,
                "reward": 0,
            }

        if ref_format in _SPOTTING_FORMATS:
            boxes_ref = _parse_by_format(ref_answer, ref_format)
        else:
            return {"analysis": f"Unknown format for ref answer: {ref_format}", "is_valid": False, "reward": -1.0}

        if not boxes_pred or not boxes_ref:
            return {
                "analysis": f"Failed to parse boxes: pred({len(boxes_pred)}) ref({len(boxes_ref)})",
                "decision": "Answers are not identical.",
            }

        is_valid, reward, reason = _compare_spotting_results_soft(boxes_pred, boxes_ref)
        reason += f"; pred format: {resp_format}, ref format: {ref_format}"
        return {"analysis": reason, "is_valid": is_valid, "reward": reward}

    except Exception as e:
        return {"analysis": f"Error processing responses: {e!s}", "is_valid": False, "reward": -1.0}
