"""Layout task rule scorer.

Entry point: ``score_layout(response, ref_answer) -> dict``.

Handles eight input combinations:
  * wire format : legacy ``hy-meta`` XML string OR new JSON string;
  * text field  : items either carry a ``text`` field (edit-distance mode) or
    omit it / leave it empty (F1 mode).

Scoring mode is decided by the reference:
  * If every reference layout has an empty / missing ``text`` field -> F1 mode
    (box + category only).
  * Otherwise -> normalised edit-distance mode (box + text).
"""

from __future__ import annotations

import json
import re

from ._shared import levenshtein_distance, normalize_text, snap_to_nearest_integer

# ---------------------------------------------------------------------------
# coordinate + format helpers
# ---------------------------------------------------------------------------


def _parse_coordinates(coord_str: str) -> list[list[int]]:
    """Parse a coordinate string into ``[[x1,y1],...,[x4,y4]]``.

    Accepts two wire forms:
      * ``(x1,y1),(x2,y2),(x3,y3),(x4,y4)``
      * ``[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]``
    """
    try:
        coord_str = coord_str.strip()
        if coord_str.startswith("(") and ")," in coord_str:
            pattern = r"\((\d+),(\d+)\)"
            matches = re.findall(pattern, coord_str)
            if matches:
                return [[int(x), int(y)] for x, y in matches]
        elif coord_str.startswith("[[") or coord_str.startswith("["):
            try:
                coords = eval(coord_str) if isinstance(coord_str, str) else coord_str
                if isinstance(coords, list) and len(coords) > 0:
                    return coords
            except Exception:
                pass
        if isinstance(coord_str, list):
            return coord_str
        return []
    except Exception as e:
        print(f"parse coordinates failed: {coord_str}, error: {e}")
        return []


def _detect_layout_format(answer: str) -> str:
    """Detect layout wire format.

    Returns:
      * ``'hy_meta'`` : ``text<hy-meta><layout>cat</layout><poly>...</poly></hy-meta>``
      * ``'json_str'``: ``[{"layout_type": ..., "bbox": ..., "text"?: ...}, ...]``
      * ``'unknown'`` : neither.
    """
    if not answer or not isinstance(answer, str):
        return "unknown"
    if (
        "<hy-meta>" in answer
        and "</hy-meta>" in answer
        and (("<poly>" in answer and "</poly>" in answer) or ("<quad>" in answer and "</quad>" in answer))
    ):
        return "hy_meta"
    stripped = answer.strip()
    if stripped.startswith("[") and "layout_type" in stripped and "bbox" in stripped:
        return "json_str"
    return "unknown"


def _parse_layout_hymeta(answer: str) -> list[dict]:
    """Parse hy-meta layout string into
    ``[{"text": str, "category": str, "location": [[x,y]*4]}, ...]``."""
    layout_infos: list[dict] = []
    if not answer:
        return layout_infos

    if "<quad>" in answer and "</quad>" in answer:
        pattern = r"([^<]*?)<hy-meta><layout>([^<]+)</layout><quad>([^<]+)</quad></hy-meta>"
    elif "<poly>" in answer and "</poly>" in answer:
        pattern = r"([^<]*?)<hy-meta><layout>([^<]+)</layout><poly>([^<]+)</poly></hy-meta>"
    else:
        return layout_infos

    matches: list[tuple[str, str, str]] = re.findall(pattern, answer)
    for text_content, category, coord_str in matches:
        text_content = text_content.strip()
        location = _parse_coordinates(coord_str)
        if location and len(location) >= 4:
            layout_infos.append({"text": text_content, "category": category.strip(), "location": location})
    return layout_infos


def _parse_layout_json(answer: str) -> list[dict]:
    """Parse the new JSON layout string. Items without a ``text`` field are
    normalised to ``text=""`` so the caller can decide between F1 mode and
    edit-distance mode uniformly."""
    layout_infos: list[dict] = []
    if not answer:
        return layout_infos
    try:
        data = json.loads(answer)
    except Exception:
        return layout_infos
    if not isinstance(data, list):
        return layout_infos

    for item in data:
        if not isinstance(item, dict):
            continue
        category = item.get("layout_type", "")
        bbox = item.get("bbox", "")
        text_content = item.get("text", "")
        if not isinstance(category, str) or not category.strip():
            continue
        if not isinstance(bbox, str) or not bbox.strip():
            continue
        location = _parse_coordinates(bbox)
        if location and len(location) >= 4:
            layout_infos.append(
                {
                    "text": text_content.strip() if isinstance(text_content, str) else "",
                    "category": category.strip(),
                    "location": location,
                }
            )
    return layout_infos


def _parse_by_format(answer: str, fmt: str) -> list[dict]:
    if fmt == "hy_meta":
        return _parse_layout_hymeta(answer)
    if fmt == "json_str":
        return _parse_layout_json(answer)
    return []


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _calculate_box_area(box: list[list[int]]) -> float:
    """Shoelace-formula polygon area for a 4-point quadrilateral."""
    if len(box) != 4:
        return 0.0
    x = [point[0] for point in box]
    y = [point[1] for point in box]
    area = 0.5 * abs(sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4)))
    return area


def _calculate_intersection_area(box1: list[list[int]], box2: list[list[int]]) -> float:
    """Approximate intersection area by axis-aligned bounding boxes derived
    from each quadrilateral. Good enough for layout IoU."""
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    x1_min = min(point[0] for point in box1)
    x1_max = max(point[0] for point in box1)
    y1_min = min(point[1] for point in box1)
    y1_max = max(point[1] for point in box1)
    x2_min = min(point[0] for point in box2)
    x2_max = max(point[0] for point in box2)
    y2_min = min(point[1] for point in box2)
    y2_max = max(point[1] for point in box2)
    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0
    return (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)


def _calculate_layout_iou(box1: list[list[int]], box2: list[list[int]]) -> float:
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    intersection = _calculate_intersection_area(box1, box2)
    area1 = _calculate_box_area(box1)
    area2 = _calculate_box_area(box2)
    union = area1 + area2 - intersection
    if union == 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# category canonicalisation
# ---------------------------------------------------------------------------


def _normalize_category(category: str) -> str:
    category = category.lower().strip()
    normalized = category.replace(" ", "_").replace("-", "_")
    if normalized in ("para_title", "paratitle"):
        return "para_title"
    if normalized in ("table_of_contents", "tableofcontents") or normalized.startswith("table_of"):
        return "table_of_contents"
    if normalized in ("table_title", "tabletitle"):
        return "table_title"
    if normalized == "table":
        return "table"
    if normalized in ("figure_title", "figuretitle", "chart_title", "charttitle"):
        return "figure_title"
    if normalized in ("figure", "chart"):
        return "figure"
    if normalized in ("paragraph", "paragraph_span", "paragraphspan", "para"):
        return "paragraph"
    if normalized in ("section_title", "sectiontitle", "title", "heading"):
        return "section_title"
    if normalized in ("header", "page_header", "page_footer", "footer"):
        return normalized.replace("page_", "")
    return normalized


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def _compare_layout_results_soft(
    layouts1: list[dict],
    layouts2: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[bool, float, str]:
    """Greedy IoU + category matching, then F1 (text-free ref) or normalised
    edit-distance (with-text ref) scoring."""
    try:
        is_f1_mode = all(not str(layout.get("text", "") or "").strip() for layout in layouts2)

        matched_boxes2: set[int] = set()
        total_norm_dist = 0.0
        counter = 0
        tp = 0

        for _, layout1 in enumerate(layouts1):
            best_iou = -1
            best_match = None
            for jdx, layout2 in enumerate(layouts2):
                if jdx in matched_boxes2:
                    continue
                pred_category = _normalize_category(layout1["category"])
                ref_category = _normalize_category(layout2["category"])
                iou = _calculate_layout_iou(layout1["location"], layout2["location"])
                if iou > best_iou and pred_category == ref_category:
                    best_iou = iou
                    best_match = (jdx, layout2)

            if best_iou >= iou_threshold and best_match is not None:
                jdx, layout2 = best_match
                text1 = normalize_text(layout1["text"])
                text2 = normalize_text(layout2["text"])
                dist = levenshtein_distance(text1, text2)
                norm_dist = dist / max(len(text1), len(text2), 1e-6)
                total_norm_dist += norm_dist
                counter += 1
                tp += 1
                matched_boxes2.add(jdx)
            else:
                # Unmatched pred: penalise by its full text length.
                text1 = normalize_text(layout1["text"])
                dist = len(text1)
                norm_dist = dist / max(len(text1), 1e-6)
                total_norm_dist += norm_dist
                counter += 1

        for jdx, layout2 in enumerate(layouts2):
            if jdx not in matched_boxes2:
                text2 = normalize_text(layout2["text"])
                dist = len(text2)
                norm_dist = dist / max(len(text2), 1e-6)
                total_norm_dist += norm_dist
                counter += 1

        if is_f1_mode:
            fp = len(layouts1) - tp
            fn = len(layouts2) - tp
            precision = tp / len(layouts1) if len(layouts1) > 0 else 0.0
            recall = tp / len(layouts2) if len(layouts2) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            reward = f1
            reason = f"F1 mode: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, TP={tp}, FP={fp}, FN={fn}"
        else:
            reward = 1 - total_norm_dist / (counter + 1e-6)
            reason = (
                f"Total norm distance: {total_norm_dist:.3f}, counter: {counter}; "
                f"reward: {reward:.3f}; "
                f"len (res_bbox): {len(layouts1)}, len (ref_bbox): {len(layouts2)}"
            )

        clamp_reward = snap_to_nearest_integer(max(min(reward, 1), 0))
        return True, clamp_reward, reason

    except BaseException as e:
        return False, -1.0, f"Error in _compare_layout_results_soft: {e}"


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def process_layout_task(response_a: str, response_b: str) -> dict:
    """Score a layout task locally.

    Args:
        response_a: rollout response (prediction).
        response_b: reference answer.

    Returns:
        ``{"analysis", "is_valid", "reward"}`` dict.
    """
    try:
        format_a = _detect_layout_format(response_a)
        format_b = _detect_layout_format(response_b)
        layouts_a = _parse_by_format(response_a, format_a)
        layouts_b = _parse_by_format(response_b, format_b)

        is_valid, reward, reason = _compare_layout_results_soft(layouts_a, layouts_b)
        reason += (
            f"; pred format: {format_a}, ref format: {format_b}; pred cnt: {len(layouts_a)}, ref cnt: {len(layouts_b)}"
        )
        return {"analysis": reason, "is_valid": is_valid, "reward": reward}

    except Exception as e:
        return {"analysis": f"Error processing responses: {e!s}", "is_valid": False, "reward": -1.0}
