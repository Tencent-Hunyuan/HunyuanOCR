"""IE (information extraction) 任务字段级评分。

打分入口：process_ie_task(response, ref_answer) -> dict
返回字典字段：
  - is_valid : 是否成功打分（True/False）
  - reward   : 字段级 edit-distance similarity 平均分，∈[0,1]；解析失败/空 ref 返回 -1.0
  - analysis : 详细打分明细的 JSON 字符串（含 ref/pred 是否解析成功、字段数、exact_match
               总数、similarity 总和、每字段 gt/pred/exact_match/similarity 等）
  - mode     : "ie_json"（标识该样本走 JSON 字段级评分）
  - need_fallback : 仅当 ref 不是 JSON dict 时为 True，提示上层 fallback 到 parsing。

设计说明：
1. ref / pred 解析容错：先 json.loads → 剥 ```json ... ``` 外壳再试 → 抠首个 {...} 块。
2. 单字段评分：normalize 后 exact_match (0/1) + 归一化 Levenshtein 相似度 (∈[0,1])。
3. 最终 reward = sum(similarity) / n_fields，n_fields = |gt_keys ∪ pred_keys|；
   - pred 解析失败时按 {} 处理 → 全字段 0；
   - gt 解析失败时返回 need_fallback=True，由上层决定走 parsing。
"""

from __future__ import annotations

import json
import re
from typing import Any

from Levenshtein import distance as levenshtein_distance


# ============================================================================
# 字段值归一化 + Levenshtein 距离 + 相似度（与 eval_ie.py 保持一致
# ============================================================================
def _normalize(s: Any) -> str:
    """字段值预处理：strip + 去换行 + 去空格 + lower。非字符串先 str 化（dict/list 走 json.dumps）。"""
    if s is None:
        return ""
    if not isinstance(s, str):
        if isinstance(s, (int, float, bool)):
            s = str(s)
        else:
            s = json.dumps(s, ensure_ascii=False, sort_keys=True)
    return s.strip().replace("\n", " ").replace(" ", "").lower()


def _similarity(a: Any, b: Any) -> float:
    """归一化 edit-distance 相似度，值域 [0, 1]；双方 normalize 后都为空时返回 1.0。"""
    a = _normalize(a)
    b = _normalize(b)
    if not a and not b:
        return 1.0
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return 1.0 - levenshtein_distance(a, b) / denom


# ============================================================================
# JSON 解析（自动剥 ```json ... ``` markdown 包装）
# ============================================================================
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _parse_json(s: Any) -> Any | None:
    """尝试解析 JSON：直接 loads → 剥 ```json ... ``` 后 loads → 抠首个 {...} 块再 loads。
    解析失败统一返回 None。"""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    m = _JSON_FENCE_RE.match(s)
    if m:
        inner = m.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ============================================================================
# 字段级评分
# ============================================================================
def _evaluate_fields(gt_dict: dict[str, Any], pred_dict: dict[str, Any]) -> dict[str, Any]:
    """按字段评分：取 gt_keys ∪ pred_keys，逐字段算 exact_match + similarity。

    Returns:
        dict 包含：
          - n_fields : int
          - n_exact  : int
          - sum_sim  : float
          - exact_acc: float（n_exact / n_fields，0 字段时 0.0）
          - avg_sim  : float（sum_sim / n_fields，0 字段时 0.0）
          - fields   : list[dict] 每字段详情（field/gt/pred/exact_match/similarity）
    """
    keys = set(gt_dict.keys()) | set(pred_dict.keys())
    field_records: list[dict[str, Any]] = []
    n_exact = 0
    sum_sim = 0.0
    for k in sorted(keys):
        gt_v = gt_dict.get(k, "")
        pr_v = pred_dict.get(k, "")
        gt_n = _normalize(gt_v)
        pr_n = _normalize(pr_v)
        exact = 1 if gt_n == pr_n else 0
        sim = _similarity(gt_v, pr_v)
        n_exact += exact
        sum_sim += sim
        field_records.append(
            {
                "field": k,
                "gt": gt_v
                if isinstance(gt_v, (str, int, float, bool)) or gt_v is None
                else json.dumps(gt_v, ensure_ascii=False),
                "pred": pr_v
                if isinstance(pr_v, (str, int, float, bool)) or pr_v is None
                else json.dumps(pr_v, ensure_ascii=False),
                "exact_match": exact,
                "similarity": round(sim, 6),
            }
        )
    n_fields = len(keys)
    return {
        "n_fields": n_fields,
        "n_exact": n_exact,
        "sum_sim": round(sum_sim, 6),
        "exact_acc": round(n_exact / n_fields, 6) if n_fields > 0 else 0.0,
        "avg_sim": round(sum_sim / n_fields, 6) if n_fields > 0 else 0.0,
        "fields": field_records,
    }


# ============================================================================
# 入口
# ============================================================================
def process_ie_task(response: str, ref_answer: str) -> dict[str, Any]:
    """IE 任务字段级评分入口。

    判定流：
      1) ref 解析为 JSON dict 失败 → 返回 {need_fallback: True, ...}，由上层走 parsing。
      2) ref 是 JSON dict 但 pred 解析失败 → 视 pred 为 {}，全字段 0 计算 reward（is_valid=True）。
      3) 双方都是 JSON dict → 字段级评分，reward = avg_sim ∈[0,1]。

    Args:
        response   : 模型输出（str，可能含 ```json ... ``` 包装）
        ref_answer : 参考答案（str，同上）

    Returns:
        dict: {is_valid, reward, analysis(JSON 字符串), mode, need_fallback?}
    """
    ref_obj = _parse_json(ref_answer) if isinstance(ref_answer, str) else None
    if not isinstance(ref_obj, dict):
        # ref 不是 JSON dict → 让上层走 parsing
        return {
            "analysis": json.dumps(
                {
                    "mode": "ie_fallback_parsing",
                    "reason": "ref_answer is not a JSON dict, fallback to parsing",
                    "ref_preview": (ref_answer or "")[:200] if isinstance(ref_answer, str) else None,
                },
                ensure_ascii=False,
            ),
            "is_valid": False,
            "reward": -1.0,
            "mode": "ie_fallback_parsing",
            "need_fallback": True,
        }

    pred_obj = _parse_json(response) if isinstance(response, str) else None
    pred_ok = isinstance(pred_obj, dict)
    pred_dict: dict[str, Any] = pred_obj if pred_ok else {}

    detail = _evaluate_fields(ref_obj, pred_dict)
    reward = detail["avg_sim"]

    analysis_obj = {
        "mode": "ie_json",
        "ref_parse_ok": True,
        "pred_parse_ok": pred_ok,
        "n_fields": detail["n_fields"],
        "n_exact": detail["n_exact"],
        "exact_acc": detail["exact_acc"],
        "sum_sim": detail["sum_sim"],
        "avg_sim": detail["avg_sim"],
        "reward": round(float(reward), 6),
        "fields": detail["fields"],
    }
    if not pred_ok:
        analysis_obj["pred_preview"] = (response or "")[:200] if isinstance(response, str) else None

    return {
        "analysis": json.dumps(analysis_obj, ensure_ascii=False),
        "is_valid": True,
        "reward": float(reward),
        "mode": "ie_json",
    }
