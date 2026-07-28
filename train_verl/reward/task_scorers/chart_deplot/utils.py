"""Chart Deplot 任务的规则评分。

任务输入：用户用统一 prompt 让模型"解析图中的图表，对于流程图使用 Mermaid 格式表示，
其他图表使用 Markdown 格式表示"。模型输出和参考答案在训练集中分布如下三种：
    - Markdown 表格        (~7543/7660，含 ### 标题前导)
    - Mermaid 流程图       (~43/7660，含 ```mermaid 围栏)
    - Markdown 无序列表    (~71/7660，含 ```markdown 围栏)

评分规则（简化版，针对训练场景）：
    1) 按参考答案的真实格式（mermaid / md_table / md_list 之一）决定走哪条评测路径；
       若 reference 也无法识别为这三类之一，视为脏数据：is_valid=False, reward=-1.0。
    2) 探测 prediction 的格式；若与 reference 不同类，**直接 reward=0**，is_valid=True，
       analysis 里写明双方类别。
    3) 同类时按对应评测算法打分，取返回 13 元组中的 map_slight 作为 reward：
        - md_table  -> csv_eval (先用 normalize_to_csv 转成内部 CSV)
        - mermaid   -> flowchart_eval
        - md_list   -> tree_eval

评分算法实现位于同目录的 eval.py（已自包含，无外部依赖）。
"""

from __future__ import annotations

import re
from typing import Any

from .eval import (
    csv_eval,
    flowchart_eval,
    is_markdown_list,
    is_markdown_table,
    is_mermaid,
    normalize_to_csv,
    tree_eval,
)

# ============================================================
# 格式探测
# ============================================================


def detect_chart_format(text: str) -> str:
    """返回 'mermaid' / 'md_list' / 'md_table' / 'unknown'。

    优先级：mermaid > md_list > md_table > unknown。
    这样可以避免 ref/pred 中带 "### 标题" 等噪声时被误判（mermaid/md_list 都使用
    更强的特征：```mermaid 围栏 + flowchart 声明 / `- ` 列表行计数）。
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    if is_mermaid(text):
        return "mermaid"
    if is_markdown_list(text):
        return "md_list"
    if is_markdown_table(text):
        return "md_table"
    # 兜底：再用一个比较宽松的 markdown 表格检测（无完整分隔线，但有 ≥2 行以 '|' 开头）
    if _is_loose_markdown_table(text):
        return "md_table"
    return "unknown"


def _is_loose_markdown_table(text: str) -> bool:
    """宽松的 markdown 表格识别（用于 detect 兜底）。"""
    has_sep = re.search(r"^\s*\|?\s*:?-{2,}", text, re.MULTILINE) is not None
    pipe_rows = sum(1 for ln in text.split("\n") if ln.strip().startswith("|"))
    return has_sep and pipe_rows >= 2


# ============================================================
# 评分主入口
# ============================================================


# csv_eval / tree_eval / flowchart_eval 返回的 13 元组顺序：
#   (em, map_strict, map_slight, map_high,
#    ap_50_strict, ap_75_strict, ap_90_strict,
#    ap_50_slight, ap_75_slight, ap_90_slight,
#    ap_50_high,   ap_75_high,   ap_90_high)
_SCORE_KEYS = (
    "em",
    "map_strict",
    "map_slight",
    "map_high",
    "ap_50_strict",
    "ap_75_strict",
    "ap_90_strict",
    "ap_50_slight",
    "ap_75_slight",
    "ap_90_slight",
    "ap_50_high",
    "ap_75_high",
    "ap_90_high",
)


def _scores_tuple_to_dict(scores_tuple: tuple) -> dict[str, float]:
    return {k: float(v) for k, v in zip(_SCORE_KEYS, scores_tuple)}


def _truncate(text: str, n: int = 300) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text if len(text) <= n else text[:n] + "...<truncated>"


def _score_by_format(prediction: str, reference: str, fmt: str) -> tuple[float, dict[str, Any]]:
    """根据已确认的同类 fmt 调用对应评测算法。

    返回 (map_slight 作为 reward, 详细信息 dict)。
    """
    info: dict[str, Any] = {"eval_path": fmt}

    if fmt == "md_table":
        # 把 markdown 表格（含 ### 标题等前后文）归一化为内部 CSV 后比对
        pred_csv = normalize_to_csv(prediction)
        ref_csv = normalize_to_csv(reference)
        info["pred_csv_preview"] = _truncate(pred_csv)
        info["ref_csv_preview"] = _truncate(ref_csv)
        if not pred_csv or not ref_csv:
            info["error"] = "empty_csv_after_normalize"
            return 0.0, info
        scores_tuple, eval_logs = csv_eval([pred_csv], [ref_csv], easy=1)
        score_dict = _scores_tuple_to_dict(scores_tuple)
        info["score_components"] = score_dict
        if eval_logs:
            info["eval_logs"] = list(eval_logs)
        return score_dict["map_slight"], info

    if fmt == "mermaid":
        scores_tuple, eval_logs = flowchart_eval([prediction], [reference], easy=1)
        score_dict = _scores_tuple_to_dict(scores_tuple)
        info["score_components"] = score_dict
        if eval_logs:
            info["eval_logs"] = list(eval_logs)
        return score_dict["map_slight"], info

    if fmt == "md_list":
        scores_tuple, eval_logs = tree_eval([prediction], [reference], easy=1)
        score_dict = _scores_tuple_to_dict(scores_tuple)
        info["score_components"] = score_dict
        if eval_logs:
            info["eval_logs"] = list(eval_logs)
        return score_dict["map_slight"], info

    info["error"] = f"unhandled_format: {fmt}"
    return 0.0, info


def process_chart_deplot_task(response: str, ref_answer: str) -> dict[str, Any]:
    """对 chart_deplot 任务的一对 (模型回复, 参考答案) 打分。

    Args:
        response: 模型回复
        ref_answer: 参考答案
    Returns:
        {
            "analysis": dict 形式的诊断信息（pred/ref 类别、是否一致、各类分数、组件分等），
            "is_valid": 是否产出可用 reward（参考答案脏 → False；其它情况均为 True），
            "reward":   最终 reward，map_slight；格式不匹配为 0.0；ref 不可识别为 -1.0
        }
    """
    analysis: dict[str, Any] = {}

    # 1) 双方文本兜底
    if not isinstance(response, str):
        response = "" if response is None else str(response)
    if not isinstance(ref_answer, str):
        ref_answer = "" if ref_answer is None else str(ref_answer)

    pred_format = detect_chart_format(response)
    ref_format = detect_chart_format(ref_answer)

    analysis["pred_format"] = pred_format
    analysis["ref_format"] = ref_format
    analysis["format_matched"] = (pred_format == ref_format) and ref_format != "unknown"
    analysis["pred_preview"] = _truncate(response)
    analysis["ref_preview"] = _truncate(ref_answer)

    # 2) ref 脏数据（极少数样本只有标题没数据等）：丢弃该样本
    if ref_format == "unknown":
        analysis["reason"] = (
            "reference format is unknown (not one of mermaid / md_table / md_list); "
            "drop this sample from training signal"
        )
        return {"analysis": analysis, "is_valid": False, "reward": -1.0}

    # 3) 类别不匹配：reward=0，仍是有效样本（给训练负向信号）
    if pred_format != ref_format:
        analysis["reason"] = (
            f"format mismatch: pred={pred_format} vs ref={ref_format}; "
            "reward=0 (valid negative signal)"
        )
        analysis["reward"] = 0.0
        return {"analysis": analysis, "is_valid": True, "reward": 0.0}

    # 4) 同类：调对应评测函数
    try:
        reward, score_info = _score_by_format(response, ref_answer, ref_format)
    except Exception as e:
        analysis["error"] = f"scoring failed: {e!r}"
        return {"analysis": analysis, "is_valid": False, "reward": -1.0}

    analysis.update(score_info)
    if not isinstance(reward, (int, float)):
        reward = 0.0
    reward = max(0.0, min(1.0, float(reward)))
    analysis["reward"] = reward
    analysis["reason"] = (
        f"format matched ({ref_format}); reward=map_slight={reward:.4f}"
    )
    return {"analysis": analysis, "is_valid": True, "reward": reward}
