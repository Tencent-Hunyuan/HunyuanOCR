"""HunyuanOCR RL reward function for verl.

Verl's ``NaiveRewardManager`` calls the reward function once per rollout as::

    compute_score(
        data_source=...,
        solution_str=<rollout text>,
        ground_truth=<reference answer>,
        extra_info=<parquet extra_info dict>,
    )

This adapter dispatches the response to ``OCRScorer.process_scoring`` — rule
scoring for spotting / layout / parsing / ie / chart_deplot and judge-model
scoring for translation / vqa — and returns a score dict verl records under
``reward_extra_infos_dict``.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

# Make the bundled `reward` package importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

os.environ.setdefault("RM_SYSTEM_JUDGE_MODEL_NAME", "Qwen/Qwen3-30B-A3B")

_scorer = None


def _get_scorer():
    """Return a cached OCRScorer singleton."""
    global _scorer
    if _scorer is None:
        from reward.ocr_scorer import OCRScorer

        _scorer = OCRScorer(max_try=3)
    return _scorer


def compute_score(
    data_source: str = "",
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict | None = None,
    # Backward-compatible aliases: older verl builds / custom callers may still
    # use ``response`` / ``ref_answer``. Prefer the verl-native kwargs above.
    response: str | None = None,
    ref_answer: str | None = None,
    **_kwargs: Any,
) -> dict:
    """Verl reward entry point.

    Args:
        data_source: reward-router key from the parquet ``data_source`` column.
        solution_str: rollout response text (verl-native kwarg name).
        ground_truth: reference answer text (verl-native kwarg name).
        extra_info: parquet ``extra_info`` column. Reads ``task_type`` (required
            for reward dispatch), ``img_path`` (optional; kept so
            RLLoggingBoard can display it), and translation-only
            ``source_lang_text`` / ``target_lang``.
        response: legacy alias for ``solution_str``.
        ref_answer: legacy alias for ``ground_truth``.

    Returns:
        {"score", "acc", "reward_func", "analysis"} dict.
    """
    if solution_str is None:
        solution_str = response or ""
    if ground_truth is None:
        ground_truth = ref_answer or ""

    extra = extra_info or {}
    task_type = extra.get("task_type", "")

    scoring_obj = {
        "image_path": extra.get("img_path", ""),
        "prompt": extra.get("question", ""),
        "response": solution_str,
        "ref_answer": ground_truth,
        "task_type": task_type,
        "source_lang_text": extra.get("source_lang_text", ""),
        "target_lang": extra.get("target_lang", ""),
    }

    reward, analysis = -1.0, ""
    try:
        model_output = _get_scorer().process_scoring(scoring_obj)
        if model_output is not None:
            result = json.loads(model_output)
            raw = result.get("reward", result.get("decision", -1))
            try:
                reward = float(raw)
            except (TypeError, ValueError):
                reward = -1.0
            analysis = result.get("analysis", "")
        else:
            analysis = f"task_type={task_type} not implemented"
    except Exception as e:
        traceback.print_exc()
        analysis = f"Error: {e}"

    return {
        "score": reward,
        "acc": 1.0 if reward > 0.0 else 0.0,
        "reward_func": data_source or "unknown",
        "analysis": analysis,
    }
