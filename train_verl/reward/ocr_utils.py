"""OCR task type enum + reward-score shaping helpers."""

import json
import re
from enum import Enum


class TaskType(Enum):
    """Supported OCR task types (each with an English + Chinese alias).

    Chart deplot uses a single prompt ("Parse the chart in the image ..."); the
    model output is one of: markdown table / mermaid flowchart / markdown
    unordered list.
    """

    SPOTTING = "Spotting"
    PARSING = "Parsing"
    IE = "IE"
    TRANSLATION = "Translation"
    VQA = "VQA"
    LAYOUT = "Layout"
    CHART_DEPLOT = "chart_deplot"

    SPOTTING_ZH = "检测识别"
    PARSING_ZH = "通用解析"
    IE_ZH = "信息抽取"
    TRANSLATION_ZH = "翻译"
    VQA_ZH = "视觉问答"
    LAYOUT_ZH = "版面分析"
    CHART_DEPLOT_ZH = "图表解析"

    @staticmethod
    def _matches(task_type: str, en: str, zh: str) -> bool:
        """A task_type string matches a task when it contains the English
        alias (case-insensitive) or the Chinese alias (verbatim). This keeps
        legacy variants like ``"IE-rule"`` / ``"rule_based_parsing"`` valid
        without an explicit alias list."""
        return en.lower() in task_type.lower() or zh in task_type

    @classmethod
    def is_spotting(cls, t: str) -> bool:
        return cls._matches(t, cls.SPOTTING.value, cls.SPOTTING_ZH.value)

    @classmethod
    def is_parsing(cls, t: str) -> bool:
        return cls._matches(t, cls.PARSING.value, cls.PARSING_ZH.value)

    @classmethod
    def is_ie(cls, t: str) -> bool:
        return cls._matches(t, cls.IE.value, cls.IE_ZH.value)

    @classmethod
    def is_layout(cls, t: str) -> bool:
        return cls._matches(t, cls.LAYOUT.value, cls.LAYOUT_ZH.value)

    @classmethod
    def is_translate(cls, t: str) -> bool:
        return cls._matches(t, cls.TRANSLATION.value, cls.TRANSLATION_ZH.value)

    @classmethod
    def is_vqa(cls, t: str) -> bool:
        return cls._matches(t, cls.VQA.value, cls.VQA_ZH.value)

    @classmethod
    def is_chart_deplot(cls, t: str) -> bool:
        return cls._matches(t, cls.CHART_DEPLOT.value, cls.CHART_DEPLOT_ZH.value)


def piecewise_linear_transform(score: float) -> float:
    """Stretch the 2-4 range of raw 0-5 judge scores to spread rewards evenly.

    Piecewise linear:
      * [0, 2] -> [0.0, 0.2]  (slope 0.1)
      * [2, 4] -> [0.2, 0.8]  (slope 0.3, the stretched band)
      * [4, 5] -> [0.8, 1.0]  (slope 0.2)
    """
    if score <= 2:
        return score * 0.1
    if score <= 4:
        return 0.2 + (score - 2) * 0.3
    return 0.8 + (score - 4) * 0.2


# ```json``` fenced block, generic ``` fenced block, "评分过程" scoring brace.
_JSON_BLOCK_PATTERNS = (
    r"```json\s*(\{.*?\})\s*```",
    r"```\s*(\{.*?\})\s*```",
    r'\{[^\{]*"评分过程".*?\}',
)


def extract_json_from_response(response: str | None):
    """Extract a JSON object from a model response (best-effort).

    Tries fenced blocks / the scoring-brace pattern first, then falls back to
    parsing the whole response as JSON. Returns the parsed dict/list, or
    ``None`` if nothing parses.
    """
    if not response:
        return None
    candidates = []
    for pattern in _JSON_BLOCK_PATTERNS:
        candidates.extend(re.findall(pattern, response, re.DOTALL))
    candidates.append(response)  # raw-parse fallback
    for text in candidates:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    return None
