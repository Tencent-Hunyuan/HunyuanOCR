"""Chart Deplot 自包含评测内核。

本模块从 ChartArena 中移植 chart_deplot 任务实际需要的最小代码集合，
不再依赖 ChartArena 任何外部模块；删除了 JSON/CSV/Python 代码/SVG/Pie/
DOT/PlantUML/D2/Diagrams 等与本任务无关的解析器与兜底逻辑。

对外暴露三件事（语义与 ChartArena 原版完全一致）：

1) 格式探测
   - is_mermaid(text)         : 是否为 mermaid 流程图
   - is_markdown_list(text)   : 是否为 markdown 多级无序列表
   - is_markdown_table(text)  : 是否为标准 markdown 表格
   - is_html_table(text)      : 是否为 HTML <table>
   - is_pipe_table(text)      : 是否为无分隔线的 pipe 表格
   - normalize_to_csv(text)   : 将 markdown 表格 / HTML 表格 / pipe 表格
                                统一归一化为内部 CSV（` \\t ` 分列、` \\n ` 分行）

2) 三类评测算法（输入 list[str]，返回 (13 元组, eval_logs)）
   - csv_eval(pred_csv_list, ref_csv_list, easy=1)   : md_table 路径
   - tree_eval(pred_list, ref_list, easy=1)          : md_list 路径
   - flowchart_eval(pred_list, ref_list, easy=1)     : mermaid 路径

返回 13 元组顺序（与原版一致）：
    (em, map_strict, map_slight, map_high,
     ap_50_strict, ap_75_strict, ap_90_strict,
     ap_50_slight, ap_75_slight, ap_90_slight,
     ap_50_high,   ap_75_high,   ap_90_high)
"""

from __future__ import annotations

import html as _html
import logging
import re
from collections import Counter
from collections.abc import Callable
from typing import Any, Literal

import Levenshtein
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ============================================================
# 公共工具：代码围栏剥离
# ============================================================


def _strip_code_fence(text: str, lang: str | None = None) -> str:
    """去掉 ```xxx ... ``` 代码块围栏，返回纯文本。"""
    if not text:
        return ""
    t = text.strip()
    if lang:
        pattern = rf"^```(?:{lang}|{lang.upper()})?\s*\n?(.*?)\n?```\s*$"
    else:
        pattern = r"^```[a-zA-Z0-9_\-]*\s*\n?(.*?)\n?```\s*$"
    m = re.match(pattern, t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


# ============================================================
# Markdown / HTML / pipe 表格 → 内部 CSV
# ============================================================


_HTML_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
_HTML_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL_FULL_RE = re.compile(r"<(t[hd])([^>]*)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_HTML_CAPTION_RE = re.compile(r"<caption[^>]*>.*?</caption>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SPAN_ATTR_RE = re.compile(r"""(colspan|rowspan)\s*=\s*["']?\s*(\d+)\s*["']?""", re.IGNORECASE)


def _clean_html_cell(cell: str) -> str:
    cell = re.sub(r"<br\s*/?>", " ", cell, flags=re.IGNORECASE)
    cell = _HTML_TAG_RE.sub("", cell)
    cell = _html.unescape(cell)
    cell = _WS_RE.sub(" ", cell).strip()
    return cell


def is_html_table(text: str) -> bool:
    """快速判断是否包含 HTML <table>。"""
    if not text:
        return False
    if "<table" not in text and "<TABLE" not in text:
        return False
    if "<tr" not in text and "<TR" not in text:
        return False
    return not ("<td" not in text and "<TD" not in text and "<th" not in text and "<TH" not in text)


def html_table_to_csv(text: str) -> str:
    """将包含 HTML <table> 的文本转为内部 CSV（` \\t ` 列、` \\n ` 行）。

    - 剥离 ```markdown / ```html 代码栅栏
    - 忽略 <caption>
    - 仅取第一个 <table>
    - 支持 colspan / rowspan（值复制方式展开）
    - 跳过整行空 cell
    """
    if not text:
        return ""
    t = _strip_code_fence(text, "markdown")
    t = _strip_code_fence(t, "html")
    t = _HTML_CAPTION_RE.sub("", t)

    m = _HTML_TABLE_RE.search(t)
    if not m:
        return ""
    table_body = m.group(1)

    grid: list[dict[int, str]] = []
    pending: dict[int, tuple[int, str]] = {}

    for tr_m in _HTML_TR_RE.finditer(table_body):
        tr_body = tr_m.group(1)
        row: dict[int, str] = {}
        col = 0

        def _skip_occupied(c: int) -> int:
            while c in pending and pending[c][0] > 0:
                remaining, value = pending[c]
                row[c] = value  # noqa: B023  called synchronously within the current iteration; row does not escape
                pending[c] = (remaining - 1, value)
                if pending[c][0] <= 0:
                    del pending[c]
                c += 1
            return c

        col = _skip_occupied(col)

        for cell_m in _HTML_CELL_FULL_RE.finditer(tr_body):
            attrs = cell_m.group(2) or ""
            inner = cell_m.group(3)
            colspan = 1
            rowspan = 1
            for sp in _SPAN_ATTR_RE.finditer(attrs):
                key = sp.group(1).lower()
                val = max(1, int(sp.group(2)))
                if key == "colspan":
                    colspan = val
                elif key == "rowspan":
                    rowspan = val
            value = _clean_html_cell(inner)

            for dc in range(colspan):
                row[col] = value
                if rowspan > 1:
                    pending[col] = (rowspan - 1, value)
                col += 1
            col = _skip_occupied(col)

        if row:
            grid.append(row)

    if not grid:
        return ""

    max_col = max(max(r.keys()) for r in grid) + 1
    rows_out: list[str] = []
    for r in grid:
        cells = [r.get(c, "") for c in range(max_col)]
        if all(not c for c in cells):
            continue
        rows_out.append(" \\t ".join(cells))

    if not rows_out:
        return ""
    return " \\n ".join(rows_out)


def is_markdown_table(text: str) -> bool:
    """判断是否为 Markdown 表格（含分隔线 + 至少 2 行 '|' 包裹的数据行）。"""
    if not text:
        return False
    lines = text.strip().split("\n")
    has_pipe_line = False
    has_separator = False
    for line in lines:
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            has_pipe_line = True
        if re.match(r"^\|[\s\-:]+\|$", line) or (
            line.startswith("|") and "-" in line and all(c in "-|: " for c in line)
        ):
            has_separator = True
    return has_pipe_line and has_separator


def markdown_to_csv(md_text: str) -> str:
    """Markdown 表格 → 内部 CSV (\\t/\\n 分隔)。"""
    lines = md_text.strip().split("\n")
    csv_rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[\s\-:]+\|$", line) or ("-" in line and all(c in "-|: " for c in line)):
            continue
        cells = line.strip("|").split("|")
        cells = [cell.strip() for cell in cells]
        csv_rows.append(" \\t ".join(cells))
    return " \\n ".join(csv_rows)


_ASCII_BORDER_RE = re.compile(r"^[+\-|\s:=]+$")


def is_pipe_table(text: str) -> bool:
    """无分隔线、无首尾竖线的 pipe 表格识别。"""
    if not text or not text.strip():
        return False
    if is_markdown_table(text):
        return False

    raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    data_lines = [ln for ln in raw_lines if not _ASCII_BORDER_RE.match(ln)]
    if len(data_lines) < 2:
        return False

    col_counts: list[int] = []
    for ln in data_lines:
        if "|" not in ln:
            continue
        body = ln.strip("|").strip()
        col_counts.append(len(body.split("|")))

    if len(col_counts) < max(2, int(len(data_lines) * 0.7)):
        return False
    if not col_counts or max(col_counts) < 2:
        return False

    cc = Counter(col_counts)
    _top_cnt, top_n = cc.most_common(1)[0]
    return not top_n < max(2, int(len(col_counts) * 0.7))


def pipe_table_to_csv(text: str) -> str:
    """无分隔线 pipe 表格 → 内部 CSV。"""
    if not text:
        return ""
    raw_lines = [ln.strip() for ln in text.split("\n")]
    rows: list[list[str]] = []
    for ln in raw_lines:
        if not ln:
            continue
        if _ASCII_BORDER_RE.match(ln):
            continue
        if "|" not in ln:
            continue
        body = ln.strip("|").strip()
        cells = [c.strip() for c in body.split("|")]
        if all(not c for c in cells):
            continue
        rows.append(cells)

    if len(rows) < 2:
        return ""

    cc = Counter(len(r) for r in rows)
    target = cc.most_common(1)[0][0]
    if target < 2:
        return ""

    aligned: list[list[str]] = []
    for cells in rows:
        if len(cells) < target:
            cells = cells + [""] * (target - len(cells))
        elif len(cells) > target:
            cells = cells[:target]
        aligned.append(cells)

    # 横向 2 行表格转置（首行非数值标签，第二行几乎全数值）
    if len(aligned) == 2 and target >= 3:
        head, data = aligned[0], aligned[1]

        def _is_numeric(s: str) -> bool:
            if not s:
                return False
            t2 = s.strip().rstrip("%").replace(",", "")
            if not t2:
                return False
            try:
                float(t2)
                return True
            except ValueError:
                return False

        head_nonnum = sum(1 for c in head if c and not _is_numeric(c))
        data_num = sum(1 for c in data if _is_numeric(c))
        if head_nonnum >= max(2, int(0.8 * target)) and data_num >= max(2, int(0.8 * target)):
            aligned = [[h, d] for h, d in zip(head, data)]

    if len(aligned) < 2:
        return ""
    out_rows = [" \\t ".join(cells) for cells in aligned]
    return " \\n ".join(out_rows)


def is_csv_format(text: str) -> bool:
    """是否已经是内部 \\t/\\n 分隔的 CSV。"""
    return "\\t" in text and "\\n" in text


def normalize_to_csv(text: str) -> str:
    """统一将 markdown 表格 / HTML 表格 / pipe 表格 / 内部 CSV 归一化为内部 CSV。"""
    if not text or not text.strip():
        return ""
    text = text.strip()

    if is_html_table(text):
        csv_text = html_table_to_csv(text)
        if csv_text:
            return csv_text

    if is_markdown_table(text):
        return markdown_to_csv(text)

    if is_pipe_table(text):
        csv_text = pipe_table_to_csv(text)
        if csv_text:
            return csv_text

    if is_csv_format(text):
        return text

    return text


# ============================================================
# 通用工具：日志去重、数值判断
# ============================================================


def _dedup_logs(logs: list[str]) -> list[str]:
    if not logs:
        return logs
    deduped: list[str] = []
    prev_msg = logs[0]
    count = 1
    for msg in logs[1:]:
        if msg == prev_msg:
            count += 1
        else:
            deduped.append(prev_msg if count == 1 else f"{prev_msg} [Repeat x{count}]")
            prev_msg = msg
            count = 1
    deduped.append(prev_msg if count == 1 else f"{prev_msg} [Repeat x{count}]")
    return deduped


def _is_int(val: Any) -> bool:
    try:
        int(val)
        return True
    except (ValueError, TypeError):
        return False


def _is_float(val: Any) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


# ============================================================
# csv_eval：基于三元组的 CSV/表格评测（SCRM）
# ============================================================


_FULLWIDTH_TO_HALFWIDTH: dict[str, str] = {
    "：": ":",
    "（": "(",
    "）": ")",
    "，": ",",
    "；": ";",
    "％": "%",
}


def _normalize_fullwidth(text: str) -> str:
    if not text:
        return text
    for fw, hw in _FULLWIDTH_TO_HALFWIDTH.items():
        if fw in text:
            text = text.replace(fw, hw)
    return text


_LATEX_SUPERSCRIPTS = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "n": "ⁿ",
    "+": "⁺",
    "-": "⁻",
}


def _normalize_latex(text: str) -> str:
    if "$" not in text and "\\" not in text:
        return text
    result = re.sub(r"\$\s*(.*?)\s*\$", r"\1", text)
    result = re.sub(r"\\text\{([^}]*)\}", r"\1", result)
    result = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", result)
    result = re.sub(r"\\textbf\{([^}]*)\}", r"\1", result)

    def _sup_repl(m: re.Match) -> str:
        content = m.group(1)
        return "".join(_LATEX_SUPERSCRIPTS.get(c, c) for c in content)

    result = re.sub(r"\^\{([^}]*)\}", _sup_repl, result)
    result = re.sub(r"\^(\d)", lambda m: _LATEX_SUPERSCRIPTS.get(m.group(1), m.group(1)), result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


_BOXPLOT_STAT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[_\-]?(?:下|第一)四分位数?\s*(?:[\(（]\s*Q1\s*[\)）])?", re.IGNORECASE), "-Q1"),
    (re.compile(r"[_\-]?Q1\b", re.IGNORECASE), "-Q1"),
    (re.compile(r"[_\-]?第二四分位数?\s*(?:[\(（]\s*Q2\s*[\)）])?", re.IGNORECASE), "-中位数"),
    (re.compile(r"[_\-]?中位数\s*(?:[\(（]\s*Q2\s*[\)）])?", re.IGNORECASE), "-中位数"),
    (re.compile(r"[_\-]?Q2\b", re.IGNORECASE), "-中位数"),
    (re.compile(r"[_\-]?(?:上|第三)四分位数?\s*(?:[\(（]\s*Q3\s*[\)）])?", re.IGNORECASE), "-Q3"),
    (re.compile(r"[_\-]?Q3\b", re.IGNORECASE), "-Q3"),
    (re.compile(r"[_\-]?\bMedian\b", re.IGNORECASE), "-中位数"),
    (re.compile(r"[_\-]?最小值", re.IGNORECASE), "-最小值"),
    (re.compile(r"[_\-]?\bMin\b", re.IGNORECASE), "-最小值"),
    (re.compile(r"[_\-]?最大值", re.IGNORECASE), "-最大值"),
    (re.compile(r"[_\-]?\bMax\b", re.IGNORECASE), "-最大值"),
]


def _normalize_triple_key(text: str, norm_logs: list[str] | None = None) -> str:
    original = text = text.strip()
    if not text:
        return text

    text = _normalize_fullwidth(text)
    text = _normalize_latex(text)
    # µ (U+00B5) → μ (U+03BC)
    text = text.replace("\u00b5", "\u03bc")

    for pattern, replacement in _BOXPLOT_STAT_PATTERNS:
        match = pattern.search(text)
        if match:
            prefix = text[: match.start()].rstrip("-_— ").rstrip(" ").rstrip("(（").rstrip(" ")
            suffix = text[match.end() :].lstrip("-_— ").lstrip(" ").lstrip(")）").lstrip(" ")
            if prefix and suffix:
                text = prefix + replacement + "-" + suffix
            elif prefix:
                text = prefix + replacement
            elif suffix:
                text = suffix + replacement
            else:
                text = replacement
            break

    if text != original:
        logger.info(f"箱线图归一化: '{original}' -> '{text}'")
        if norm_logs is not None:
            norm_logs.append(f"归一化: '{original}' -> '{text}'")
    return text


def _is_empty_header(header: list[str]) -> bool:
    """判断 header 是否为无表头表格。"""
    if all(h.strip() == "" for h in header):
        return True
    return bool(len(header) >= 2 and header[0].strip() != "" and all(h.strip() == "" for h in header[1:]))


def _csv2triples(
    csv_text: str,
    separator: str = " \\t ",
    delimiter: str = " \\n ",
    norm_logs: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """CSV 字符串 → 三元组列表 (entity, header, value)。

    注意：内部 CSV 以 ' \\t ' 分列、' \\n ' 分行，**首单元为空时字符串以 ' \\t ' 开头**
    （首字符是空格）。这里只用 rstrip 去末尾换行/空白，**保留前导空白**，避免吞掉
    首单元的空 token，导致首列 cell 数比数据行少 1、整张表被错位解析（行/列 entity 互换）。
    """
    lines: list[str] = csv_text.rstrip("\n\r \t").split(delimiter)
    header: list[str] = lines[0].split(separator)

    if _is_empty_header(header):
        logger.info("检测到空表头，按无表头模式解析")
        triples: list[tuple[str, str, str]] = []
        for line in lines:
            if not line:
                continue
            values: list[str] = line.split(separator)
            if all(v.strip() == "" for v in values):
                continue
            if len(values) >= 2 and values[0].strip() != "" and all(v.strip() == "" for v in values[1:]):
                continue
            if len(values) >= 2:
                entity = values[0].strip()
                entity = _normalize_triple_key(entity, norm_logs=norm_logs)
                for col_idx in range(1, len(values)):
                    value = _normalize_fullwidth(values[col_idx].strip()).replace("%", "").replace("$", "")
                    triples.append((entity, "", value))
            elif len(values) == 1 and values[0].strip():
                key = _normalize_triple_key(values[0].strip(), norm_logs=norm_logs)
                triples.append((key, "", ""))
        return triples

    triples = []
    for line in lines[1:]:
        if not line:
            continue
        values = line.split(separator)
        entity = values[0]
        for i in range(1, len(values)):
            if i >= len(header):
                break
            value = _normalize_fullwidth(values[i].strip()).replace("%", "").replace("$", "")
            norm_entity = _normalize_triple_key(entity.strip(), norm_logs=norm_logs)
            norm_header = _normalize_triple_key(header[i].strip(), norm_logs=norm_logs)
            key0, key1 = sorted([norm_entity, norm_header])
            triples.append((key0, key1, value))
    return triples


def _process_triplets(triplets: list[tuple[str, str, str]]) -> list[tuple]:
    new_triplets = []
    for triplet in triplets:
        if len(triplet) > 2:
            if _is_int(triplet[2]) or _is_float(triplet[2]):
                triplet_temp = (triplet[0].lower(), triplet[1].lower(), float(triplet[2]))
            else:
                triplet_temp = (triplet[0].lower(), triplet[1].lower(), triplet[2].lower())
        else:
            triplet_temp = (triplet[0].lower(), triplet[1].lower(), "no meaning")
        new_triplets.append(triplet_temp)
    return new_triplets


def _intersection_with_tolerance(
    a: list[tuple[str, str, Any]],
    b: list[tuple[str, str, Any]],
    tol_word: int,
    tol_num: float,
):
    a, b, c = set(a), set(b), set()
    for elem1 in a:
        for elem2 in b:
            if _is_float(elem1[-1]) and _is_float(elem2[-1]):
                if (Levenshtein.distance("".join(elem1[:-1]), "".join(elem2[:-1])) <= tol_word) and (
                    abs(elem1[-1] - elem2[-1]) / (elem2[-1] + 0.000001) <= tol_num
                ):
                    c.add(elem1)
            else:
                if (
                    Levenshtein.distance(
                        "".join([str(i) for i in elem1]),
                        "".join([str(j) for j in elem2]),
                    )
                    <= tol_word
                ):
                    c.add(elem1)
    return list(c)


def _union_with_tolerance(
    a: list[tuple[str, str, Any]],
    b: list[tuple[str, str, Any]],
    tol_word: int,
    tol_num: float,
):
    c = set(a) | set(b)
    d = set(a) & set(b)
    e = _intersection_with_tolerance(a, b, tol_word, tol_num)
    f = set(e)
    g = c - (f - d)
    return list(g)


# 需要后处理过滤的三元组类别
_EXTRA_TRIPLE_FILTERS: list[tuple[list[str], str]] = [
    (["异常值", "outlier", "离群"], "异常值"),
    (["离散点", "离散值", "discrete", "scatter"], "离散点"),
]


def _make_keyword_triple_checker(keywords: list[str]) -> Callable:
    def _checker(triple: tuple) -> bool:
        text = str(triple[0]) + str(triple[1])
        return any(kw in text for kw in keywords)

    return _checker


def _postprocess_extra_triples(
    pred_triple_list: list[list[tuple]],
    label_triple_list: list[list[tuple]],
    is_target_triple: Callable,
    tag: str,
    logs: list[str] | None = None,
    force_filter: bool = True,
) -> tuple[list[list[tuple]], list[list[tuple]]]:
    new_pred_list = []
    for idx, (pred, label) in enumerate(zip(pred_triple_list, label_triple_list)):
        label_target_count = sum(1 for t in label if is_target_triple(t))
        pred_target_triples = [t for t in pred if is_target_triple(t)]
        pred_target_count = len(pred_target_triples)

        if label_target_count == 0 and pred_target_count > 0:
            entity_target_counts: Counter = Counter()
            for t in pred_target_triples:
                if is_target_triple((str(t[0]), "", "")):
                    entity = str(t[1])
                else:
                    entity = str(t[0])
                entity_target_counts[entity] += 1

            unique_counts = set(entity_target_counts.values())
            if len(unique_counts) <= 1:
                filtered_pred = [t for t in pred if not is_target_triple(t)]
                msg = (
                    f"移除{pred_target_count}条{tag}三元组"
                    f"({len(entity_target_counts)}个entity×{next(iter(unique_counts)) if unique_counts else 0})"
                )
                logger.info(f"[样本 {idx}] {msg}")
                if logs is not None:
                    logs.append(msg)
                new_pred_list.append(filtered_pred)
            else:
                if force_filter:
                    filtered_pred = [t for t in pred if not is_target_triple(t)]
                    msg = f"{tag}数量不一致({dict(entity_target_counts)})，强制移除{pred_target_count}条{tag}三元组"
                    logger.warning(f"[样本 {idx}] {msg}")
                    if logs is not None:
                        logs.append(msg)
                    new_pred_list.append(filtered_pred)
                else:
                    msg = f"{tag}数量不一致({dict(entity_target_counts)})，跳过过滤"
                    logger.warning(f"[样本 {idx}] {msg}")
                    if logs is not None:
                        logs.append(msg)
                    new_pred_list.append(pred)
        else:
            new_pred_list.append(pred)

    return new_pred_list, label_triple_list


def csv_eval(
    predictions: list[str],
    references: list[str],
    easy: Literal[0, 1] = 1,
    separator: str = " \\t ",
    delimiter: str = " \\n ",
) -> tuple[tuple, list[str]]:
    """CSV/表格三元组评测，返回 13 元组分数和评测日志。"""
    predictions_arr = np.asarray(predictions)
    labels_arr = np.asarray(references)
    eval_logs: list[str] = []

    # ---- 0. CSV 层面的无表头对齐（含 A/B/C/D 四种分支） ----
    def _parse_rows(csv_str: str) -> list[list[str]]:
        if not csv_str:
            return []
        trimmed = csv_str.rstrip("\n\r")
        if not trimmed.strip():
            return []
        return [ln.split(separator) for ln in trimmed.split(delimiter)]

    aligned_predictions: list[str] = []
    aligned_labels: list[str] = []
    for pred_csv, ref_csv in zip(predictions_arr, labels_arr):
        pred_rows = _parse_rows(pred_csv)
        ref_rows = _parse_rows(ref_csv)

        ref_header = ref_rows[0] if ref_rows else []
        pred_header = pred_rows[0] if pred_rows else []
        ref_headerless = _is_empty_header(ref_header) if ref_header else False
        pred_headerless = _is_empty_header(pred_header) if pred_header else False

        if ref_headerless and not pred_headerless and len(pred_rows) > 1:
            # A: ref 无表头, pred 有表头 → 去掉 pred 表头
            num_cols = len(pred_header)
            empty_header = separator.join([" "] * num_cols)
            rest = pred_csv.rstrip("\n\r").split(delimiter)[1:]
            new_pred_csv = empty_header + delimiter + delimiter.join(rest)
            msg = "去掉pred表头行(ref无表头)"
            logger.info(msg)
            eval_logs.append(msg)
            aligned_predictions.append(new_pred_csv)
            aligned_labels.append(ref_csv)
        elif pred_headerless and not ref_headerless and len(ref_rows) > 1:
            # B: pred 无表头, ref 有表头 → 把 ref 改为空表头
            pred_data_rows = len(pred_rows) - 1
            ref_data_rows = len(ref_rows) - 1
            pred_ncols = len(pred_rows[1]) if len(pred_rows) > 1 else len(pred_header)
            ref_ncols = len(ref_rows[1]) if len(ref_rows) > 1 else len(ref_header)

            row_diff_ok = abs(pred_data_rows - ref_data_rows) <= 1
            col_match_ok = pred_ncols == ref_ncols and ref_ncols >= 2

            if row_diff_ok and col_match_ok:
                num_cols = len(ref_header)
                empty_header = separator.join([" "] * num_cols)
                rest = ref_csv.rstrip("\n\r").split(delimiter)[1:]
                new_ref_csv = empty_header + delimiter + delimiter.join(rest)
                msg = "去掉ref表头行(pred无表头, 行列数一致)"
                logger.info(msg)
                eval_logs.append(msg)
                aligned_predictions.append(pred_csv)
                aligned_labels.append(new_ref_csv)
            else:
                aligned_predictions.append(pred_csv)
                aligned_labels.append(ref_csv)
        else:
            # C/D: pred 比 ref 多一列行头 / 两边都是 2 列但列名不同
            try_branch_c = not pred_headerless and not ref_headerless and len(pred_rows) > 1 and len(ref_rows) > 1
            handled = False
            if try_branch_c:
                pred_data_rows = len(pred_rows) - 1
                ref_data_rows = len(ref_rows) - 1
                pred_ncols = len(pred_rows[1])
                ref_ncols = len(ref_rows[1])

                ref_first_col_empty = len(ref_header) > 0 and ref_header[0].strip() == ""
                row_diff_ok = abs(pred_data_rows - ref_data_rows) <= 1
                col_match_ok = (pred_ncols == ref_ncols + 1) and ref_ncols >= 2

                pred_first_col_named = len(pred_header) > 0 and pred_header[0].strip() != ""

                def _is_auto_index_column(pred_rows=pred_rows, pred_header=pred_header) -> bool:
                    if len(pred_rows) < 2:
                        return False
                    if len(pred_header) == 0 or pred_header[0].strip() != "":
                        return False
                    first_col_vals: list[str] = []
                    for r in pred_rows[1:]:
                        if not r:
                            continue
                        first_col_vals.append(r[0].strip())
                    if not first_col_vals:
                        return False
                    try:
                        ints = [int(v) for v in first_col_vals]
                    except ValueError:
                        return False
                    if len(ints) < 2:
                        return False
                    start = ints[0]
                    if start not in (0, 1):
                        return False
                    return ints == list(range(start, start + len(ints)))

                trigger_c1 = row_diff_ok and col_match_ok and ref_first_col_empty and pred_first_col_named
                trigger_c2 = row_diff_ok and col_match_ok and _is_auto_index_column()

                if trigger_c1 or trigger_c2:
                    raw_pred_lines = pred_csv.rstrip("\n\r").split(delimiter)
                    new_pred_lines = []
                    for ln in raw_pred_lines:
                        cells = ln.split(separator)
                        if len(cells) <= 1:
                            new_pred_lines.append(ln)
                        else:
                            new_pred_lines.append(separator.join(cells[1:]))
                    new_pred_csv = delimiter.join(new_pred_lines)
                    tag = "有名首列" if trigger_c1 else "pandas自动行号"
                    msg = f"去掉pred首列({tag}, 比 ref 多一列)"
                    logger.info(msg)
                    eval_logs.append(msg)
                    aligned_predictions.append(new_pred_csv)
                    aligned_labels.append(ref_csv)
                    handled = True

                if not handled and pred_ncols == 2 and ref_ncols == 2 and row_diff_ok:
                    pred_first_empty = len(pred_header) > 0 and pred_header[0].strip() == ""
                    pred_col_named = len(pred_header) >= 2 and pred_header[1].strip() != ""
                    ref_col_named = len(ref_header) >= 2 and ref_header[1].strip() != ""
                    first_col_sym = pred_first_empty == ref_first_col_empty
                    first_col_pred_named_ref_empty = (not pred_first_empty) and ref_first_col_empty

                    if pred_col_named and ref_col_named and (first_col_sym or first_col_pred_named_ref_empty):
                        empty_header_2 = separator.join([" "] * 2)
                        pred_rest = pred_csv.rstrip("\n\r").split(delimiter)[1:]
                        ref_rest = ref_csv.rstrip("\n\r").split(delimiter)[1:]
                        new_pred_csv = empty_header_2 + delimiter + delimiter.join(pred_rest)
                        new_ref_csv = empty_header_2 + delimiter + delimiter.join(ref_rest)
                        msg = (
                            f"抹空表头(2列表格, pred_col='{pred_header[1].strip()}' "
                            f"vs ref_col='{ref_header[1].strip()}')"
                        )
                        logger.info(msg)
                        eval_logs.append(msg)
                        aligned_predictions.append(new_pred_csv)
                        aligned_labels.append(new_ref_csv)
                        handled = True

            if not handled:
                aligned_predictions.append(pred_csv)
                aligned_labels.append(ref_csv)

    predictions_arr = np.asarray(aligned_predictions)
    labels_arr = np.asarray(aligned_labels)

    # ---- 1. 解析三元组 ----
    pred_triple_list: list[list[tuple]] = []
    for it in predictions_arr:
        pred_triple_temp = _csv2triples(it, separator=separator, delimiter=delimiter, norm_logs=eval_logs)
        pred_triple_list.append(_process_triplets(pred_triple_temp))

    label_triple_list: list[list[tuple]] = []
    for it in labels_arr:
        label_triple_temp = _csv2triples(it, separator=separator, delimiter=delimiter, norm_logs=eval_logs)
        label_triple_list.append(_process_triplets(label_triple_temp))

    # ---- 1.2 后处理：表头公共子串对齐 ----
    _BOXPLOT_SUFFIXES = {"-最小值", "-q1", "-中位数", "-q3", "-最大值"}

    for idx in range(len(pred_triple_list)):
        pred_triples = pred_triple_list[idx]
        label_triples = label_triple_list[idx]
        changed_triples = False

        # 场景 A：箱线图公共前缀对齐
        ref_prefixes: set[str] = set()
        for t in label_triples:
            for key in (t[0], t[1]):
                for suffix in _BOXPLOT_SUFFIXES:
                    if key.endswith(suffix) and len(key) > len(suffix):
                        ref_prefixes.add(key[: -len(suffix)])

        if len(ref_prefixes) == 1:
            common_prefix = ref_prefixes.pop()
            has_bare_suffix = any(t[0] in _BOXPLOT_SUFFIXES or t[1] in _BOXPLOT_SUFFIXES for t in pred_triples)
            if has_bare_suffix:
                new_pred = []
                for k0, k1, v in pred_triples:
                    c = False
                    if k0 in _BOXPLOT_SUFFIXES:
                        k0 = common_prefix + k0
                        c = True
                    if k1 in _BOXPLOT_SUFFIXES:
                        k1 = common_prefix + k1
                        c = True
                    if c:
                        k0, k1 = sorted([k0, k1])
                    new_pred.append((k0, k1, v))
                pred_triples = new_pred
                changed_triples = True
                msg = f"箱线图表头前缀对齐: pred补上公共前缀'{common_prefix}'"
                logger.info(msg)
                eval_logs.append(msg)

        # 场景 B：通用公共后缀对齐
        ref_keys: set[str] = set()
        for t in label_triples:
            for key in (t[0], t[1]):
                if key and not key.replace(".", "").replace("-", "").isdigit():
                    ref_keys.add(key)
        if len(ref_keys) >= 2:
            ref_key_list = sorted(ref_keys)
            min_len = min(len(k) for k in ref_key_list)
            common_suffix_len = 0
            for i in range(1, min_len + 1):
                if all(k[-i] == ref_key_list[0][-i] for k in ref_key_list):
                    common_suffix_len = i
                else:
                    break
            common_suffix = ref_key_list[0][-common_suffix_len:] if common_suffix_len > 0 else ""
            common_suffix = common_suffix.lstrip()
            if common_suffix and common_suffix[0] in ("-", "_"):
                ref_stems = {k[: len(k) - common_suffix_len].rstrip("-_ ") for k in ref_keys}
                pred_keys: set[str] = set()
                for t in pred_triples:
                    for key in (t[0], t[1]):
                        if key and not key.replace(".", "").replace("-", "").isdigit():
                            pred_keys.add(key)
                if ref_stems and pred_keys and pred_keys <= ref_stems and not (pred_keys & ref_keys):
                    new_pred = []
                    for k0, k1, v in pred_triples:
                        c = False
                        if k0 in ref_stems and k0 not in ref_keys:
                            k0 = k0 + common_suffix
                            c = True
                        if k1 in ref_stems and k1 not in ref_keys:
                            k1 = k1 + common_suffix
                            c = True
                        if c:
                            k0, k1 = sorted([k0, k1])
                        new_pred.append((k0, k1, v))
                    pred_triples = new_pred
                    changed_triples = True
                    msg = f"表头后缀对齐: pred补上公共后缀'{common_suffix}'"
                    logger.info(msg)
                    eval_logs.append(msg)

        # 场景 C：通用公共前缀对齐
        if not changed_triples:
            ref_keys_c: set[str] = set()
            for t in label_triples:
                for key in (t[0], t[1]):
                    if key and not key.replace(".", "").replace("-", "").isdigit():
                        ref_keys_c.add(key)
            if len(ref_keys_c) >= 2:
                ref_key_list_c = sorted(ref_keys_c)
                min_len_c = min(len(k) for k in ref_key_list_c)
                common_prefix_len = 0
                for i in range(min_len_c):
                    if all(k[i] == ref_key_list_c[0][i] for k in ref_key_list_c):
                        common_prefix_len = i + 1
                    else:
                        break
                common_prefix = ref_key_list_c[0][:common_prefix_len] if common_prefix_len > 0 else ""
                common_prefix = common_prefix.rstrip()
                if common_prefix and common_prefix[-1] in ("-", "_"):
                    ref_suffixes = {k[common_prefix_len:].lstrip("-_ ") for k in ref_keys_c}
                    pred_keys_c: set[str] = set()
                    for t in pred_triples:
                        for key in (t[0], t[1]):
                            if key and not key.replace(".", "").replace("-", "").isdigit():
                                pred_keys_c.add(key)
                    if ref_suffixes and pred_keys_c and pred_keys_c <= ref_suffixes and not (pred_keys_c & ref_keys_c):
                        new_pred = []
                        for k0, k1, v in pred_triples:
                            c = False
                            if k0 in ref_suffixes and k0 not in ref_keys_c:
                                k0 = common_prefix + k0
                                c = True
                            if k1 in ref_suffixes and k1 not in ref_keys_c:
                                k1 = common_prefix + k1
                                c = True
                            if c:
                                k0, k1 = sorted([k0, k1])
                            new_pred.append((k0, k1, v))
                        pred_triples = new_pred
                        changed_triples = True
                        msg = f"表头前缀对齐: pred补上公共前缀'{common_prefix}'"
                        logger.info(msg)
                        eval_logs.append(msg)

        if changed_triples:
            pred_triple_list[idx] = pred_triples

    # ---- 1.5 后处理：过滤 label 中不存在的异常值 / 离散点等三元组 ----
    for keywords, tag in _EXTRA_TRIPLE_FILTERS:
        checker = _make_keyword_triple_checker(keywords)
        pred_triple_list, label_triple_list = _postprocess_extra_triples(
            pred_triple_list, label_triple_list, checker, tag, logs=eval_logs
        )

    # ---- 2. 容差参数 ----
    tolerance_params: dict[str, tuple[int, float]] = {
        "strict": (0, 0 if easy == 1 else 0.1),
        "slight": (2, 0.05 if easy == 1 else 0.3),
        "high": (5, 0.1 if easy == 1 else 0.5),
    }

    def _compute_sim_list(tol_word: int, tol_num: float) -> list[float]:
        sim_list: list[float] = []
        for pred, label in zip(pred_triple_list, label_triple_list):
            intersection = _intersection_with_tolerance(pred, label, tol_word=tol_word, tol_num=tol_num)
            union = _union_with_tolerance(pred, label, tol_word=tol_word, tol_num=tol_num)
            sim = len(intersection) / len(union) if len(union) > 0 else 0.0
            sim_list.append(sim)
        return sim_list

    sim_lists: dict[str, list[float]] = {}
    for tol_name, (tol_word, tol_num) in tolerance_params.items():
        sim_lists[tol_name] = _compute_sim_list(tol_word, tol_num)

    def _get_ap(sim_list: list[float], sim_threshold: float) -> float:
        if not sim_list:
            return 0.0
        return len([s for s in sim_list if s >= sim_threshold]) / len(sim_list)

    map_strict = 0.0
    map_slight = 0.0
    map_high = 0.0
    for sim_threshold in np.arange(0.5, 1, 0.05):
        map_strict += _get_ap(sim_lists["strict"], sim_threshold) / 10
        map_slight += _get_ap(sim_lists["slight"], sim_threshold) / 10
        map_high += _get_ap(sim_lists["high"], sim_threshold) / 10

    em = _get_ap(sim_lists["strict"], 1.0)
    ap_50_strict = _get_ap(sim_lists["strict"], 0.5)
    ap_75_strict = _get_ap(sim_lists["strict"], 0.75)
    ap_90_strict = _get_ap(sim_lists["strict"], 0.90)
    ap_50_slight = _get_ap(sim_lists["slight"], 0.5)
    ap_75_slight = _get_ap(sim_lists["slight"], 0.75)
    ap_90_slight = _get_ap(sim_lists["slight"], 0.90)
    ap_50_high = _get_ap(sim_lists["high"], 0.5)
    ap_75_high = _get_ap(sim_lists["high"], 0.75)
    ap_90_high = _get_ap(sim_lists["high"], 0.90)

    scores = (
        em,
        map_strict,
        map_slight,
        map_high,
        ap_50_strict,
        ap_75_strict,
        ap_90_strict,
        ap_50_slight,
        ap_75_slight,
        ap_90_slight,
        ap_50_high,
        ap_75_high,
        ap_90_high,
    )
    return scores, _dedup_logs(eval_logs)


# ============================================================
# tree_eval：Markdown 无序列表的路径集合匹配
# ============================================================


def is_markdown_list(text: str) -> bool:
    """判断文本是否为 Markdown 多级无序列表格式。"""
    if not text or not text.strip():
        return False
    lines = text.strip().split("\n")
    list_line_count = 0
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        if re.match(r"^\s*<[^>]+>", stripped):
            continue
        if re.match(r"^(\s*)- ", stripped):
            list_line_count += 1
    return list_line_count >= 2


def _parse_indent_level(line: str) -> tuple[int, str]:
    stripped = line.rstrip()
    content_start = 0
    spaces = 0
    for ch in stripped:
        if ch == " ":
            spaces += 1
            content_start += 1
        elif ch == "\t":
            spaces += 2
            content_start += 1
        else:
            break
    rest = stripped[content_start:]
    if rest.startswith("- "):
        content = rest[2:]
    elif rest.startswith("-"):
        content = rest[1:].lstrip()
    else:
        content = rest
    level = spaces // 2
    return level, content.strip()


def _parse_markdown_list(text: str) -> list[dict]:
    lines = text.strip().split("\n")
    list_lines: list[tuple[int, str]] = []
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        if re.match(r"^\s*<[^>]+>", stripped):
            continue
        if re.match(r"^(\s*)- ", stripped) or re.match(r"^(\s*)-\S", stripped):
            level, content = _parse_indent_level(stripped)
            if content:
                list_lines.append((level, content))

    if not list_lines:
        return []

    roots: list[dict] = []
    stack: list[tuple[int, dict]] = []
    for level, content in list_lines:
        node = {"name": content, "children": []}
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            parent = stack[-1][1]
            parent["children"].append(node)
        else:
            roots.append(node)
        stack.append((level, node))

    return roots


_SEPARATOR_PATTERN = re.compile(
    r"(?<=\S)"
    r"\s*"
    r"(?:——|—|->|-->|--|-|:|：)"
    r"\s*"
    r"(?=\S)"
)


def _split_separator_nodes(node: dict) -> dict:
    node["children"] = [_split_separator_nodes(child) for child in node["children"]]
    name = node["name"]
    parts = _SEPARATOR_PATTERN.split(name)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return node
    current = {"name": parts[-1], "children": node["children"]}
    for i in range(len(parts) - 2, 0, -1):
        current = {"name": parts[i], "children": [current]}
    return {"name": parts[0], "children": [current]}


def _merge_single_child_nodes(node: dict) -> dict:
    node["children"] = [_merge_single_child_nodes(child) for child in node["children"]]
    while len(node["children"]) == 1:
        child = node["children"][0]
        node["name"] = node["name"] + "——" + child["name"]
        node["children"] = child["children"]
    return node


def _normalize_tree(roots: list[dict]) -> list[dict]:
    import copy

    roots = copy.deepcopy(roots)
    roots = [_split_separator_nodes(root) for root in roots]
    roots = [_merge_single_child_nodes(root) for root in roots]
    return roots


def _tree_to_paths(roots: list[dict]) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = []

    def _dfs(node: dict, current_path: tuple[str, ...]):
        new_path = current_path + (node["name"],)
        paths.append(new_path)
        for child in node["children"]:
            _dfs(child, new_path)

    for root in roots:
        _dfs(root, ())
    return paths


def _path_similarity(path_a: tuple[str, ...], path_b: tuple[str, ...]) -> float:
    str_a = " -> ".join(path_a).lower()
    str_b = " -> ".join(path_b).lower()
    return Levenshtein.ratio(str_a, str_b)


def _tree_hungarian_score_with_threshold(
    pred_paths: list[tuple[str, ...]],
    ref_paths: list[tuple[str, ...]],
    sim_threshold: float = 0.0,
) -> float:
    if not pred_paths and not ref_paths:
        return 1.0
    if not pred_paths or not ref_paths:
        return 0.0

    n_pred = len(pred_paths)
    n_ref = len(ref_paths)
    sim_matrix = np.zeros((n_pred, n_ref))
    for i, p_path in enumerate(pred_paths):
        for j, r_path in enumerate(ref_paths):
            sim_matrix[i, j] = _path_similarity(p_path, r_path)

    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_sim_sum = 0.0
    for r, c in zip(row_ind, col_ind):
        if sim_matrix[r, c] >= sim_threshold:
            matched_sim_sum += sim_matrix[r, c]
    score = matched_sim_sum / max(n_pred, n_ref)
    return score


def tree_eval(
    predictions: list[str],
    references: list[str],
    easy: Literal[0, 1] = 1,
) -> tuple[tuple, list[str]]:
    """Markdown 无序列表评测，返回 13 元组分数与日志（始终空）。"""
    pred_paths_list: list[list[tuple[str, ...]]] = []
    ref_paths_list: list[list[tuple[str, ...]]] = []

    for pred_text, ref_text in zip(predictions, references):
        pred_roots = _parse_markdown_list(pred_text)
        pred_roots = _normalize_tree(pred_roots)
        pred_paths = _tree_to_paths(pred_roots)

        ref_roots = _parse_markdown_list(ref_text)
        ref_roots = _normalize_tree(ref_roots)
        ref_paths = _tree_to_paths(ref_roots)

        pred_paths_list.append(pred_paths)
        ref_paths_list.append(ref_paths)

    tolerance_params: dict[str, float] = {
        "strict": 1.0 if easy == 1 else 0.95,
        "slight": 0.85 if easy == 1 else 0.75,
        "high": 0.6 if easy == 1 else 0.5,
    }

    def _compute_sim_list(path_sim_threshold: float) -> list[float]:
        sim_list: list[float] = []
        for pred_paths, ref_paths in zip(pred_paths_list, ref_paths_list):
            score = _tree_hungarian_score_with_threshold(pred_paths, ref_paths, sim_threshold=path_sim_threshold)
            sim_list.append(score)
        return sim_list

    sim_lists: dict[str, list[float]] = {}
    for tol_name, threshold in tolerance_params.items():
        sim_lists[tol_name] = _compute_sim_list(threshold)

    def _get_ap(sim_list: list[float], sim_threshold: float) -> float:
        if not sim_list:
            return 0.0
        return len([s for s in sim_list if s >= sim_threshold]) / len(sim_list)

    map_strict = 0.0
    map_slight = 0.0
    map_high = 0.0
    for sim_threshold in np.arange(0.5, 1, 0.05):
        map_strict += _get_ap(sim_lists["strict"], sim_threshold) / 10
        map_slight += _get_ap(sim_lists["slight"], sim_threshold) / 10
        map_high += _get_ap(sim_lists["high"], sim_threshold) / 10

    em = _get_ap(sim_lists["strict"], 1.0)
    ap_50_strict = _get_ap(sim_lists["strict"], 0.5)
    ap_75_strict = _get_ap(sim_lists["strict"], 0.75)
    ap_90_strict = _get_ap(sim_lists["strict"], 0.90)
    ap_50_slight = _get_ap(sim_lists["slight"], 0.5)
    ap_75_slight = _get_ap(sim_lists["slight"], 0.75)
    ap_90_slight = _get_ap(sim_lists["slight"], 0.90)
    ap_50_high = _get_ap(sim_lists["high"], 0.5)
    ap_75_high = _get_ap(sim_lists["high"], 0.75)
    ap_90_high = _get_ap(sim_lists["high"], 0.90)

    scores = (
        em,
        map_strict,
        map_slight,
        map_high,
        ap_50_strict,
        ap_75_strict,
        ap_90_strict,
        ap_50_slight,
        ap_75_slight,
        ap_90_slight,
        ap_50_high,
        ap_75_high,
        ap_90_high,
    )
    return scores, []


# ============================================================
# flowchart_eval：Mermaid 流程图相似度评测
# ============================================================


def _strip_mermaid_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```\s*mermaid\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def is_mermaid(text: str) -> bool:
    """判断文本是否为 mermaid 流程图（含 flowchart/graph 声明 + 箭头）。"""
    if not text or not text.strip():
        return False
    cleaned = _strip_mermaid_fence(text)
    lines = cleaned.strip().split("\n")
    has_graph_decl = False
    has_arrow = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(flowchart|graph)\s+(TD|TB|LR|RL|BT)\b", stripped, re.IGNORECASE):
            has_graph_decl = True
        if re.search(r"-->|---|\.\->|==>|--\s*\w+\s*-->", stripped):
            has_arrow = True
    return has_graph_decl and has_arrow


def _clean_node_label(label: str) -> str:
    if not label:
        return ""
    s = re.sub(r"<\s*br\s*/?\s*>", " ", label, flags=re.IGNORECASE)
    s = re.sub(r"\\[nrl]", " ", s)
    s = re.sub(r"[\n\r\t]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_node_def(token: str) -> tuple[str, str]:
    """解析单个节点定义 token，返回 (node_id, label)。"""
    token = token.strip()
    if not token:
        return "", ""

    m = re.match(
        r"^([A-Za-z_]\w*)\s*"
        r"(?:"
        r'\(\(\s*"([^"]*)"\s*\)\)'
        r'|\(\s*\[\s*"([^"]*)"\s*\]\s*\)'
        r'|\[\s*\[\s*"([^"]*)"\s*\]\s*\]'
        r'|\[\s*\(\s*"([^"]*)"\s*\)\s*\]'
        r'|\{\s*\{\s*"([^"]*)"\s*\}\s*\}'
        r'|\[\s*"([^"]*)"\s*\]'
        r'|\(\s*"([^"]*)"\s*\)'
        r'|\{\s*"([^"]*)"\s*\}'
        r'|>\s*"([^"]*)"\s*\]'
        r'|\[\s*/\s*"([^"]*)"\s*/\s*\]'
        r'|\[\s*\\\s*"([^"]*)"\s*\\\s*\]'
        r"|\(\(\s*([^)]*?)\s*\)\)"
        r"|\(\s*\[\s*([^\]]*?)\s*\]\s*\)"
        r"|\[\s*\[\s*([^\]]*?)\s*\]\s*\]"
        r"|\[\s*\(\s*([^)]*?)\s*\)\s*\]"
        r"|\{\s*\{\s*([^}]*?)\s*\}\s*\}"
        r"|\[\s*([^\]]*)\s*\]"
        r"|\(\s*([^)]*)\s*\)"
        r"|\{\s*([^}]*)\s*\}"
        r"|>\s*([^\]]*)\s*\]"
        r")?$",
        token,
    )

    if m:
        node_id = m.group(1)
        label = None
        if m.lastindex and m.lastindex >= 2:
            for i in range(2, m.lastindex + 1):
                if m.group(i) is not None:
                    label = m.group(i).strip()
                    break
        if label is None:
            label = node_id
        return node_id, _clean_node_label(label)

    m_id = re.match(r"^([A-Za-z_]\w*)$", token)
    if m_id:
        return m_id.group(1), m_id.group(1)

    return token, _clean_node_label(token)


def _parse_edge_line(line: str) -> list[tuple[str, str, str]] | None:
    arrow_patterns = [
        r'-->\s*\|\s*"([^"]*)"\s*\|',
        r"-->\s*\|([^|]*)\|",
        r'==>\s*\|\s*"([^"]*)"\s*\|',
        r"==>\s*\|([^|]*)\|",
        r'-\.->\s*\|\s*"([^"]*)"\s*\|',
        r"-\.->\s*\|([^|]*)\|",
        r'==\s*"([^"]*)"\s*==>',
        r'--\s*"([^"]*)"\s*-->',
        r"--\s*([^->\s][^->]*?)\s*-->",
        r"==\s*([^=>\s][^=>]*?)\s*==>",
        r"==>",
        r"-->",
        r"-\.->",
        r"---",
    ]
    if not re.search(r"-->|---|==>|-\.->|\.\->", line):
        return None

    results: list[tuple[str, str, str]] = []
    parts: list[str] = []
    labels: list[str] = []
    remaining = line
    safety = 256

    while remaining and safety > 0:
        safety -= 1
        best_match = None
        best_pos = len(remaining)
        best_label = ""

        for pattern in arrow_patterns:
            m = re.search(pattern, remaining)
            if m and m.start() < best_pos:
                best_match = m
                best_pos = m.start()
                best_label = ""
                for g in m.groups():
                    if g is not None:
                        best_label = g.strip()
                        break

        if best_match:
            before = remaining[: best_match.start()].strip()
            if before:
                parts.append(before)
            labels.append(best_label)
            remaining = remaining[best_match.end() :].strip()
        else:
            if remaining.strip():
                parts.append(remaining.strip())
            break

    if len(parts) >= 2:
        for i in range(len(parts) - 1):
            edge_label = labels[i] if i < len(labels) else ""
            results.append((parts[i], parts[i + 1], edge_label))
    return results if results else None


def _parse_mermaid(text: str) -> tuple[dict[str, str], list[tuple[str, str]], list[tuple[str, str, str]]]:
    cleaned = _strip_mermaid_fence(text)
    lines = cleaned.strip().split("\n")

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    labeled_edges: list[tuple[str, str, str]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(flowchart|graph)\s+(TD|TB|LR|RL|BT)\b", stripped, re.IGNORECASE):
            continue
        if re.match(r"^(subgraph|end|style|classDef|class|click|linkStyle)\b", stripped, re.IGNORECASE):
            continue
        if stripped.startswith("%%"):
            continue

        edge_match = _parse_edge_line(stripped)
        if edge_match:
            for src_token, dst_token, edge_label in edge_match:
                src_id, src_label = _parse_node_def(src_token)
                dst_id, dst_label = _parse_node_def(dst_token)
                if src_id:
                    nodes.setdefault(src_id, src_label)
                if dst_id:
                    nodes.setdefault(dst_id, dst_label)
                if src_id and dst_id:
                    edges.append((src_id, dst_id))
                    labeled_edges.append((src_id, dst_id, edge_label or ""))
            continue

        node_id, label = _parse_node_def(stripped)
        if node_id:
            nodes.setdefault(node_id, label)

    return nodes, edges, labeled_edges


def _edge_similarity(edge_a: tuple, edge_b: tuple) -> float:
    src_a, dst_a = edge_a[0], edge_a[1]
    src_b, dst_b = edge_b[0], edge_b[1]
    lab_a = edge_a[2] if len(edge_a) >= 3 else ""
    lab_b = edge_b[2] if len(edge_b) >= 3 else ""

    src_a = _clean_node_label(src_a or "")
    dst_a = _clean_node_label(dst_a or "")
    src_b = _clean_node_label(src_b or "")
    dst_b = _clean_node_label(dst_b or "")
    lab_a = _clean_node_label(lab_a or "")
    lab_b = _clean_node_label(lab_b or "")

    src_sim = Levenshtein.ratio(src_a.lower(), src_b.lower())
    dst_sim = Levenshtein.ratio(dst_a.lower(), dst_b.lower())
    if not lab_a and not lab_b:
        lab_sim = 1.0
    else:
        lab_sim = Levenshtein.ratio(lab_a.lower(), lab_b.lower())
    return (src_sim + dst_sim + lab_sim) / 3.0


def _node_similarity(label_a: str, label_b: str) -> float:
    return Levenshtein.ratio(
        _clean_node_label(label_a).lower(),
        _clean_node_label(label_b).lower(),
    )


def _flowchart_hungarian_score(
    pred_items: list,
    ref_items: list,
    sim_func: Callable,
    sim_threshold: float = 0.0,
) -> float:
    if not pred_items and not ref_items:
        return 1.0
    if not pred_items or not ref_items:
        return 0.0

    n_pred = len(pred_items)
    n_ref = len(ref_items)
    sim_matrix = np.zeros((n_pred, n_ref))
    for i, p in enumerate(pred_items):
        for j, r in enumerate(ref_items):
            sim_matrix[i, j] = sim_func(p, r)

    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_sim_sum = 0.0
    for r, c in zip(row_ind, col_ind):
        if sim_matrix[r, c] >= sim_threshold:
            matched_sim_sum += sim_matrix[r, c]
    score = matched_sim_sum / max(n_pred, n_ref)
    return score


def _flowchart_similarity(
    pred_text: str,
    ref_text: str,
    edge_weight: float = 0.6,
    node_weight: float = 0.4,
    sim_threshold: float = 0.0,
) -> float:
    pred_nodes, _, pred_labeled_edges = _parse_mermaid(pred_text)
    ref_nodes, _, ref_labeled_edges = _parse_mermaid(ref_text)

    pred_edge_labels = [(pred_nodes.get(s, s), pred_nodes.get(d, d), lab or "") for s, d, lab in pred_labeled_edges]
    ref_edge_labels = [(ref_nodes.get(s, s), ref_nodes.get(d, d), lab or "") for s, d, lab in ref_labeled_edges]

    pred_node_labels = list(pred_nodes.values())
    ref_node_labels = list(ref_nodes.values())

    edge_score = _flowchart_hungarian_score(pred_edge_labels, ref_edge_labels, _edge_similarity, sim_threshold)
    node_score = _flowchart_hungarian_score(pred_node_labels, ref_node_labels, _node_similarity, sim_threshold)

    if not pred_edge_labels and not ref_edge_labels:
        return node_score

    return edge_weight * edge_score + node_weight * node_score


def flowchart_eval(
    predictions: list[str],
    references: list[str],
    easy: Literal[0, 1] = 1,
) -> tuple[tuple, list[str]]:
    """Mermaid 流程图评测（mermaid ↔ mermaid），返回 13 元组分数与日志。"""
    tolerance_params: dict[str, float] = {
        "strict": 1.0 if easy == 1 else 0.95,
        "slight": 0.85 if easy == 1 else 0.75,
        "high": 0.6 if easy == 1 else 0.5,
    }

    def _compute_sim_list(sim_threshold: float) -> list[float]:
        sim_list: list[float] = []
        for pred_text, ref_text in zip(predictions, references):
            score = _flowchart_similarity(pred_text, ref_text, sim_threshold=sim_threshold)
            sim_list.append(score)
        return sim_list

    sim_lists: dict[str, list[float]] = {}
    for tol_name, threshold in tolerance_params.items():
        sim_lists[tol_name] = _compute_sim_list(threshold)

    def _get_ap(sim_list: list[float], sim_threshold: float) -> float:
        if not sim_list:
            return 0.0
        return len([s for s in sim_list if s >= sim_threshold]) / len(sim_list)

    map_strict = 0.0
    map_slight = 0.0
    map_high = 0.0
    for sim_threshold in np.arange(0.5, 1, 0.05):
        map_strict += _get_ap(sim_lists["strict"], sim_threshold) / 10
        map_slight += _get_ap(sim_lists["slight"], sim_threshold) / 10
        map_high += _get_ap(sim_lists["high"], sim_threshold) / 10

    em = _get_ap(sim_lists["strict"], 1.0)
    ap_50_strict = _get_ap(sim_lists["strict"], 0.5)
    ap_75_strict = _get_ap(sim_lists["strict"], 0.75)
    ap_90_strict = _get_ap(sim_lists["strict"], 0.90)
    ap_50_slight = _get_ap(sim_lists["slight"], 0.5)
    ap_75_slight = _get_ap(sim_lists["slight"], 0.75)
    ap_90_slight = _get_ap(sim_lists["slight"], 0.90)
    ap_50_high = _get_ap(sim_lists["high"], 0.5)
    ap_75_high = _get_ap(sim_lists["high"], 0.75)
    ap_90_high = _get_ap(sim_lists["high"], 0.90)

    scores = (
        em,
        map_strict,
        map_slight,
        map_high,
        ap_50_strict,
        ap_75_strict,
        ap_90_strict,
        ap_50_slight,
        ap_75_slight,
        ap_90_slight,
        ap_50_high,
        ap_75_high,
        ap_90_high,
    )
    return scores, []


__all__ = [
    # 评测
    "csv_eval",
    "flowchart_eval",
    "is_csv_format",
    "is_html_table",
    "is_markdown_list",
    "is_markdown_table",
    # 格式探测
    "is_mermaid",
    "is_pipe_table",
    "normalize_to_csv",
    "tree_eval",
]
