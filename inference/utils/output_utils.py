"""
HunyuanOCR-1.5 output utilities (shared, importable).

Two groups of helpers, all operating on model output text:

  1. Streaming + tail-repetition control, shared verbatim by all inference entry
     points (`inference/vLLM/infer_vllm_client.py`,
     `inference/vLLM/batch_infer.py`,
     `inference/transformers/infer_hf_8gpu.py`):
       - has_tail_repetition / clean_repeated_substrings / infer_stream
       - encode_image_as_data_url
     These guard against the greedy-decoding repetition degeneration.

  2. Markdown normalization for the doc_parse task (``normalize_doc_parse_markdown``):
     10 conservative patterns that align the model's markdown to the OmniDocBench
     GT convention (table caption placement, math-env splitting, layout-coord
     stripping, ...). Apply ONLY for doc_parse — other tasks (spotting JSON,
     table HTML, formula LaTeX, translation) have different output formats and
     must not be touched.

Can also be run as a CLI to normalize a directory of .md files (group 2 only):
    python output_utils.py <src_md_dir> <dst_md_dir> [--report r.json] [--dry-run]
"""

import argparse
import base64
import json
import os
import re
import shutil
import sys
from collections import defaultdict

# ==========================================================================
# Group 1 — streaming / tail-repetition control + image encoding
# ==========================================================================


# ---------- tail-repetition helpers ----------
def has_tail_repetition(text: str, min_repeats: int = 8, max_unit: int = 256) -> bool:
    """Detect if the tail of `text` is stuck in a small repeated unit."""
    n = len(text)
    if n < min_repeats * 2:
        return False
    upper = min(max_unit, n // min_repeats)
    for length in range(1, upper + 1):
        unit = text[-length:]
        if not unit.strip():
            continue
        ok = True
        for k in range(2, min_repeats + 1):
            if text[-length * k : -length * (k - 1)] != unit:
                ok = False
                break
        if ok:
            return True
    return False


def clean_repeated_substrings(text: str, min_repeats: int = 10) -> str:
    """Trim long repeated suffixes as a final safety net."""
    n = len(text)
    if n < 2000:
        return text
    for length in range(2, n // min_repeats + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length
        if count >= min_repeats:
            return text[: n - length * (count - 1)]
    return text


# ---------- image encoding ----------
def encode_image_as_data_url(path: str) -> str:
    """Read image -> base64 data URL. Mime is fixed to `image/jpeg`
    (vLLM does not care about the declared mime for base64 image payloads)."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------- inference (streaming + early-stop) ----------
def infer_stream(client, common_kwargs, repeat_min_repeats: int = 8) -> tuple[str, bool]:
    """Streaming generation with tail-repetition early-stop. Returns (text, early_stopped)."""
    stream = client.chat.completions.create(stream=True, **common_kwargs)
    parts: list[str] = []
    acc_len = 0
    next_check_at = 4000  # start checking after 4k chars
    check_step = 1000
    early_stopped = False
    for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        piece = getattr(delta, "content", None)
        if not piece:
            continue
        parts.append(piece)
        acc_len += len(piece)
        if acc_len >= next_check_at:
            next_check_at = acc_len + check_step
            tail = "".join(parts)[-8000:]
            if has_tail_repetition(tail, min_repeats=repeat_min_repeats):
                early_stopped = True
                # Best-effort close; a broken stream at this point is fine.
                try:
                    stream.close()
                except Exception:  # noqa: S110
                    pass
                break
    return "".join(parts), early_stopped


# ==========================================================================
# Group 2 — doc_parse markdown normalization (OmniDocBench GT convention)
#   normalize_doc_parse_markdown() applies 10 conservative patterns; use ONLY for doc_parse.
# ==========================================================================

# ============== Pattern E — array{l} + \\ \\ ==============
PARA_BREAK = re.compile(r"\\\\\s*\\\\")
ARRAY_L_BLOCK = re.compile(r"\\begin\{array\}\{l\}([\s\S]*?)\\end\{array\}")


def split_array_l_block(inner):
    if not PARA_BREAK.search(inner):
        return None
    parts = PARA_BREAK.split(inner)
    cleaned = []
    for p in parts:
        p = p.strip()
        p = re.sub(r"\\\\$", "", p).strip()
        if p.startswith("{") and p.endswith("}"):
            inner2 = p[1:-1]
            depth = 0
            ok = True
            for ch in inner2:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth < 0:
                        ok = False
                        break
            if ok and depth == 0:
                p = inner2.strip()
        if p:
            cleaned.append(p)
    return cleaned


def apply_pattern_E(text, stats):
    def replace_block(m, delim_pair):
        nonlocal stats
        outer = m.group(0)
        inner_full = m.group(1)
        stripped = inner_full.strip()
        am = ARRAY_L_BLOCK.fullmatch(stripped)
        if not am:
            return outer
        parts = split_array_l_block(am.group(1))
        if not parts or len(parts) < 2:
            return outer
        stats["E_blocks"] += 1
        if delim_pair == "$$":
            return "\n\n" + "\n\n".join(f"$$ {p} $$" for p in parts) + "\n\n"
        else:
            return "\n\n" + "\n\n".join(f"\\[ {p} \\]" for p in parts) + "\n\n"

    text = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: replace_block(m, "$$"), text)
    text = re.sub(r"\\\[([\s\S]*?)\\\]", lambda m: replace_block(m, r"\["), text)
    return text


# ============== Pattern C — multi-row math env split ==============
# Math environments where rows separated by '\\' are INDEPENDENT equations that GT-side
# also splits. But the SAME environments are also used for a single multi-line aligned
# formula (e.g. "x &= a \\ &= b \\ &= c"): there the '\\' rows are NOT independent and
# must stay together. Splitting those corrupts the formula (leaves bare '&'/operator
# fragments) and tanks CDM. So we only split when EVERY row after the first looks
# independent: it must NOT contain a '&' alignment tab and must NOT start with a
# continuation token ('=', operator, relation) that ties it to the previous line.
ROW_ENVS = ["aligned", "align\\*", "align", "split", "cases", "gather", "gathered"]
ROW_ENV_RE = re.compile(r"\\begin\{(" + "|".join(ROW_ENVS) + r")\}([\s\S]*?)\\end\{\1\}")
ROW_BREAK = re.compile(r"\\\\(?:\s*\[[^\]]*\])?")  # '\\' optional with [3pt] spacing
# a row that continues the previous line (alignment / relation-led) — do NOT split
_ROW_CONTINUATION = re.compile(
    r"^(&|=|\\approx|\+|-|\\geq|\\leq|\\le|\\ge|<|>|\\times|\\cdot|\\pm|\\mp|\\Rightarrow|\\rightarrow|\\to|\\equiv)"
)


def split_env_rows(env_name, inner):
    """Split inner content by single '\\' (with optional spacing arg) into rows.
    Filter empty rows. Return list of cleaned strings."""
    parts = ROW_BREAK.split(inner)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Strip leading/trailing { } braces for cell content
        out.append(p)
    return out


def _rows_are_independent(rows):
    """True if these rows are separate equations (safe to split), False if they form
    one multi-line aligned formula (rows continue each other via '&' or relation-lead)."""
    for r in rows:
        if "&" in r:
            return False
    for r in rows[1:]:
        if _ROW_CONTINUATION.match(r):
            return False
    return True


def apply_pattern_C(text, stats):
    r"""Find $$..$$ or \[..\] containing exactly one ROW_ENV that spans (mostly) the
    block, with >= 2 INDEPENDENT rows. Split into N \[...\] blocks."""

    def expand(m, delim_pair):
        nonlocal stats
        outer = m.group(0)
        inner_full = m.group(1)
        stripped = inner_full.strip()
        env_match = ROW_ENV_RE.fullmatch(stripped)
        if not env_match:
            return outer
        env_name = env_match.group(1)
        env_inner = env_match.group(2)
        rows = split_env_rows(env_name, env_inner)
        if len(rows) < 2:
            return outer
        if not _rows_are_independent(rows):
            return outer  # single multi-line aligned formula — keep intact
        stats["C_blocks"] += 1
        # Each row -> independent \[...\] (same as GT-side fragments)
        return "\n\n" + "\n\n".join(f"\\[ {r} \\]" for r in rows) + "\n\n"

    text = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: expand(m, "$$"), text)
    text = re.sub(r"\\\[([\s\S]*?)\\\]", lambda m: expand(m, r"\["), text)
    return text


# ============== Pattern C2 — array{} with &-separated INDEPENDENT equations ==============
# Worksheet/口算题 pages: the model dumps many independent little equations into one
#   \begin{array}{l} a= &b= &c= \\ d= &e= &f= \\ ... \end{array}
# where '&' separates *independent* equations (each cell has its own '='), NOT math
# alignment. GT stores each little equation as a SEPARATE formula, so the packed array
# matches 1-to-N and scores ~0. Splitting both '\\' (rows) and '&' (cells) into
# independent \[..\] blocks recovers the alignment.
#
# CRITICAL guard vs real alignment (e.g. Putnam "x &= a \\ &= b"): only split '&' when
# EVERY cell in EVERY row is an independent equation — i.e. contains its own '='/'\approx'
# and does NOT start with an operator/'='/'\approx' (which would mean it continues the
# previous line). Real alignment fails this test (cells start with '&=') and is left intact.
ARRAY_ENV_RE = re.compile(r"\\begin\{array\}\{[lrc|]+\}([\s\S]*?)\\end\{array\}")
_CELL_CONT_START = re.compile(r"^(=|\\approx|\+|-|\\times|\\div|\\leq|\\geq|\\le|\\ge|<|>|\\cdot)")


def _cell_is_independent_eq(cell):
    c = cell.strip()
    if not c:
        return False
    if _CELL_CONT_START.match(c):
        return False
    return ("=" in c) or ("\\approx" in c)


def apply_pattern_C2(text, stats):
    def expand(m, delim_pair):
        nonlocal stats
        outer = m.group(0)
        stripped = m.group(1).strip()
        am = ARRAY_ENV_RE.fullmatch(stripped)
        if not am:
            return outer
        rows = [r for r in re.split(r"\\\\(?:\s*\[[^\]]*\])?", am.group(1)) if r.strip()]
        if len(rows) < 2:
            return outer
        # every row must contain '&', and every NON-EMPTY &-cell must be an
        # independent equation (trailing empty cells = unfilled worksheet slots, ignored)
        amp_rows = [r for r in rows if "&" in r]
        if len(amp_rows) < 2:
            return outer
        for r in rows:
            cells = [c for c in r.split("&") if c.strip()]
            if not cells or not all(_cell_is_independent_eq(c) for c in cells):
                return outer  # abort: real alignment / mixed content / non-equation row
        out = []
        for r in rows:
            for c in r.split("&"):
                c = c.strip()
                if c:
                    out.append(c)
        if len(out) < 2:
            return outer
        stats["C2_blocks"] += 1
        stats["C2_cells"] += len(out)
        return "\n\n" + "\n\n".join(f"\\[ {o} \\]" for o in out) + "\n\n"

    text = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: expand(m, "$$"), text)
    text = re.sub(r"\\\[([\s\S]*?)\\\]", lambda m: expand(m, r"\["), text)
    return text


# ============== Pattern D — \left/\right balance ==============
# Count occurrences as in CDM render: \left followed by non-letter; \right followed by non-letter.
LEFT_RE = re.compile(r"\\left(?![a-zA-Z@])")
RIGHT_RE = re.compile(r"\\right(?![a-zA-Z@])")


def fix_left_right(content):
    """If imbalanced, try a safe fix:
    left > right: insert \right. before the LAST \\end{array}/\\end{aligned}/...,
                  or before the closing delimiter at the very end.
    right > left: insert \\left. right after the first \begin{...}, or at start.
    """
    nl = len(LEFT_RE.findall(content))
    nr = len(RIGHT_RE.findall(content))
    if nl == nr:
        return content, 0
    if nl > nr:
        diff = nl - nr
        # Try inserting before last \end{X} as many times as needed
        end_match = list(re.finditer(r"\\end\{[^}]+\}", content))
        if end_match:
            pos = end_match[-1].start()
            return content[:pos] + (r"\right." * diff) + " " + content[pos:], diff
        # Else append before stripped end
        return content.rstrip() + (" " + (r"\right." * diff)), diff
    else:
        diff = nr - nl
        # Insert \left. after first \begin{X}, or at very start
        begin_match = re.search(r"\\begin\{[^}]+\}", content)
        if begin_match:
            pos = begin_match.end()
            return content[:pos] + " " + (r"\left." * diff) + content[pos:], diff
        return " " + (r"\left." * diff) + content, diff


def apply_pattern_D(text, stats):
    def fixer(m, delim_open, delim_close):
        nonlocal stats
        inner = m.group(1)
        new_inner, diff = fix_left_right(inner)
        if diff > 0:
            stats["D_blocks"] += 1
            stats["D_inserts"] += diff
        return delim_open + new_inner + delim_close

    text = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: fixer(m, "$$", "$$"), text)
    text = re.sub(r"\\\[([\s\S]*?)\\\]", lambda m: fixer(m, r"\[", r"\]"), text)
    return text


# ============== Pattern A — inline → display ==============
# Promote a $..$ that occupies an entire line and contains a "real formula".
# Rule: standalone-line, length>=15 chars OR contains \frac/\sum/\int/\therefore/\because.
INLINE_FULL_LINE = re.compile(r"(?m)^[ \t]*\$([^\$\n]+?)\$[ \t]*$")
RICH_TOKEN = re.compile(
    r"\\(?:frac|sum|int|prod|therefore|because|forall|exists|begin|left|right|sqrt|partial|nabla|infty|cdot|times|div|leq|geq|neq|approx|equiv|to|rightarrow|leftarrow|leftrightarrow|cap|cup|in|notin|subset|supseteq|cong|sim)\b"
)


def looks_like_real_formula(s):
    s = s.strip()
    if len(s) >= 15:
        return True
    return bool(RICH_TOKEN.search(s))


def apply_pattern_A(text, stats):
    def repl(m):
        inner = m.group(1).strip()
        if looks_like_real_formula(inner):
            stats["A_blocks"] += 1
            return f"\n\n$$ {inner} $$\n\n"
        return m.group(0)

    return INLINE_FULL_LINE.sub(repl, text)


# ============== Pattern W — bare arithmetic line → display ==============
# Worksheet/口算题 pages: the model emits arithmetic drills as PLAIN TEXT lines
#   "50÷8="   "73.33+65.66-71.73=67.26    1.5×4=6"
# with unicode ÷/× and no $ wrapping. GT stores each as a display formula, so the
# unwrapped text never matches -> whole page's formulas score 0. We promote such a
# line to $$..$$ and normalize ÷/× to \div/\times.
#
# STRICT guard to avoid touching prose/tables:
#   - line must consist ONLY of digits, whitespace, and arithmetic chars ×÷+-=.,()（） —
#     any Chinese char / latin letter / '$' / '<' disqualifies it (so prose, variables
#     like "x=1", existing math, and table rows are all skipped)
#   - must contain '=' AND a real operator (÷ / × / "digit +/- digit")
BARE_ARITH_LINE = re.compile(r"^[\d\s×÷+\-=.,()（）]+$")


def _is_bare_arith(s):
    s = s.strip()
    if len(s) < 3 or len(s) > 80:
        return False
    if "=" not in s:
        return False
    if not BARE_ARITH_LINE.match(s):
        return False
    return ("÷" in s) or ("×" in s) or bool(re.search(r"\d\s*[+\-]\s*\d", s))


def apply_pattern_W(text, stats):
    out = []
    for line in text.split("\n"):
        if _is_bare_arith(line):
            s = line.strip()
            # A worksheet line often packs several INDEPENDENT drills separated by 2+
            # spaces ("315÷3.5=90     31.2÷6=5.2"). GT stores each as a separate
            # formula, so emit one $$..$$ per drill. Only split when every part is its
            # own equation (contains '='); otherwise keep the line intact.
            parts = [p.strip() for p in re.split(r"\s{2,}", s) if p.strip()]
            if len(parts) >= 2 and all("=" in p for p in parts):
                drills = parts
            else:
                drills = [s]
            for d in drills:
                d = d.replace("÷", r"\div ").replace("×", r"\times ")
                out.append(f"$$ {d} $$")
                stats["W_lines"] += 1
        else:
            out.append(line)
    return "\n".join(out)


# ============== Pattern F — strip equation numbers ==============
TAG_RE = re.compile(r"\\tag\s*\{[^}]*\}\s*")
QUAD_NUM_RE = re.compile(r"\\q?quad\s*\{?\s*\((\d[\d\.\-]*)\)\s*\}?\s*$")
CDOTS_NUM_RE = re.compile(r"(?:\\cdots){2,}\s*\((\d[\d\.\-]*)\)\s*$")
TRAIL_PAREN_NUM_RE = re.compile(r"\s+\((\d[\d\.\-]+)\)\s*$")  # trailing space + (1.2.3)


def strip_eq_numbers(content):
    n = 0
    for pat in (TAG_RE, QUAD_NUM_RE, CDOTS_NUM_RE):
        m = pat.search(content)
        while m:
            content = content[: m.start()] + content[m.end() :]
            n += 1
            m = pat.search(content)
    # trailing standalone (1.2.3) only at very end
    m = TRAIL_PAREN_NUM_RE.search(content)
    if m and m.start() > 5:  # avoid (i,t)= type
        content = content[: m.start()].rstrip()
        n += 1
    return content, n


def apply_pattern_F(text, stats):
    def fixer(m, delim_open, delim_close):
        nonlocal stats
        inner = m.group(1)
        new_inner, n = strip_eq_numbers(inner)
        if n > 0:
            stats["F_blocks"] += 1
            stats["F_strips"] += n
        return delim_open + new_inner + delim_close

    text = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: fixer(m, "$$", "$$"), text)
    text = re.sub(r"\\\[([\s\S]*?)\\\]", lambda m: fixer(m, r"\[", r"\]"), text)
    return text


# ============== Pattern V — display block LaTeX repair (low-risk only) ==============
# In CDM, display formulas render via \begin{displaymath}...\end{displaymath}. Two
# common pred artifacts make pdflatex emit errors and (under nonstopmode) render a
# garbled page -> CDM scored 0 even though the content is nearly correct:
#   1. a leading bare '&' inside the block (alignment tab is illegal in displaymath)
#      -> "! Misplaced alignment tab character &."
#   2. surplus trailing '}' (net brace depth < 0)
#      -> "! Extra }, or forgotten $."
# Both fixes are loss-free: removing a leading '&' that can't legally be there, and
# removing only the trailing surplus '}' (never touching interior braces, never
# ADDING braces — global '}' insertion is unsafe so we deliberately skip it).
def _net_brace_depth(s):
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return depth


def repair_display_block(content):
    n = 0
    orig = content
    # 1. strip a single leading bare '&' (after optional whitespace)
    stripped = re.sub(r"^(\s*)&", r"\1", content)
    if stripped != content:
        content = stripped
        n += 1
    # 2. drop surplus trailing '}' only when net depth is negative
    depth = _net_brace_depth(content)
    if depth < 0:
        need = -depth
        # peel trailing '}' (allowing trailing whitespace) without crossing an escape
        while need > 0:
            m = re.search(r"\}\s*$", content)
            if not m:
                break
            # ensure the '}' isn't escaped (preceding backslash count even)
            bs = 0
            j = m.start() - 1
            while j >= 0 and content[j] == "\\":
                bs += 1
                j -= 1
            if bs % 2 == 1:
                break
            content = content[: m.start()] + content[m.start() + 1 :]
            need -= 1
            n += 1
    return content, (n if content != orig else 0)


def apply_pattern_V(text, stats):
    def fixer(m, delim_open, delim_close):
        nonlocal stats
        inner = m.group(1)
        new_inner, n = repair_display_block(inner)
        if n > 0:
            stats["V_blocks"] += 1
            stats["V_fixes"] += n
        return delim_open + new_inner + delim_close

    text = re.sub(r"\$\$([\s\S]*?)\$\$", lambda m: fixer(m, "$$", "$$"), text)
    text = re.sub(r"\\\[([\s\S]*?)\\\]", lambda m: fixer(m, r"\[", r"\]"), text)
    return text


# ============== Pattern T — table caption split ==============
# Pred: <table><caption>XXX</caption><tr>...</tr>...</table>
# Want: XXX\n\n<table><tr>...</tr>...</table>
# (caption inside <table> isn't matched against GT's caption entity which lives outside the table)
TABLE_CAPTION_RE = re.compile(
    r"<table([^>]*)>\s*<caption[^>]*>([\s\S]*?)</caption>\s*([\s\S]*?)</table>",
    re.IGNORECASE,
)


def apply_pattern_T(text, stats):
    def fixer(m):
        attrs = m.group(1) or ""
        caption = m.group(2).strip()
        body = m.group(3).strip()
        # Strip any leading stray <td> that some preds emit before <tr>
        body = re.sub(r"^<td>\s*(?=<tr>)", "", body, flags=re.IGNORECASE)
        if not caption:
            return m.group(0)
        stats["T_captions"] += 1
        return f"{caption}\n\n<table{attrs}>{body}</table>"

    return TABLE_CAPTION_RE.sub(fixer, text)


# ============== Pattern U — strip layout coordinate tokens ==============
# Pred emits layout bbox coords as a trailing pair of points after figures/captions:
#   "数据来源于高德地图驾车导航数据(39,337),(512,900)"      -> "数据来源于高德地图驾车导航数据"
#   "(147,90),(925,415)"  (a bare figure box, no text)        -> deleted entirely
# These artificial coord tokens are not GT content and depress the text Edit metric.
#
# Safety rules (verified against real preds):
#   - NEVER touch coords inside a <table>...</table> block (they are cell bbox the
#     TEDS path ignores, and deleting them would corrupt table structure).
#   - NEVER touch coords inside math spans ($..$, $$..$$, \[..\]) — real point sets
#     like \{(2,4),(3,3)\} must survive.
#   - Only strip when the coord pair(s) sit at the END of a text line (nothing but
#     whitespace / further coord pairs after them). A coord followed by more prose is
#     left alone (could be an inline icon ref — too risky, only 4 such cases).
COORD_PAIR = re.compile(r"\(\d{1,4},\d{1,4}\),\(\d{1,4},\d{1,4}\)")
# one-or-more trailing coord pairs (optionally separated by , / whitespace) at line end
TRAILING_COORDS = re.compile(r"(?:[ \t,，]*\(\d{1,4},\d{1,4}\),\(\d{1,4},\d{1,4}\))+[ \t]*$")


def _has_html_table_token(line):
    low = line.lower()
    return ("<table" in low) or ("<tr" in low) or ("<td" in low) or ("</td" in low)


def apply_pattern_U(text, stats):
    # Mask math spans so coords inside formulas are never matched.
    spans = []

    def _mask(m):
        spans.append(m.group(0))
        return f"\x00U{len(spans) - 1}\x00"

    masked = re.sub(r"\$\$[\s\S]*?\$\$|\$[^\$\n]+\$|\\\[[\s\S]*?\\\]", _mask, text)

    out_lines = []
    for line in masked.split("\n"):
        # Leave table-bearing lines untouched.
        if _has_html_table_token(line):
            out_lines.append(line)
            continue
        if not COORD_PAIR.search(line):
            out_lines.append(line)
            continue
        stripped = TRAILING_COORDS.sub("", line)
        if stripped == line:
            # coord pair exists but not as a clean trailing token -> skip (risky)
            out_lines.append(line)
            continue
        if stripped.strip():
            # had leading text -> keep text, drop trailing coords
            stats["U_coord_text"] += 1
            out_lines.append(stripped.rstrip())
        else:
            # bare coord line (figure box) -> drop the whole line
            stats["U_coord_bare"] += 1
            # skip appending (line removed)
    masked = "\n".join(out_lines)

    # Restore math spans.
    def _unmask(m):
        return spans[int(m.group(1))]

    return re.sub(r"\x00U(\d+)\x00", _unmask, masked)


# ============== Driver ==============
def normalize_doc_parse_markdown(text):
    """Normalize a doc_parse model output.

    Applies 10 conservative patterns to align the model's markdown with the
    OmniDocBench GT convention (table caption placement, math-env splitting,
    layout-coord stripping, ...). Apply ONLY for the ``doc_parse`` task type;
    other tasks (spotting JSON, table HTML, formula LaTeX, translation) have
    different output formats and must not be touched.

    Args:
        text: raw model output string.

    Returns:
        ``(normalized_text, stats_dict)`` where ``stats_dict`` counts how many
        edits each pattern made (useful for offline audits).
    """
    stats = defaultdict(int)
    text = apply_pattern_U(text, stats)  # 0. strip trailing layout coord tokens (do first, before others)
    text = apply_pattern_T(text, stats)  # 0. extract <caption> from <table>
    text = apply_pattern_E(text, stats)  # array{l} + \\ \\
    text = apply_pattern_C(text, stats)  # aligned/split/cases multi-row
    text = apply_pattern_C2(text, stats)  # array{} with &-separated independent equations (口算题)
    text = apply_pattern_D(text, stats)  # \left/\right rebalance
    text = apply_pattern_A(text, stats)  # inline → display
    text = apply_pattern_W(text, stats)  # bare arithmetic line → display (口算题裸算式)
    text = apply_pattern_F(text, stats)  # strip eq numbers
    text = apply_pattern_V(text, stats)  # low-risk display-block LaTeX repair (bare &, surplus })
    return text, dict(stats)


# ========== CLI (group 2 only) ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    if not args.dry_run:
        if os.path.exists(dst):
            if not os.path.isdir(dst):
                print(f"refuse to overwrite non-dir {dst}")
                sys.exit(1)
            shutil.rmtree(dst)
        os.makedirs(dst, exist_ok=True)

    n_files = 0
    n_changed = 0
    total_stats = defaultdict(int)
    per_file = {}

    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".md"):
            continue
        n_files += 1
        sp = os.path.join(src, fn)
        with open(sp, "r", encoding="utf-8") as f:
            text = f.read()
        new_text, stats = normalize_doc_parse_markdown(text)
        if any(v > 0 for v in stats.values()):
            n_changed += 1
            per_file[fn] = stats
            for k, v in stats.items():
                total_stats[k] += v
        if not args.dry_run:
            dp = os.path.join(dst, fn)
            with open(dp, "w", encoding="utf-8") as f:
                f.write(new_text)

    print(f"Files processed: {n_files}")
    print(f"Files changed:   {n_changed}")
    print("Aggregate stats:")
    for k in sorted(total_stats):
        print(f"  {k}: {total_stats[k]}")
    if args.report:
        with open(args.report, "w") as f:
            json.dump(
                {
                    "total": dict(total_stats),
                    "n_files": n_files,
                    "n_changed": n_changed,
                    "per_file": per_file,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
