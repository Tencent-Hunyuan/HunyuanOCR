# ruff: noqa: F601

import difflib
import html  # noqa # necessary for lxml parser
import re

import Levenshtein  # noqa
import pandas as pd  # noqa
from bs4 import BeautifulSoup  # noqa
from Levenshtein import distance as levenshtein_distance
from table_recognition_metric import TEDS

teds = TEDS()
teds_struct = TEDS(structure_only=True)


def remove_caption(text: str) -> str:
    """
    删除字符串中所有 <caption> ... </caption> 段落
    """
    return re.sub(r"<caption>.*?</caption>", "", text, flags=re.DOTALL | re.IGNORECASE)


def extract_tables(text: str) -> list:
    """
    从字符串中提取所有 <table> ... </table> 段落
    """
    if not isinstance(text, str):
        return []
    return re.findall(r"<table>.*?</table>", text, flags=re.DOTALL | re.IGNORECASE)


def normalize_table(t: str) -> str:
    return re.sub(r"\s+", "", t).lower()


def match_tables(pred_tables: list, gt_tables: list) -> list:
    norm_pred = [normalize_table(t) for t in pred_tables]
    norm_gt = [normalize_table(t) for t in gt_tables]
    matched = []
    used_gt = set()
    for i, p in enumerate(norm_pred):
        best_j = None
        best_score = -1
        for j, g in enumerate(norm_gt):
            if j in used_gt:
                continue
            score = difflib.SequenceMatcher(None, p, g).ratio()
            if score > best_score:
                best_score = score
                best_j = j
        used_gt.add(best_j)
        matched.append((pred_tables[i], gt_tables[best_j]))

    return matched


# fmt: off
alphabet_map = {
    "\\longleftrightarrow":"⟷", "\\Longleftrightarrow":"⟺", "\\rightleftharpoons":"⇌", "\\bigtriangledown":"▽", "\\textcircled1":"①", "\\textcircled8":"⑧", "\\textcircled7":"⑦",
    "\\textcircled6":"⑥", "\\Leftrightarrow":"⇔", "\\textcircled3":"③", "\\textcircled2":"②", "\\beginaligned":"", "\\textcircled5":"⑤", "\\leftrightarrow":"↔",
    "\\Longrightarrow":"⟹", "\\longrightarrow":"⟶", "\\textcircled9":"⑨", "\\textcircled4":"④", "\\longleftarrow":"⟵", "\\Longleftarrow":"⟸", "\\bigtriangleup":"△",
    "\\beginaligned":"", "\\endaligned":"", "\\beginarray":"", "\\triangledown":"▽", "\\Updownarrow":"⇕", "\\updownarrow":"↕", "\\endarray":"",
    "\\beginarray":"", "\\endaligned":"", "\\rightarrow":"→", "\\Rightarrow":"⇒", "\\leftarrow":"←", "\\downarrow":"↓", "\\Downarrow":"⇓",
    "\\backslash":"\\", "\\Leftarrow":"⇐", "\\therefore":"∴", "\\bigotimes":"⨂", "\\parallel":"∥", "\\geqslant":"≥", "\\leqslant":"≤",
    "\\subseteq":"⊆", "\\supseteq":"⊇", "\\bigwedge":"⋀", "\\endarray":"", "\\bigoplus":"⨁", "\\emptyset":"∅", "\\triangle":"△",
    "\\searrow":"↘", "\\because":"∵", "\\Omicron":"Ο", "\\nwarrow":"↖", "\\nearrow":"↗", "\\Uparrow":"⇑", "\\Upsilon":"Υ",
    "\\uparrow":"↑", "\\swarrow":"↙", "\\omicron":"ο", "\\epsilon":"ε", "\\Epsilon":"Ε", "\\upsilon":"υ", "\\bigstar":"★",
    "\\bigodot":"⨀", "\\bigcirc":"◯", "\\partial":"∂", "\\bumpeq":"\\", "\\propto":"∝", "\\lambda":"λ", "\\approx":"≈",
    "\\subset":"⊂", "\\bullet":"∙", "\\Lambda":"Λ", "\\otimes":"⊗", "\\ominus":"⊖", "\\supset":"⊃", "\\forall":"∀",
    "\\exists":"∃", "\\arcsin":"arcsin", "\\bigvee":"⋁", "\\arccos":"arccos", "\\bigcup":"⋃", "\\arctan":"arctan", "\\arccot":"arccot",
    "\\times":"×", "\\prime":"'", "\\sigma":"σ", "\\right":"", "\\kappa":"κ", "\\vdots":"⋮", "\\oplus":"⊕",
    "\\theta":"θ", "\\delta":"δ", "\\gamma":"γ", "\\alpha":"α", "\\Omega":"Ω", "\\Sigma":"Σ", "\\Alpha":"Α",
    "\\Gamma":"Γ", "\\Delta":"Δ", "\\Kappa":"Κ", "\\Theta":"Θ", "\\wedge":"∧", "\\omega":"ω", "\\simeq":"≃",
    "\\angle":"∠", "\\nabla":"∇", "\\ddots":"⋱", "\\cdots":"⋯", "\\infty":"∞", "\\prod":"∏", "\\beta":"β",
    "\\perp":"⊥", "\\left":"", "\\sqrt":"√", "\\zeta":"ζ", "\\dots":"…", "\\Zeta":"Ζ", "\\circ":"°",
    "\\oint":"∮", "\\iota":"ι", "\\Beta":"Β", "\\star":"⋆", "\\cdot":"⋅", "\\Iota":"Ι", "\\odot":"⊙",
    "\\surd":"√", "\\gets":"←", "\\Tau":"Τ", "\\mid":"∣", "\\Psi":"Ψ", "\\Chi":"Χ", "\\Phi":"Φ",
    "\\Rho":"Ρ", "\\log":"log", "\\Eta":"Η", "\\lim":"lim", "\\cot":"cot", "\\eta":"η", "\\bot":"⊥",
    "\\sec":"sec", "\\csc":"csc", "\\tan":"tan", "\\vee":"∨", "\\cup":"∪", "\\cap":"∩", "\\ast":"∗",
    "\\div":"÷", "\\cos":"cos", "\\sin":"sin", "\\Box":"□", "\\psi":"ψ", "\\chi":"χ", "\\phi":"φ",
    "\\sim":"∼", "\\tau":"τ", "\\rho":"ρ", "\\min":"min", "\\max":"max", "\\int":"∫", "\\sum":"∑",
    "\\lg":"lg", "\\in":"∈", "\\Mu":"Μ", "\\mp":"∓", "\\le":"≤", "\\ln":"ln", "\\to":"→",
    "\\ni":"∋", "\\ge":"≥", "\\Nu":"Ν", "\\ll":"≪", "\\gg":"≫", "\\pm":"±", "\\pi":"π",
    "\\xi":"ξ", "\\nu":"ν", "\\mu":"μ", "\\Pi":"Π", "\\Xi":"Ξ", "\\ne":"≠", "\\&":"&",
    "\\%":"%", "\\#":"#", "\\_":"_", "\\cong":"≅", "\\square":"▱", "\\blacksquare":"■",
    "\\textcircledA":"Ⓐ", "\\textcircledB":"Ⓑ", "\\textcircledC":"Ⓒ", "\\textcircledD":"Ⓓ",
    "\\textcircleda":"ⓐ", "\\textcircledb":"ⓑ", "\\textcircledc":"ⓒ", "\\textcircledd":"ⓓ",
    "\\textcircledA":"Ⓐ", "\\textcircledB":"Ⓑ", "\\textcircledC":"Ⓒ", "\\textcircledD":"Ⓓ",
    "\\textcircleda":"ⓐ", "\\textcircledb":"ⓑ", "\\textcircledc":"ⓒ", "\\textcircledd":"ⓓ"
}
# fmt: on


def simplify_formula(text: str) -> str:
    text = text.replace(" ", "")
    for k, v in alphabet_map.items():
        text = text.replace(k, v)
    if "\\{&" in text:
        text = text.replace("\\{&", "")
        text = text.replace("&", ";")

    text = re.sub(r"<pFig>.*?</pFig>", "", text)
    text = re.sub(r"<hy-meta>.*?</hy-meta>", "", text)
    text = re.sub(r"<quad>.*?</quad>", "", text)
    text = re.sub(r"<pSeal>.*?</pSeal>", "", text)
    text = text.replace("<|endoftext|>", "")
    text = text.replace("\\overrightarrow", "")
    text = text.replace("\\mathrm", "")
    text = text.replace("\\overline", "")
    text = text.replace('align="center"', "")
    text = text.replace("\\underline", "")
    text = text.replace("\\widehat", "")
    text = text.replace("\\operatorname", "")
    text = text.replace("\\begin{array}", "")
    text = text.replace("\\end{array}", "")
    text = text.replace("\\end{matrix}", "")
    text = text.replace("\\begin{matrix}", "")
    text = text.replace("\\begin{aligned}", "")
    text = text.replace("\\end{aligned}", "")
    text = text.replace("\\begin{pmatrix}", "")
    text = text.replace("\\end{pmatrix}", "")
    text = text.replace("\\displaystyle", "")
    text = text.replace("\\quad", "")
    text = text.replace("\\text{", "")
    text = text.replace("\\mathbf{", "")
    text = text.replace("{l}", "")
    text = text.replace("\\boxed", "")
    return text


def process_mid_text(text: str) -> str:
    text = text.replace("<|endoftext|>", "")
    text = text.replace("\n\n", "2")
    text = text.replace("\n", "")
    text = simplify_formula(text)
    text = text.replace("\\", "")
    text = text.replace("$", "")
    text = text.replace("<pFig>", "1")
    text = text.replace("</pFig>", "")
    text = text.replace("，", ",")
    text = text.replace("<pseal>", "")
    text = text.replace("<quad>", "")
    text = text.replace("</quad>", "")
    text = text.replace("（", "(")
    text = text.replace("）", ")")
    text = text.replace(" ", "")
    text = text.replace("<br>", "")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("<caption>", "")
    text = text.replace("</caption>", "")
    text = text.replace("</pseal>", "")
    text = text.replace("....", "")
    text = text.replace("....", "")
    text = text.replace('align="center"', "")
    return text


def calculate_score(text1: str, text2: str) -> float:
    stripped_text1 = text1.strip("2").strip()
    stripped_text2 = text2.strip("2").strip()
    edit_distance = levenshtein_distance(stripped_text1, stripped_text2)
    score = 1 - (
        edit_distance / max(len(stripped_text1), len(stripped_text2))
        if len(stripped_text1) > 0 or len(stripped_text2) > 0
        else 0
    )
    if score >= 0:
        return score
    else:
        return 0


def check_repeated_substrings(text: str) -> bool:
    n = len(text)
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return True

    return False


def process_parsing_task(response: str, ref_answer: str) -> dict:
    """Parsing 任务的评分函数（character accuracy + TEDS）。

    Args:
        response  : 模型生成的答案字符串
        ref_answer: 参考答案字符串

    Returns:
        dict: {"analysis", "is_valid", "reward"}
    """
    try:
        return _score_parsing(response, ref_answer)
    except Exception as e:
        return {
            "analysis": f"Error processing responses: {e!s}",
            "is_valid": False,
            "reward": -1.0,
        }


def _score_parsing(response: str, ref_answer: str) -> dict:
    if check_repeated_substrings(response) and len(response) > len(ref_answer) * 2:
        return {
            "analysis": "Repeat, response is too long with repeated substrings",
            "is_valid": True,
            "reward": 0,
        }

    pattern = r"\(\s*\d+\s*,\s*\d+\s*\)"
    ref_mid = process_mid_text(ref_answer)
    our_mid = process_mid_text(response)
    our_mid = re.sub(pattern, "", our_mid)

    text_score = calculate_score(ref_mid, our_mid)
    if text_score < 0.5:
        return {
            "analysis": "Text score < 0.5, reward set to 0",
            "is_valid": True,
            "reward": 0,
        }
    if text_score > 0.99:
        return {
            "analysis": "Text score > 0.99, perfect match, reward set to 1",
            "is_valid": True,
            "reward": 1,
        }

    # 面向表格元素抽取并进行 TEDS 打分
    if "<table" in ref_answer:
        pred_tables = extract_tables(response)
        gt_tables = extract_tables(ref_answer)
        if len(pred_tables) != len(gt_tables):
            return {
                "analysis": "Table num incorrect",
                "is_valid": True,
                "reward": 0,
            }
        matched_pairs = match_tables(pred_tables, gt_tables)
        tab_match = [{"pred_tab": p, "gt_tab": g} for p, g in matched_pairs]
        sum_teds = 0
        cnt_tab = 0
        sum_teds_s = 0
        cnt_tab_s = 0
        for tab in tab_match:
            pred = (
                "<html><body>"
                + remove_caption(tab["pred_tab"]).replace("\n", "").replace(" ", "").strip()
                + "</body></html>"
            )
            gt = (
                "<html><body>"
                + remove_caption(tab["gt_tab"]).replace("\n", "").replace(" ", "").strip()
                + "</body></html>"
            )
            score = teds(pred, gt)
            score_s = teds_struct(pred, gt)
            if score < 0.4 or score_s < 0.5:
                return {
                    "analysis": f"Table struct error,teds:{score},teds_s:{score_s}, reward set to 0",
                    "is_valid": True,
                    "reward": 0,
                }
            sum_teds += score
            cnt_tab += 1
            sum_teds_s += score_s
            cnt_tab_s += 1
        table_score = sum_teds / cnt_tab
        table_score_s = sum_teds_s / cnt_tab_s

        reward = 0.5 * text_score + 0.25 * table_score + 0.25 * table_score_s

        text_score, table_score, table_score_s = round(text_score, 2), round(table_score, 2), round(table_score_s, 2)

        if reward < 0.5:
            return {
                "analysis": f"Low text/tab score, reward set to 0, text_edit score: {text_score}, TEDS: {table_score}, TEDS_S: {table_score_s}",
                "is_valid": True,
                "reward": 0,
            }
        else:
            return {
                "analysis": f"text_edit score: {text_score}, TEDS: {table_score}, TEDS_S: {table_score_s}",
                "is_valid": True,
                "reward": reward,
            }

    else:
        return {
            "analysis": f"plain text matching, text_edit score: {text_score}",
            "is_valid": True,
            "reward": text_score,
        }
