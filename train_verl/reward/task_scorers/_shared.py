"""Shared helpers for task-level rule scorers (spotting / layout / ...)."""

from __future__ import annotations

import re

from Levenshtein import distance as _lev_distance


def normalize_text(text: str) -> str:
    """Strip XML-ish tags and whitespace, then normalise a few common
    subscript / chemistry variants used across OCR corpora."""
    text = re.sub(r"<[^>]+>", "", text)
    text = "".join(text.split())
    text = text.replace("₂", "2").replace("₁", "1")
    text = text.replace("O₂", "O2").replace("CO₂", "CO2").replace("H₂O", "H2O")
    return text


def snap_to_nearest_integer(reward: float, tolerance: float = 1e-4) -> float:
    """Snap ``reward`` to 0 or 1 when it lies within ``tolerance``; otherwise
    return the value unchanged. Reward staying at exactly 0/1 avoids downstream
    floating-point drift in RLLoggingBoard and analytics tables."""
    if abs(reward - 0) <= tolerance:
        return 0
    if abs(reward - 1) <= tolerance:
        return 1
    return reward


def levenshtein_distance(s1: str, s2: str) -> int:
    """Character-level Levenshtein distance with a pure-Python fallback in
    case the C accelerator raises on unusual inputs."""
    try:
        return _lev_distance(s1, s2)
    except Exception:
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
