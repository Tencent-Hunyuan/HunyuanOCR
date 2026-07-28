"""IE (information extraction) task rule scorer.

Entry function: ``process_ie_task(response, ref_answer)``
- When ``ref_answer`` parses as a JSON dict: field-level exact_match +
  edit-distance similarity scoring.
- Otherwise (need_fallback=True): the caller should fall back to the parsing
  local scorer.
"""

from .eval import process_ie_task

__all__ = ["process_ie_task"]
