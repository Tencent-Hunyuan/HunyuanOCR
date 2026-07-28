"""Task-level rule scorers.

One entry point per rule-scored task; every entry returns a
``{"analysis", "is_valid", "reward"}`` dict (chart / ie / parsing may add
task-specific extra fields, kept as-is for downstream analytics).
"""

from .chart_deplot.utils import process_chart_deplot_task
from .ie import process_ie_task
from .layout import process_layout_task
from .parsing import process_parsing_task
from .spotting import process_spotting_task

__all__ = [
    "process_chart_deplot_task",
    "process_ie_task",
    "process_layout_task",
    "process_parsing_task",
    "process_spotting_task",
]
