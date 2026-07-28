"""
HunyuanOCR-1.5 official task prompts.

Prompts are FIXED per task type. The client only exposes `--task-type`
(not a free-form `--prompt`) so that end users cannot silently degrade model
quality by hand-editing the instruction text. Every prompt here is the
officially recommended wording for its task.

To add/adjust a task, edit this file only — the client and any batch runner
import `TASK_PROMPTS` from here.

Note on "task_type" scope:
    The 12 keys below (``doc_parse`` / ``spotting_json`` / ``layout`` / ...)
    are **inference-time prompt variants** exposed via the ``--task-type`` CLI
    flag. They pick which official prompt to send to the model.

    The reward system under ``train_verl/`` defines a separate, coarser set
    of 7 training task categories (``Spotting`` / ``Layout`` / ``Parsing`` /
    ``IE`` / ``Translation`` / ``VQA`` / ``chart_deplot``) which drives reward
    dispatch during RL training. The two vocabularies live at different
    abstraction levels and are not interchangeable.
"""

TASK_PROMPTS = {
    # 端到端文档解析
    "doc_parse": "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。",
    # 结构化解析（古文、街景等非文档结构化场景）
    "structured_parse": "提取图中的文字。",
    # Spotting — JSON 格式
    "spotting_json": "检测并识别图中所有的文字行，请按从上到下、从左到右的阅读顺序进行识别。 "
    "输出格式为 JSON 数组，每个元素必须包含："
    '"box": [xmin, ymin, xmax, ymax]（坐标需归一化到 [0, 1000] 范围内）；'
    '"text": "识别出的文字内容"。 '
    "注意：请直接输出 JSON 数组，不要包含任何多余的描述性文字。",
    # Spotting — Hunyuan 模式
    "spotting_hunyuan": "检测并识别图片中的文字，将文本坐标格式化输出。",
    # 版式分析
    "layout": "按照阅读顺序解析图中的版式信息。",
    # 版式分析 + 解析
    "layout_parse": "提取文档图片中所有内容用markdown格式表示，表格用html格式表达，文档中公式用latex格式表示，请按照阅读顺序组织进行全文解析，并输出版式分析信息。",
    # 图表解析
    "chart_parse": "解析图中的图表，对于流程图使用Mermaid格式表示，其他图表使用Markdown格式表示。",
    # 公式解析
    "formula": "识别图片中的公式，用LaTeX格式表示。",
    # 表格解析
    "table": "把图中的表格解析为HTML。",
    # 文档英译中
    "doc_trans_en2zh": "先解析文档，再将文档内容翻译为中文，其中页眉、页脚忽略，公式用latex格式表示，表格用html格式表示。",
    # 通用场景翻译 other2en
    "trans_other2en": "按照阅读顺序，提取图中文字，公式用latex格式表示，表格用markdown格式表示，再将文字内容翻译为英文。",
    # 通用场景翻译 other2zh
    "trans_other2zh": "按照阅读顺序，提取图中文字，公式用latex格式表示，表格用markdown格式表示，再将文字内容翻译为中文。",
}

# 供 --help / 展示用的中文说明
TASK_DESCRIPTIONS = {
    "doc_parse": "端到端文档解析 (正文→markdown, 表格html, 公式latex, 忽略页眉页脚)",
    "structured_parse": "结构化解析 (古文/街景等非文档场景)",
    "spotting_json": "文字检测+识别, 输出 JSON 数组 (box 归一化到 0-1000 + text)",
    "spotting_hunyuan": "文字检测+识别, Hunyuan 坐标格式",
    "layout": "版式分析 (按阅读顺序解析版式)",
    "layout_parse": "版式分析 + 全文解析",
    "chart_parse": "图表解析 (流程图→Mermaid, 其他→Markdown)",
    "formula": "公式解析 (→LaTeX)",
    "table": "表格解析 (→HTML)",
    "doc_trans_en2zh": "文档英译中 (先解析再译, 公式latex, 表格html)",
    "trans_other2en": "通用场景翻译 → 英文",
    "trans_other2zh": "通用场景翻译 → 中文",
}

DEFAULT_TASK = "doc_parse"


def get_prompt(task_type: str) -> str:
    if task_type not in TASK_PROMPTS:
        raise KeyError(f"unknown task_type '{task_type}'. choices: {', '.join(TASK_PROMPTS.keys())}")
    return TASK_PROMPTS[task_type]
