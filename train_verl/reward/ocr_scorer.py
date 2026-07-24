"""OCR reward dispatcher.

``OCRScorer.process_scoring`` dispatches a scoring request to one of the
per-task rule scorers under ``reward/task_scorers/`` (spotting / layout /
parsing / ie / chart_deplot) or to a judge model over vLLM
(translation / vqa). Rule scorers return a dict; the judge path formats a
prompt via ``JUDGE_PROMPTS`` and validates the score with
``validate_format_response``.

The ``reward`` field returned by ``process_scoring`` is either a real score in
``[0, 1]`` or ``-1`` to signal "drop this sample from the training signal".
When ``reward = -1``, the accompanying ``decision`` field carries a sentinel
code that says *why*:

  * -2  response and/or ref_answer is empty (invalid sample)
  * -3  judge model failed to return a validated score
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable

from .ocr_utils import TaskType, extract_json_from_response, piecewise_linear_transform
from .task_scorers import (
    process_chart_deplot_task,
    process_ie_task,
    process_layout_task,
    process_parsing_task,
    process_spotting_task,
)
from .utils.prompt import JUDGE_PROMPTS
from .utils.request_utils import request_vllm_server
from .utils.server import ServerManager

DEBUG = False

# Judge model name; expected to match a key in reward/utils/judge_server_routes.json.
DEFAULT_JUDGE_MODEL_NAME = os.getenv("RM_SYSTEM_JUDGE_MODEL_NAME", "Qwen/Qwen3-30B-A3B") or "Qwen/Qwen3-30B-A3B"

# Global round-robin server pool.
server_manager = ServerManager()


class OCRScorer:
    def __init__(self, max_try=3, text_sim_threshold=0.99, temperature=0.2, top_p=0.95):
        self.max_try = max_try
        self.request_count = 0
        self.text_sim_threshold = text_sim_threshold
        self.temperature = temperature
        self.top_p = top_p
        self.task_type_list = [task.value for task in TaskType]

    # ------------------------------------------------------------------
    # judge model plumbing
    # ------------------------------------------------------------------

    def call_api_offline(self, img_path: str, question: str, system_prompt=None) -> str:
        """Call the local OpenAI-compatible vLLM judge server (round-robin)."""
        server = server_manager.get_next_server(DEFAULT_JUDGE_MODEL_NAME)
        if server is None:
            print(f"No available servers for {DEFAULT_JUDGE_MODEL_NAME}")
            return None
        return request_vllm_server(
            server,
            question,
            img_path,
            temperature=self.temperature,
            top_p=self.top_p,
            system_prompt=system_prompt,
        )

    def call_api(
        self,
        img_path: str | None,
        question: str,
        validate_func: Callable = None,
        type_check=None,
        system_prompt=None,
    ) -> tuple[str, str]:
        """Call the judge with ``max_try`` retries; return (validated, raw)."""
        raw_response = None
        for num_request in range(self.max_try):
            try:
                raw_response = self.call_api_offline(img_path, question, system_prompt)
                if raw_response is not None:
                    if validate_func is not None:
                        validated_response = validate_func(raw_response, type_check)
                        if validated_response is not None:
                            return validated_response, raw_response
                    else:
                        return None, raw_response
            except Exception as e:
                print(f"Error in call_api: {e}")
            if num_request + 1 < self.max_try:
                time.sleep(1)
        return None, raw_response

    def validate_format_response(self, response: str, type_check: str) -> str | int | float:
        """Post-parser for judge responses. Task-type-specific extraction of
        the numeric score / verdict from a free-form judge reply."""
        if type_check == "task_type":
            task_type = response.split("|")[0].strip()
            # 兼容 "翻译 (Translation)" 之类带括号的输出
            if "(" in task_type:
                task_type = task_type.split("(")[0].strip()
            assert task_type in self.task_type_list, f"task_type {task_type} not in {self.task_type_list}, 尝试中..."
            return task_type
        elif TaskType.is_translate(type_check):
            # 翻译任务的多策略得分提取：兼容加粗、全角空格、英文 Final Score、JSON 结构等。
            if not response:
                raise ValueError("最终得分提取失败（空响应）, 尝试中...")

            clean = response.replace("**", "").replace("　", " ")

            patterns = [
                r"最终得分\s*[：:]\s*([0-5](?:\.\d+)?)",
                r"最终评分\s*[：:]\s*([0-5](?:\.\d+)?)",
                r"综合得分\s*[：:]\s*([0-5](?:\.\d+)?)",
                r"综合评分\s*[：:]\s*([0-5](?:\.\d+)?)",
                r"总分\s*[：:]\s*([0-5](?:\.\d+)?)",
                r"final[_\s]*score\s*[:：]\s*([0-5](?:\.\d+)?)",
                r"score\s*[:：]\s*([0-5](?:\.\d+)?)\s*/\s*5",  # 形如 3.5/5
            ]
            for p in patterns:
                m = re.search(p, clean, re.IGNORECASE)
                if m:
                    final_score = float(m.group(1))
                    if 0 <= final_score <= 5:
                        return final_score

            # JSON 兜底：兼容 {"final_score": 3.5} / {"最终得分": 3.5} / {"score": 3.5} 等
            json_obj = extract_json_from_response(clean)
            if isinstance(json_obj, dict):
                for key in ("final_score", "最终得分", "最终评分", "score", "评分", "总分"):
                    if key in json_obj:
                        try:
                            final_score = float(json_obj[key])
                            if 0 <= final_score <= 5:
                                return final_score
                        except (TypeError, ValueError):
                            continue

            # 末尾数字兜底：仅当上下文存在"得分/评分/score"语义关键词时才尝试
            if re.search(r"(得分|评分|score)", clean, re.IGNORECASE):
                tail = clean[-200:]
                nums = re.findall(r"(?<![\.\d])([0-5](?:\.\d+)?)(?![\.\d])", tail)
                if nums:
                    try:
                        final_score = float(nums[-1])
                        if 0 <= final_score <= 5:
                            return final_score
                    except (TypeError, ValueError):
                        pass

            raise ValueError("最终得分提取失败, 尝试中...")
        elif TaskType.is_vqa(type_check):
            response = response.replace("**", "")
            judgement_pattern = r"Judgement:\s*(\d+)"
            judgement_match = re.search(judgement_pattern, response)
            if judgement_match:
                judgement = int(judgement_match.group(1))
                assert judgement in [0, 1], f"Judgement {judgement} 不在0-1之间"
                return judgement
            else:
                raise ValueError(f"Judgement 提取失败: {response}")

        else:
            raise ValueError(f"Invalid type in validate_format_response: {type_check}")

    # ------------------------------------------------------------------
    # dispatcher
    # ------------------------------------------------------------------

    def process_scoring(self, obj: dict) -> str | None:
        """Run rule-based / GenRM scoring; return a JSON string (or None if the
        task type is not supported)."""
        question = obj["prompt"]
        response, ref_answer = obj["response"], obj["ref_answer"]
        task_type = obj["task_type"]

        # ---- 兜底：response / ref_answer 全空 ----
        if response == "" and ref_answer == "":
            result = {
                "analysis": "答案都为空",
                "decision": -2,
                "task_type": task_type,
                "reward": -1,
            }
            return json.dumps(result, ensure_ascii=False)
        elif response == "" and ref_answer != "":
            result = {
                "analysis": "policy 模型的答案为空",
                "decision": -2,
                "task_type": task_type,
                "reward": -1,
            }
            return json.dumps(result, ensure_ascii=False)

        # ---- IE ----
        # ref 解析为 JSON dict → 字段级 exact_match + edit-distance similarity 评分；
        # 否则（need_fallback=True） → fallback 到本地 parsing 打分。
        if TaskType.is_ie(task_type):
            ie_result = process_ie_task(response, ref_answer)
            if ie_result.get("need_fallback"):
                result = process_parsing_task(response, ref_answer)
                return json.dumps(result, ensure_ascii=False)
            return json.dumps(ie_result, ensure_ascii=False)

        # ---- Parsing ----
        if TaskType.is_parsing(task_type):
            result = process_parsing_task(response, ref_answer)
            return json.dumps(result, ensure_ascii=False)

        # ---- Spotting ----
        if TaskType.is_spotting(task_type):
            result = process_spotting_task(response, ref_answer)
            return json.dumps(result, ensure_ascii=False)

        # ---- Layout ----
        if TaskType.is_layout(task_type):
            result = process_layout_task(response, ref_answer)
            return json.dumps(result, ensure_ascii=False)

        # ---- Chart deplot ----
        # 按 ref 格式分发到 csv_eval / tree_eval / flowchart_eval
        if TaskType.is_chart_deplot(task_type):
            result = process_chart_deplot_task(response, ref_answer)
            return json.dumps(result, ensure_ascii=False)

        # ---- Translation (judge model) ----
        if TaskType.is_translate(task_type):
            system_prompt = JUDGE_PROMPTS["translation"]["system_prompt"]
            template = JUDGE_PROMPTS["translation"]["template"]
            full_prompt = template.replace("{question}", question if question is not None else "")
            full_prompt = full_prompt.replace("{answer}", response)
            full_prompt = full_prompt.replace("{gt}", ref_answer)

            source_lang_text = obj.get("source_lang_text") or "[未提供，请根据参考译文判断]"
            full_prompt = full_prompt.replace("{text}", source_lang_text)

            target_lang = obj.get("target_lang") or "[未提供，请根据参考译文判断]"
            full_prompt = full_prompt.replace("{reference}", target_lang)

            judge_response, judge_response_raw = self.call_api(
                None,
                full_prompt,
                validate_func=self.validate_format_response,
                type_check=TaskType.TRANSLATION.value,
                system_prompt=system_prompt,
            )

            if judge_response is None or judge_response_raw is None:
                print(
                    "call_api error when scoring Translation:"
                    f"\n  full_prompt: {full_prompt}"
                    f"\n  judge_response: {judge_response}"
                    f"\n  judge_response_raw: {judge_response_raw}"
                )
                return None

            judge_response = piecewise_linear_transform(judge_response)
            if judge_response is None:
                result = {
                    "analysis": "judge 模型评分请求没有返回正确结果",
                    "decision": -3,
                    "task_type": task_type,
                    "reward": -1,
                }
                return json.dumps(result, ensure_ascii=False)
            result = {
                "analysis": judge_response_raw,
                "decision": judge_response,
                "task_type": task_type,
                "reward": judge_response,
            }
            return json.dumps(result, ensure_ascii=False)

        # ---- VQA (judge model) ----
        if TaskType.is_vqa(task_type):
            system_prompt = JUDGE_PROMPTS["vqa"]["system_prompt"]
            template = JUDGE_PROMPTS["vqa"]["template"]
            full_prompt = template.replace("{question}", question if question is not None else "")
            full_prompt = full_prompt.replace("{answer}", response)
            full_prompt = full_prompt.replace("{gt}", ref_answer)

            judge_response, judge_response_raw = self.call_api(
                None,
                full_prompt,
                validate_func=self.validate_format_response,
                type_check=TaskType.VQA.value,
                system_prompt=system_prompt,
            )

            if judge_response is None or judge_response_raw is None:
                print(
                    "call_api error when scoring VQA:"
                    f"\n  full_prompt: {full_prompt}"
                    f"\n  judge_response: {judge_response}"
                    f"\n  judge_response_raw: {judge_response_raw}"
                )
                return None

            result = {
                "analysis": judge_response_raw,
                "decision": judge_response,
                "task_type": task_type,
                "reward": float(judge_response),
            }
            return json.dumps(result, ensure_ascii=False)

        return None
