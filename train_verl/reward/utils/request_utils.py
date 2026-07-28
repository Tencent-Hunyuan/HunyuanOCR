"""Judge-server request utilities (offline / OpenAI-compatible vLLM only).

This module talks to a locally deployed vLLM judge server through the standard
OpenAI HTTP API. Server addresses are read from ``judge_server_routes.json``
(picked by model name).
"""

import json
import os

import httpx
from openai import OpenAI


def get_model_servers(data_path: str | None = None) -> dict:
    """Return the judge model → server-list mapping (from judge_server_routes.json)."""
    if data_path is None:
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_server_routes.json")
        with open(json_path) as f:
            return json.load(f)
    return json.loads(data_path)


def request_vllm_server(server: str, prompt: str, **kwargs):
    """Call a locally deployed OpenAI-compatible vLLM judge server.

    Args:
        server: ``"<ip>:<port>"`` picked from ``judge_server_routes.json``.
        prompt: text prompt sent to the judge model.

    Kwargs (all optional):
        timeout, temperature, top_p, top_k, max_tokens, n_params, system_prompt.
    """
    client = OpenAI(
        base_url=f"http://{server}/v1",
        api_key="EMPTY",
        http_client=httpx.Client(timeout=kwargs.get("timeout", 400)),
    )

    messages = []
    if kwargs.get("system_prompt") is not None:
        messages.append({"role": "system", "content": kwargs["system_prompt"]})
    messages.append({"role": "user", "content": prompt})

    sampling = {
        "temperature": kwargs.get("temperature", 0.2),
        "top_p": kwargs.get("top_p", 0.95),
        "max_tokens": kwargs.get("max_tokens", 4096),
    }
    if kwargs.get("n_params"):
        sampling["n"] = kwargs["n_params"]

    completion = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=messages,
        **sampling,
        extra_body={"top_k": kwargs.get("top_k", 10)},
    )
    if kwargs.get("n_params"):
        return [choice.message.content for choice in completion.choices]
    return completion.choices[0].message.content
