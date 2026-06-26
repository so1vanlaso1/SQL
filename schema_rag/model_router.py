"""LLM client for remote and local OpenAI-compatible chat APIs."""
from __future__ import annotations

from dataclasses import dataclass
import json
import time
import uuid
from typing import Any, Optional

import requests

from . import config


_TEXT_LOGGED_CALL_IDS: set[str] = set()


@dataclass
class ChatResult:
    backend: str
    model: str
    content: str
    tool_calls: list[dict] | None = None
    raw: dict | None = None
    request_url: str = ""
    request_payload: dict | None = None
    response_status: int | None = None


def _message_content(message: Any) -> str:
    """Normalize common OpenAI-compatible response message shapes to text."""
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content) if content is not None else ""


def _write_call_log(call: dict) -> None:
    path = call.get("immediate_log_path")
    if not path:
        return
    config.LLM_IO_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(call, fh, ensure_ascii=False, indent=2, default=str)
    if call.get("raw_response") is not None or call.get("error"):
        _append_text_call_log(call)


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(value)


def _extract_response_preview(raw_response: Any) -> str:
    if not isinstance(raw_response, dict):
        return "" if raw_response is None else str(raw_response)
    if "response" in raw_response:
        return str(raw_response.get("response") or "")
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message")
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = _message_content(message)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            tool_text = _json_text(tool_calls)
            return f"{content}\n\nTool calls:\n{tool_text}".strip()
        return content
    return str(choice.get("text", "") or "")


def _append_text_call_log(call: dict) -> None:
    call_id = str(call.get("call_id") or "")
    if call_id and call_id in _TEXT_LOGGED_CALL_IDS:
        return
    if call_id:
        _TEXT_LOGGED_CALL_IDS.add(call_id)

    payload = call.get("request_payload") if isinstance(call.get("request_payload"), dict) else {}
    messages = payload.get("messages") if isinstance(payload, dict) else None
    status = "ERROR" if call.get("error") else "OK"
    lines = [
        "=" * 96,
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Call ID: {call_id or '(unknown)'}",
        f"Status: {status}",
        f"HTTP Status: {call.get('response_status')}",
        f"URL: {call.get('request_url')}",
        f"Model: {payload.get('model') if isinstance(payload, dict) else ''}",
        f"JSON log: {call.get('immediate_log_path')}",
        "",
    ]
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "message").upper()
            lines.extend([f"{role}:", str(message.get("content") or ""), ""])
    elif isinstance(payload, dict) and "prompt" in payload:
        lines.extend(["PROMPT:", str(payload.get("prompt") or ""), ""])

    if isinstance(payload, dict):
        extra_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"messages", "prompt"}
        }
        if extra_payload:
            lines.extend(["REQUEST SETTINGS:", _json_text(extra_payload), ""])

    if call.get("error"):
        lines.extend(["ERROR:", str(call.get("error")), ""])
    else:
        response_preview = _extract_response_preview(call.get("raw_response"))
        if response_preview:
            lines.extend(["RESPONSE:", response_preview, ""])
        lines.extend(["RAW RESPONSE JSON:", _json_text(call.get("raw_response")), ""])

    text_path = config.LLM_IO_TEXT_LOG_PATH
    text_path.parent.mkdir(parents=True, exist_ok=True)
    with open(text_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n\n")


class ModelRouter:
    def __init__(self, backend: str | None = None):
        requested = (backend or config.PIPELINE_LLM_BACKEND).lower()
        # Accept a few intuitive aliases, but keep one canonical code path.
        self.backend = {"api": "remote", "remote_api": "remote"}.get(requested, requested)
        self.last_call: dict = {}

    @staticmethod
    def _completion_url(url_or_base: str) -> str:
        """Accept either a full /v1/chat/completions URL or an API base URL."""
        cleaned = url_or_base.rstrip("/")
        if cleaned.endswith("/chat/completions"):
            return cleaned
        if cleaned.endswith("/v1/chat/completions"):
            return cleaned
        if cleaned.endswith("/v1"):
            return f"{cleaned}/chat/completions"
        return f"{cleaned}/v1/chat/completions"

    def _base_url(self) -> str:
        if self.backend == "llamacpp":
            return config.LLAMACPP_BASE_URL.rstrip("/")
        if self.backend == "openai":
            return config.OPENAI_BASE_URL.rstrip("/")
        if self.backend == "ollama":
            return config.OLLAMA_URL.rstrip("/")
        raise ValueError(f"Unsupported model backend: {self.backend}")

    def _chat_url(self, model: str) -> str:
        if self.backend == "remote":
            if model == config.GEMMA_PLANNER_MODEL:
                return self._completion_url(config.GEMMA_PLANNER_API_URL)
            if model == config.GEMMA_SKILL_MODEL:
                return self._completion_url(config.GEMMA_SKILL_API_URL)
            if model == config.QWEN_SQL_MODEL:
                return self._completion_url(config.QWEN_SQL_API_URL)
            raise ValueError(
                f"No remote chat endpoint configured for model {model!r}. "
                "Use GEMMA_PLANNER_MODEL or QWEN_SQL_MODEL, or add a route in ModelRouter."
            )
        return self._completion_url(self._base_url())

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.backend == "remote" and config.REMOTE_LLM_API_KEY:
            headers["Authorization"] = f"Bearer {config.REMOTE_LLM_API_KEY}"
        if self.backend == "llamacpp" and config.LLAMACPP_API_KEY:
            headers["Authorization"] = f"Bearer {config.LLAMACPP_API_KEY}"
        if self.backend == "openai" and config.OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {config.OPENAI_API_KEY}"
        return headers

    def load(self, model: str) -> str:
        # Remote endpoints already have their models served. No local load/unload is needed.
        if self.backend != "llamacpp" or not config.LLAMA_MANUAL_LOAD:
            return "not requested"
        resp = requests.post(
            f"{self._base_url()}/models/load",
            headers=self._headers(),
            json={"model": model},
            timeout=600,
        )
        if resp.status_code == 400 and "already running" in resp.text.lower():
            return "already running"
        resp.raise_for_status()
        return "loaded"

    def unload(self, model: str) -> str:
        # Remote endpoints keep lifecycle outside this app.
        if self.backend != "llamacpp" or not config.LLAMA_MANUAL_UNLOAD:
            return "not requested"
        resp = requests.post(
            f"{self._base_url()}/models/unload",
            headers=self._headers(),
            json={"model": model},
            timeout=120,
        )
        resp.raise_for_status()
        return "unloaded"

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 800,
        temperature: float = 0,
        response_format: Optional[dict] = None,
        chat_template_kwargs: Optional[dict] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[dict | str] = None,
    ) -> ChatResult:
        if self.backend == "ollama":
            prompt = f"{system}\n\n{user}"
            url = f"{self._base_url()}/api/generate"
            payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
            call_id = f"llm_call_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.last_call = {
                "call_id": call_id,
                "request_url": url,
                "request_payload": payload,
                "response_status": None,
                "raw_response": None,
                "error": None,
                "immediate_log_path": str((config.LLM_IO_LOG_DIR / f"{call_id}.json").resolve()),
            }
            _write_call_log(self.last_call)
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    timeout=600,
                )
                self.last_call["response_status"] = resp.status_code
                resp.raise_for_status()
                data = resp.json()
                self.last_call["raw_response"] = data
                _write_call_log(self.last_call)
                return ChatResult(self.backend, model, data.get("response", ""), None, data, url, payload, resp.status_code)
            except Exception as exc:
                self.last_call["error"] = f"{exc.__class__.__name__}: {exc}"
                _write_call_log(self.last_call)
                raise

        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        # e.g. {"enable_thinking": False} to stop a reasoning model (Qwen3.x) from
        # spending its whole token budget inside a <think> block and returning empty
        # content. llama.cpp and many OpenAI-compatible servers accept/ignore this.
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs

        url = self._chat_url(model)
        call_id = f"llm_call_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.last_call = {
            "call_id": call_id,
            "request_url": url,
            "request_payload": payload,
            "response_status": None,
            "raw_response": None,
            "error": None,
            "immediate_log_path": str((config.LLM_IO_LOG_DIR / f"{call_id}.json").resolve()),
        }
        _write_call_log(self.last_call)
        try:
            resp = requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=config.REMOTE_LLM_TIMEOUT_SECONDS if self.backend == "remote" else 600,
            )
            self.last_call["response_status"] = resp.status_code
            resp.raise_for_status()
            data = resp.json()
            self.last_call["raw_response"] = data
            choice = (data.get("choices") or [{}])[0]
            tool_calls = None
            if "message" in choice:
                message = choice.get("message") or {}
                content = _message_content(message)
                if isinstance(message, dict):
                    raw_tool_calls = message.get("tool_calls")
                    if isinstance(raw_tool_calls, list):
                        tool_calls = raw_tool_calls
                    elif isinstance(message.get("function_call"), dict):
                        tool_calls = [{"function": message["function_call"]}]
            else:
                content = str(choice.get("text", "") or "")
            _write_call_log(self.last_call)
            return ChatResult(self.backend, model, content, tool_calls, data, url, payload, resp.status_code)
        except Exception as exc:
            self.last_call["error"] = f"{exc.__class__.__name__}: {exc}"
            _write_call_log(self.last_call)
            raise
