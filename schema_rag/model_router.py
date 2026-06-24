"""LLM client for llama.cpp router mode and OpenAI-compatible chat APIs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from . import config


@dataclass
class ChatResult:
    backend: str
    model: str
    content: str
    raw: dict | None = None


class ModelRouter:
    def __init__(self, backend: str | None = None):
        self.backend = (backend or config.PIPELINE_LLM_BACKEND).lower()

    def _base_url(self) -> str:
        if self.backend == "llamacpp":
            return config.LLAMACPP_BASE_URL.rstrip("/")
        if self.backend == "openai":
            return config.OPENAI_BASE_URL.rstrip("/")
        if self.backend == "ollama":
            return config.OLLAMA_URL.rstrip("/")
        raise ValueError(f"Unsupported model backend: {self.backend}")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.backend == "llamacpp" and config.LLAMACPP_API_KEY:
            headers["Authorization"] = f"Bearer {config.LLAMACPP_API_KEY}"
        if self.backend == "openai" and config.OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {config.OPENAI_API_KEY}"
        return headers

    def load(self, model: str) -> str:
        if self.backend != "llamacpp" or not config.LLAMA_MANUAL_LOAD:
            return "not requested"
        resp = requests.post(
            f"{self._base_url()}/models/load",
            headers=self._headers(),
            json={"model": model},
            timeout=600,
        )
        resp.raise_for_status()
        return "loaded"

    def unload(self, model: str) -> str:
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
    ) -> ChatResult:
        if self.backend == "ollama":
            prompt = f"{system}\n\n{user}"
            resp = requests.post(
                f"{self._base_url()}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            return ChatResult(self.backend, model, data.get("response", ""), data)

        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        # e.g. {"enable_thinking": False} to stop a reasoning model (Qwen3.x) from
        # spending its whole token budget inside a <think> block and returning empty
        # content. llama.cpp passes these straight into the chat template.
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        resp = requests.post(
            f"{self._base_url()}/v1/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "")
        return ChatResult(self.backend, model, content, data)
