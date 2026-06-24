"""Pluggable SQL generator. The RAG retrieval works WITHOUT any of these - the default
backend is "none", which just builds the prompt so you can inspect the schema pack.

Backends:
  none          - return the prompt only (no model call)
  ollama        - POST to a local Ollama server (e.g. `ollama run qwen2.5-coder`)
  openai        - any OpenAI-compatible /chat/completions endpoint (OpenAI, DeepSeek, ...)
  transformers  - local HuggingFace model (heavy; needs torch)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from . import config

SYSTEM_PROMPT = (
    "You are an expert data analyst who writes correct, minimal SQLite SQL. "
    "Use ONLY the tables and columns in the provided schema subset. "
    "Use the listed join paths for any JOINs. Return a single SQL query and nothing else."
)


def build_prompt(question: str, schema_pack: str) -> str:
    return (
        f"{schema_pack}\n\n"
        f"-- Question --\n{question}\n\n"
        f"-- Task --\n"
        f"Write one SQLite query that answers the question using only the schema above.\n"
        f"Return only the SQL, no explanation, no markdown fences.\n\n"
        f"SQL:"
    )


def _extract_sql(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    # cut at first statement terminator if the model rambled afterwards
    if ";" in text:
        text = text[: text.index(";") + 1]
    return text.strip()


@dataclass
class SqlGenResult:
    backend: str
    prompt: str
    sql: Optional[str]
    raw: Optional[str] = None
    note: str = ""


def _gen_ollama(prompt: str) -> str:
    import requests  # lazy

    resp = requests.post(
        f"{config.OLLAMA_URL}/api/generate",
        json={"model": config.OLLAMA_MODEL, "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}", "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _gen_openai(prompt: str) -> str:
    import requests  # lazy

    resp = requests.post(
        f"{config.OPENAI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": config.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _gen_transformers(prompt: str) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy, heavy

    tok = AutoTokenizer.from_pretrained(config.HF_SQL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(config.HF_SQL_MODEL, device_map="auto")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=400, do_sample=False)
    return tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)


def generate_sql(question: str, schema_pack: str, backend: Optional[str] = None) -> SqlGenResult:
    backend = (backend or config.SQL_LLM_BACKEND).lower()
    prompt = build_prompt(question, schema_pack)

    if backend == "none":
        return SqlGenResult(backend, prompt, None, note="SQL_LLM_BACKEND=none - prompt built, no model called.")

    try:
        if backend == "ollama":
            raw = _gen_ollama(prompt)
        elif backend == "openai":
            raw = _gen_openai(prompt)
        elif backend == "transformers":
            raw = _gen_transformers(prompt)
        else:
            return SqlGenResult(backend, prompt, None, note=f"Unknown backend '{backend}'.")
    except Exception as exc:  # noqa: BLE001
        return SqlGenResult(backend, prompt, None, note=f"{backend} call failed: {exc.__class__.__name__}: {exc}")

    return SqlGenResult(backend, prompt, _extract_sql(raw), raw=raw)
