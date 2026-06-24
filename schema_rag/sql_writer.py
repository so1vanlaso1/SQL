"""Qwen SQL writer stage: validated plan -> one SQL query."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from . import config
from .model_router import ModelRouter


SQL_SYSTEM = """You are a SQL generation model.

Write one SQL query only.
Do not explain.
Do not use markdown.
Use only the provided schema.
Use only the validated plan.
Do not invent columns.
Use explicit JOIN syntax.
Use table aliases."""


SQL_TEMPLATE = """SQL dialect:
{dialect}

User question:
{user_question}

Schema:
{schema_context}

Validated plan:
{validated_plan_json}

Return SQL only."""


@dataclass
class SqlWriterResult:
    backend: str
    model: str
    prompt: str
    sql: Optional[str]
    raw: Optional[str] = None
    note: str = ""


def _extract_sql(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    if ";" in text:
        text = text[: text.index(";") + 1]
    return text.strip()


def build_sql_prompt(user_question: str, schema_context: str, validated_plan: dict) -> str:
    return SQL_TEMPLATE.format(
        dialect=config.SQL_DIALECT,
        user_question=user_question,
        schema_context=schema_context,
        validated_plan_json=json.dumps(validated_plan, ensure_ascii=False, indent=2),
    )


def write_sql(
    user_question: str,
    schema_context: str,
    validated_plan: dict,
    backend: str | None = None,
) -> SqlWriterResult:
    backend = (backend or config.PIPELINE_LLM_BACKEND).lower()
    model = config.QWEN_SQL_MODEL
    prompt = build_sql_prompt(user_question, schema_context, validated_plan)
    if backend == "none":
        return SqlWriterResult(backend, model, prompt, None, note="PIPELINE_LLM_BACKEND=none - SQL prompt built, no model called.")

    router = ModelRouter(backend)
    raw = ""
    try:
        router.load(model)
        chat = router.chat(model=model, system=SQL_SYSTEM, user=prompt, max_tokens=700, temperature=0)
        raw = chat.content
        return SqlWriterResult(backend, model, prompt, _extract_sql(raw), raw=raw)
    except Exception as exc:  # noqa: BLE001
        return SqlWriterResult(backend, model, prompt, None, raw=raw, note=f"sql writer failed: {exc.__class__.__name__}: {exc}")
    finally:
        try:
            router.unload(model)
        except Exception:
            pass


def repair_sql(
    bad_sql: str,
    errors: list[str],
    user_question: str,
    schema_context: str,
    validated_plan: dict,
    backend: str | None = None,
) -> SqlWriterResult:
    prompt = (
        "The SQL failed validation.\n\n"
        f"SQL:\n{bad_sql}\n\n"
        "Errors:\n"
        + "\n".join(f"- {err}" for err in errors)
        + "\n\nRepair the SQL. Return SQL only.\n\n"
        + build_sql_prompt(user_question, schema_context, validated_plan)
    )
    backend = (backend or config.PIPELINE_LLM_BACKEND).lower()
    model = config.QWEN_SQL_MODEL
    if backend == "none":
        return SqlWriterResult(backend, model, prompt, None, note="SQL repair skipped; backend is none.")
    router = ModelRouter(backend)
    raw = ""
    try:
        router.load(model)
        chat = router.chat(model=model, system=SQL_SYSTEM, user=prompt, max_tokens=700, temperature=0)
        raw = chat.content
        return SqlWriterResult(backend, model, prompt, _extract_sql(raw), raw=raw)
    except Exception as exc:  # noqa: BLE001
        return SqlWriterResult(backend, model, prompt, None, raw=raw, note=f"sql repair failed: {exc.__class__.__name__}: {exc}")
    finally:
        try:
            router.unload(model)
        except Exception:
            pass
