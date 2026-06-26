"""Gemma pre-retrieval rewrite stage.

This stage rewrites the user's question into a short retrieval query for the
embedding model. It is intentionally not chat memory: the only context sent to
Gemma is the current question plus table/column names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from . import config, schema_catalog
from .model_router import ModelRouter


REWRITE_SYSTEM = """You are only a table-retrieval parser.

Your output is used to choose candidate tables for a later planner.
Do not answer the user.
Do not create a SQL plan.
Do not decide joins.
Do not decide final metrics.
Do not reason step by step.
Do not output thought, analysis, or chain-of-thought text.
Use only the provided table names and column names as schema context.

Call submit_embedding_query with:
- embedding_query: a concise retrieval query that preserves the user's original
  intent, metrics, filters, entities, and grouping keys.
- target_tables: only exact table names from the provided schema that are likely
  needed. Prefer tables that already contain both the metric and requested
  dimension. Include multiple tables only when the question really needs them.

This parser is only for retrieval. The later planner will receive the original
user question again and must not treat embedding_query as the user request."""


REWRITE_TEMPLATE = """Original user question:
{question}

Available tables and columns:
{schema_summary}

Return only by calling submit_embedding_query. Do not write SQL."""


REWRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_embedding_query",
        "description": "Submit the rewritten query that will be embedded for vector table retrieval.",
        "parameters": {
            "type": "object",
            "properties": {
                "embedding_query": {
                    "type": "string",
                    "description": "Short natural-language/keyword query for the embedding model.",
                },
                "target_tables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional likely table names from the provided schema.",
                },
            },
            "required": ["embedding_query"],
        },
    },
}


@dataclass
class QueryRewriteResult:
    backend: str
    model: str
    prompt: str
    embedding_query: str
    target_tables: list[str] = field(default_factory=list)
    raw: str = ""
    note: str = ""
    tool_args: dict[str, Any] | None = None
    llm_call: dict | None = None


def table_column_summary(joined_only: bool = True) -> str:
    catalog = schema_catalog.load_catalog()
    lines: list[str] = []
    for table_name, meta in sorted(catalog.get("tables", {}).items()):
        if joined_only and not table_name.startswith("jt_"):
            continue
        columns = ", ".join(meta.get("columns", {}).keys())
        lines.append(f"- {table_name}: {columns}")
    return "\n".join(lines)


def build_rewrite_prompt(question: str, joined_only: bool = True) -> str:
    return REWRITE_TEMPLATE.format(
        question=question,
        schema_summary=table_column_summary(joined_only=joined_only) or "(no tables)",
    )


def _tool_args(tool_calls: list[dict] | None) -> dict[str, Any] | None:
    if not tool_calls:
        return None
    for call in tool_calls:
        fn = call.get("function") if isinstance(call, dict) else None
        if not isinstance(fn, dict) or fn.get("name") != "submit_embedding_query":
            continue
        args = fn.get("arguments", {})
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _strip_reasoning_text(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)<\|channel\>thought\s*.*?<channel\|>", "", text)
    text = re.sub(r"(?is)<\|channel\|>thought\s*.*?<\|/channel\|>", "", text)
    return text.strip()


def _content_args(content: str) -> dict[str, Any] | None:
    text = _strip_reasoning_text(content)
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"embedding_query": text}
    except json.JSONDecodeError:
        return {"embedding_query": text}


def validated_target_tables(args: dict[str, Any] | None, joined_only: bool = True) -> list[str]:
    if not isinstance(args, dict):
        return []
    raw_tables = args.get("target_tables")
    if not isinstance(raw_tables, list):
        return []
    catalog = schema_catalog.load_catalog()
    valid_tables: list[str] = []
    for raw_table in raw_tables:
        table = str(raw_table).strip()
        if not table:
            continue
        if table not in catalog.get("tables", {}):
            continue
        if joined_only and not table.startswith("jt_"):
            continue
        if table not in valid_tables:
            valid_tables.append(table)
    return valid_tables


def rewrite_for_embedding(
    question: str,
    backend: str | None = None,
    joined_only: bool = True,
) -> QueryRewriteResult:
    backend = (backend or config.PIPELINE_LLM_BACKEND).lower()
    model = config.GEMMA_PLANNER_MODEL
    prompt = build_rewrite_prompt(question, joined_only=joined_only)

    if backend == "none":
        return QueryRewriteResult(
            backend=backend,
            model=model,
            prompt=prompt,
            embedding_query=question,
            note="PIPELINE_LLM_BACKEND=none - embedding rewrite skipped.",
        )

    router = ModelRouter(backend)
    raw = ""
    try:
        router.load(model)
        chat = router.chat(
            model=model,
            system=REWRITE_SYSTEM,
            user=prompt,
            max_tokens=350,
            temperature=0,
            tools=[REWRITE_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_embedding_query"}},
            chat_template_kwargs={"enable_thinking": False},
        )
        raw = chat.content
        args = _tool_args(chat.tool_calls) or _content_args(raw) or {}
        embedding_query = str(args.get("embedding_query") or "").strip()
        target_tables = validated_target_tables(args, joined_only=joined_only)
        if not embedding_query:
            return QueryRewriteResult(
                backend=backend,
                model=model,
                prompt=prompt,
                embedding_query=question,
                target_tables=target_tables,
                raw=raw,
                note="embedding rewrite returned no query; using original question.",
                tool_args=args,
                llm_call=router.last_call,
            )
        return QueryRewriteResult(
            backend=backend,
            model=model,
            prompt=prompt,
            embedding_query=embedding_query,
            target_tables=target_tables,
            raw=raw,
            tool_args=args,
            llm_call=router.last_call,
        )
    except Exception as exc:  # noqa: BLE001
        return QueryRewriteResult(
            backend=backend,
            model=model,
            prompt=prompt,
            embedding_query=question,
            raw=raw,
            note=f"embedding rewrite failed: {exc.__class__.__name__}: {exc}; using original question.",
            llm_call=router.last_call,
        )
    finally:
        try:
            router.unload(model)
        except Exception:
            pass
