"""Gemma planner stage: question/context -> structured JSON plan."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from . import config
from .model_router import ModelRouter


PLANNER_SYSTEM = """You are a database planning model.

Your job is to create a SQL plan.
Do not write SQL.
Use only the provided tables, columns, and allowed joins.
Do not invent tables.
Do not invent columns.
Do not invent joins.
Return valid JSON only."""


PLAN_TEMPLATE = """User question:
{user_question}

Selected table descriptions:
{skill_md_context}

Schema:
{schema_context}

Allowed joins:
{allowed_join_graph}

Return:
{{
  "intent": "",
  "required_tables": [],
  "join_plan": [],
  "filters": [],
  "metrics": [],
  "group_by": [],
  "order_by": [],
  "limit": null,
  "missing_information": []
}}"""


# JSON schema the planner output is constrained to (llama.cpp / OpenAI json_schema
# response_format). This forces ONE stable structure so downstream code never has to
# guess between key variants (left vs left_table, field vs column, name/formula vs
# field/aggregation). Values still need semantic validation in plan_validator, but the
# *shape* is now fixed. Metrics use field+aggregation (a qualified column + an enum)
# rather than a free-form formula, because the model writes qualified columns reliably
# but tends to invent aliases inside hand-written formulas.
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "required_tables": {"type": "array", "items": {"type": "string"}},
        "join_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "left": {"type": "string"},
                    "right": {"type": "string"},
                    "join_type": {"type": "string"},
                },
                "required": ["left", "right"],
            },
        },
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "op": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["column"],
            },
        },
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "aggregation": {
                        "type": "string",
                        "enum": ["SUM", "COUNT", "AVG", "MIN", "MAX", "COUNT_DISTINCT"],
                    },
                    "alias": {"type": "string"},
                },
                "required": ["field", "aggregation"],
            },
        },
        "group_by": {"type": "array", "items": {"type": "string"}},
        "order_by": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "direction": {"type": "string", "enum": ["ASC", "DESC"]},
                },
                "required": ["field"],
            },
        },
        "limit": {"type": ["integer", "null"]},
        "missing_information": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["intent", "required_tables", "join_plan", "metrics"],
}


def _plan_response_format(backend: str) -> Optional[dict]:
    """Constrain the planner output to PLAN_SCHEMA on backends that support it."""
    if backend in {"llamacpp", "openai"}:
        return {
            "type": "json_schema",
            "json_schema": {"name": "sql_plan", "schema": PLAN_SCHEMA},
        }
    return None


@dataclass
class PlannerResult:
    backend: str
    model: str
    prompt: str
    plan: Optional[dict]
    raw: Optional[str] = None
    note: str = ""


def build_planner_prompt(
    user_question: str,
    skill_md_context: str,
    schema_context: str,
    allowed_join_graph: list[dict],
) -> str:
    return PLAN_TEMPLATE.format(
        user_question=user_question,
        skill_md_context=skill_md_context,
        schema_context=schema_context,
        allowed_join_graph=json.dumps(allowed_join_graph, ensure_ascii=False, indent=2),
    )


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def create_plan(
    user_question: str,
    skill_md_context: str,
    schema_context: str,
    allowed_join_graph: list[dict],
    backend: str | None = None,
) -> PlannerResult:
    backend = (backend or config.PIPELINE_LLM_BACKEND).lower()
    model = config.GEMMA_PLANNER_MODEL
    prompt = build_planner_prompt(user_question, skill_md_context, schema_context, allowed_join_graph)
    if backend == "none":
        return PlannerResult(backend, model, prompt, None, note="PIPELINE_LLM_BACKEND=none - planner prompt built, no model called.")

    router = ModelRouter(backend)
    raw = ""
    try:
        router.load(model)
        response_format = _plan_response_format(backend)
        chat = router.chat(
            model=model,
            system=PLANNER_SYSTEM,
            user=prompt,
            max_tokens=2048,
            temperature=0,
            response_format=response_format,
        )
        raw = chat.content
        plan = _extract_json(raw)
        return PlannerResult(backend, model, prompt, plan, raw=raw)
    except Exception as exc:  # noqa: BLE001
        return PlannerResult(backend, model, prompt, None, raw=raw, note=f"planner failed: {exc.__class__.__name__}: {exc}")
    finally:
        try:
            router.unload(model)
        except Exception:
            pass


def repair_plan(
    previous_plan: dict,
    errors: list[str],
    user_question: str,
    skill_md_context: str,
    schema_context: str,
    allowed_join_graph: list[dict],
    backend: str | None = None,
) -> PlannerResult:
    prompt = build_planner_prompt(user_question, skill_md_context, schema_context, allowed_join_graph)
    repair = (
        "Your previous plan was invalid.\n\n"
        f"Previous plan:\n{json.dumps(previous_plan, ensure_ascii=False, indent=2)}\n\n"
        "Validation errors:\n"
        + "\n".join(f"- {err}" for err in errors)
        + "\n\nRepair the JSON plan. Return JSON only.\n\n"
        + prompt
    )
    backend = (backend or config.PIPELINE_LLM_BACKEND).lower()
    if backend == "none":
        return PlannerResult(backend, config.GEMMA_PLANNER_MODEL, repair, None, note="planner repair skipped; backend is none.")
    router = ModelRouter(backend)
    raw = ""
    try:
        router.load(config.GEMMA_PLANNER_MODEL)
        response_format = _plan_response_format(backend)
        chat = router.chat(
            model=config.GEMMA_PLANNER_MODEL,
            system=PLANNER_SYSTEM,
            user=repair,
            max_tokens=2048,
            temperature=0,
            response_format=response_format,
        )
        raw = chat.content
        return PlannerResult(backend, config.GEMMA_PLANNER_MODEL, repair, _extract_json(raw), raw=raw)
    except Exception as exc:  # noqa: BLE001
        return PlannerResult(backend, config.GEMMA_PLANNER_MODEL, repair, None, raw=raw, note=f"planner repair failed: {exc.__class__.__name__}: {exc}")
    finally:
        try:
            router.unload(config.GEMMA_PLANNER_MODEL)
        except Exception:
            pass
