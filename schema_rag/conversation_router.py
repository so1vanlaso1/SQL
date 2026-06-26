"""Decide how a chat turn should use prior context."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from . import config
from .model_router import ModelRouter


ROUTER_SYSTEM = """You route database chat turns.

Return JSON only. Do not answer the user and do not reveal hidden reasoning.
Choose one action:
- new_query: independent question needing a fresh database query.
- refine_previous_query: follow-up that narrows, filters, sorts, or asks one slice
  of the previous database result.
- reuse_previous_result: answer can be computed directly from prior result rows.
- hybrid: use prior intent/query plus a new database query or additional data.
"""


ROUTER_TEMPLATE = """Current user question:
{question}

Recent chat state:
{state}

Return JSON:
{{
  "action": "new_query|refine_previous_query|reuse_previous_result|hybrid",
  "reason": "short user-safe explanation",
  "uses_previous": true,
  "confidence": 0.0
}}"""


@dataclass
class RouteDecision:
    action: str
    reason: str = ""
    uses_previous: bool = False
    confidence: float = 0.0
    raw: str = ""
    llm_call: dict | None = None
    note: str = ""
    trace: list[dict[str, str]] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "uses_previous": self.uses_previous,
            "confidence": self.confidence,
            "raw": self.raw,
            "note": self.note,
        }


def _compact_state(messages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for row in messages[-8:]:
        role = row.get("role")
        if role == "user":
            blocks.append(f"User: {row.get('content', '')}")
            continue
        bits = [f"Assistant: {row.get('content', '')}"]
        if row.get("selected_tables"):
            bits.append("tables=" + ", ".join(row["selected_tables"]))
        if row.get("sql"):
            bits.append("sql=" + " ".join(str(row["sql"]).split())[:500])
        if row.get("result_columns"):
            bits.append("columns=" + ", ".join(map(str, row["result_columns"])))
        if row.get("row_count") is not None:
            bits.append(f"rows={row.get('row_count')}")
        blocks.append(" | ".join(bits))
    return "\n".join(blocks) or "(no previous messages)"


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    parsed = json.loads(text)
    return parsed if isinstance(parsed, dict) else {}


def _heuristic(question: str, messages: list[dict[str, Any]], note: str = "") -> RouteDecision:
    has_prior = any(row.get("role") == "assistant" for row in messages)
    normalized = question.strip().lower()
    tokens = re.findall(r"\w+", normalized)
    followup_markers = {
        "riêng",
        "chỉ",
        "chi",
        "còn",
        "con",
        "đó",
        "do",
        "nó",
        "loc",
        "lọc",
        "them",
        "thêm",
        "này",
        "nay",
    }
    if not has_prior:
        return RouteDecision(
            "new_query",
            "Không có kết quả trước đó để tái sử dụng.",
            False,
            0.86,
            note=note,
        )
    if followup_markers & set(tokens):
        return RouteDecision(
            "refine_previous_query",
            "Câu hỏi có vẻ đang lọc hoặc thu hẹp kết quả trước đó.",
            True,
            0.78,
            note=note,
        )
    if len(tokens) <= 5:
        return RouteDecision(
            "refine_previous_query",
            "Câu hỏi ngắn và có thể phụ thuộc vào ngữ cảnh trước.",
            True,
            0.66,
            note=note,
        )
    return RouteDecision("new_query", "Câu hỏi đủ độc lập để chạy truy vấn mới.", False, 0.68, note=note)


def route_question(question: str, messages: list[dict[str, Any]], backend: str | None = None) -> RouteDecision:
    backend = (backend or config.PIPELINE_LLM_BACKEND).lower()
    if backend == "none":
        return _heuristic(question, messages, note="router used deterministic fallback because backend=none")
    router = ModelRouter(backend)
    prompt = ROUTER_TEMPLATE.format(question=question, state=_compact_state(messages))
    raw = ""
    try:
        router.load(config.GEMMA_PLANNER_MODEL)
        chat = router.chat(
            model=config.GEMMA_PLANNER_MODEL,
            system=ROUTER_SYSTEM,
            user=prompt,
            max_tokens=300,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "chat_route",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["new_query", "refine_previous_query", "reuse_previous_result", "hybrid"],
                            },
                            "reason": {"type": "string"},
                            "uses_previous": {"type": "boolean"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["action", "uses_previous"],
                    },
                },
            },
        )
        raw = chat.content
        parsed = _extract_json(raw)
        action = str(parsed.get("action") or "new_query")
        if action not in {"new_query", "refine_previous_query", "reuse_previous_result", "hybrid"}:
            action = "new_query"
        return RouteDecision(
            action=action,
            reason=str(parsed.get("reason") or ""),
            uses_previous=bool(parsed.get("uses_previous") or action != "new_query"),
            confidence=float(parsed.get("confidence") or 0),
            raw=raw,
            llm_call=router.last_call,
        )
    except Exception as exc:  # noqa: BLE001
        return _heuristic(
            question,
            messages,
            note=f"router LLM failed: {exc.__class__.__name__}: {exc}; used deterministic fallback",
        )
    finally:
        try:
            router.unload(config.GEMMA_PLANNER_MODEL)
        except Exception:
            pass
