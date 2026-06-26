"""End-to-end RAG-assisted text-to-SQL orchestration."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import config, entity_resolver, plan_validator, planner, query_rewriter, retriever, sql_writer, validator
from .plan_validator import PlanValidationResult
from .retriever import RetrievalResult
from .validator import ValidationResult


@dataclass
class PipelineResult:
    retrieval: RetrievalResult
    request_id: str = ""
    plan: Optional[dict] = None
    planner_prompt: str = ""
    planner_raw: str = ""
    plan_validation: Optional[PlanValidationResult] = None
    sql_prompt: str = ""
    sql_raw: str = ""
    sql: Optional[str] = None
    backend: str = "none"
    gen_note: str = ""
    validation: Optional[ValidationResult] = None
    rows: Optional[List[tuple]] = None
    columns: Optional[List[str]] = None
    run_error: str = ""
    answer: str = ""
    decision: Optional[dict] = None
    entity_matches: List[dict] = field(default_factory=list)
    trace: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    llm_io: List[dict] = field(default_factory=list)


def run_query(sql: str, db_path: Path | None = None, limit: int | None = None, timeout_seconds: int | None = None):
    db_path = Path(db_path or config.DB_PATH)
    limit = limit or config.MAX_RESULT_ROWS
    timeout_seconds = timeout_seconds or config.QUERY_TIMEOUT_SECONDS
    con = sqlite3.connect(db_path)
    start = time.monotonic()

    def _progress() -> int:
        return 1 if time.monotonic() - start > timeout_seconds else 0

    try:
        con.execute("PRAGMA query_only = ON;")
        con.set_progress_handler(_progress, 1000)
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(limit + 1)
        if len(rows) > limit:
            return cols, rows[:limit], f"Result exceeded max rows ({limit}); returned first {limit}."
        return cols, rows, ""
    except sqlite3.Error as exc:
        return [], [], str(exc)
    finally:
        con.close()


def _answer(question: str, columns: Optional[List[str]], rows: Optional[List[tuple]], sql: Optional[str], error: str) -> str:
    if error:
        return f"Tôi không thể chạy truy vấn: {error}"
    if not rows:
        return "Truy vấn không trả về dòng nào."
    if len(rows) == 1 and columns:
        values = ", ".join(f"{col}={val}" for col, val in zip(columns, rows[0]))
        return f"Kết quả cho '{question}': {values}"
    return f"Trả về {len(rows)} dòng cho '{question}'."


def _answer_rows(question: str, columns: Optional[List[str]], rows: Optional[List[tuple]], error: str) -> str:
    if error:
        return f"Toi khong the chay truy van: {error}"
    if not rows:
        return "Truy van khong tra ve dong nao."
    if len(rows) == 1 and columns:
        values = ", ".join(f"{col}={val}" for col, val in zip(columns, rows[0]))
        return f"Ket qua cho '{question}': {values}"
    if columns and len(rows) <= 5:
        snippets = []
        for row in rows:
            snippets.append(", ".join(f"{col}={val}" for col, val in zip(columns, row)))
        return f"Ket qua cho '{question}': " + "; ".join(snippets)
    return f"Tra ve {len(rows)} dong cho '{question}'."


def _trace(result: PipelineResult, stage: str, detail: str) -> None:
    result.trace.append({"stage": stage, "detail": detail})


def _entity_context(matches: list[dict] | None) -> str:
    return entity_resolver.entity_context_text(matches)


def _apply_entity_filters_to_plan(plan: dict | None, matches: list[dict]) -> None:
    if not plan or not matches:
        return
    filters = list(plan.get("filters") or [])
    seen = set()
    for item in filters:
        if not isinstance(item, dict):
            continue
        vals = item.get("values")
        if vals is None and item.get("value") is not None:
            vals = [item.get("value")]
        seen.add((str(item.get("column") or item.get("field") or ""), tuple(str(v) for v in vals or [])))
    for match in matches:
        qcol = str(match.get("qualified_column") or "")
        values = [str(v) for v in match.get("values") or []]
        if not qcol or not values:
            continue
        key = (qcol, tuple(values))
        if key in seen:
            continue
        filters.append({"column": qcol, "op": "IN" if len(values) > 1 else "=", "values": values})
    plan["filters"] = filters


def _log_result(result: PipelineResult) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.LLM_IO_LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "request_id": result.request_id,
        "question": result.retrieval.question,
        "embedding_query": result.retrieval.embedding_query,
        "seed_tables": result.retrieval.seed_tables,
        "expanded_tables": result.retrieval.expanded_tables,
        "plan": result.plan,
        "decision": result.decision,
        "entity_matches": result.entity_matches,
        "trace": result.trace,
        "planner_prompt": result.planner_prompt,
        "planner_raw": result.planner_raw,
        "plan_validation": result.plan_validation.__dict__ if result.plan_validation else None,
        "sql_prompt": result.sql_prompt,
        "sql_raw": result.sql_raw,
        "sql": result.sql,
        "sql_validation": result.validation.__dict__ if result.validation else None,
        "columns": result.columns,
        "rows": result.rows,
        "run_error": result.run_error,
        "warnings": result.warnings,
        "llm_io": result.llm_io,
    }
    path = config.LOG_DIR / f"{result.request_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    llm_path = config.LLM_IO_LOG_DIR / f"{result.request_id}.json"
    llm_payload = {
        "request_id": result.request_id,
        "question": result.retrieval.question,
        "embedding_query": result.retrieval.embedding_query,
        "backend": result.backend,
        "llm_io": result.llm_io,
    }
    llm_path.write_text(json.dumps(llm_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _validate_and_repair_plan(
    result: PipelineResult,
    backend: str,
    history_context: str = "",
    entity_context: str = "",
) -> None:
    if result.plan is None:
        return
    catalog_tables = result.retrieval.expanded_tables
    result.plan_validation = plan_validator.validate_plan(result.plan, catalog_tables)
    attempts = 0
    while not result.plan_validation.valid and attempts < config.PLANNER_REPAIR_ATTEMPTS:
        attempts += 1
        repaired = planner.repair_plan(
            previous_plan=result.plan,
            errors=result.plan_validation.errors,
            user_question=result.retrieval.question,
            skill_md_context=result.retrieval.skill_md_context,
            schema_context=result.retrieval.schema_context,
            allowed_join_graph=result.retrieval.allowed_join_graph,
            backend=backend,
            history_context=history_context,
            entity_context=entity_context,
        )
        if repaired.note:
            result.gen_note = (result.gen_note + " | " + repaired.note).strip(" |")
        result.llm_io.append(
            {
                "stage": "planner_repair",
                "backend": repaired.backend,
                "model": repaired.model,
                "prompt": repaired.prompt,
                "llm_call": repaired.llm_call,
                "raw_response": repaired.raw,
                "note": repaired.note,
            }
        )
        if not repaired.plan:
            break
        result.plan = repaired.plan
        result.planner_raw = repaired.raw or result.planner_raw
        result.plan_validation = plan_validator.validate_plan(result.plan, catalog_tables)


def _validate_and_repair_sql(result: PipelineResult, backend: str) -> None:
    if not result.sql or not result.plan:
        return
    result.validation = validator.validate(result.sql)
    attempts = 0
    while result.validation and not result.validation.ok and attempts < config.SQL_REPAIR_ATTEMPTS:
        attempts += 1
        repaired = sql_writer.repair_sql(
            bad_sql=result.sql,
            errors=result.validation.errors,
            user_question=result.retrieval.question,
            schema_context=result.retrieval.schema_context,
            validated_plan=result.plan,
            backend=backend,
            entity_context=_entity_context(result.entity_matches),
        )
        if repaired.note:
            result.gen_note = (result.gen_note + " | " + repaired.note).strip(" |")
        result.llm_io.append(
            {
                "stage": "sql_repair",
                "backend": repaired.backend,
                "model": repaired.model,
                "prompt": repaired.prompt,
                "llm_call": repaired.llm_call,
                "raw_response": repaired.raw,
                "sql": repaired.sql,
                "note": repaired.note,
            }
        )
        if not repaired.sql:
            break
        result.sql = repaired.sql
        result.sql_prompt = repaired.prompt
        result.sql_raw = repaired.raw or result.sql_raw
        result.validation = validator.validate(result.sql)


def ask(
    question: str,
    backend: Optional[str] = None,
    execute: bool = True,
    gold_sql: Optional[str] = None,
    selected_tables: Optional[List[str]] = None,
    history_context: str = "",
    entity_matches: Optional[List[dict]] = None,
    decision: Optional[dict] = None,
    trace: Optional[List[dict]] = None,
) -> PipelineResult:
    """Run the full RAG-assisted text-to-SQL pipeline for one question."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    backend = (backend or config.PIPELINE_LLM_BACKEND).lower()
    rewrite_result = None
    rewrite_target_tables: list[str] = []
    if selected_tables is None:
        rewrite_result = query_rewriter.rewrite_for_embedding(question, backend=backend, joined_only=True)
        rewrite_target_tables = rewrite_result.target_tables
    r = retriever.retrieve(
        question,
        history_context=history_context,
        embedding_query=rewrite_result.embedding_query if rewrite_result else None,
        selected_tables=selected_tables or rewrite_target_tables or None,
    )
    result = PipelineResult(retrieval=r, request_id=request_id)
    result.backend = backend
    result.decision = decision
    result.trace = list(trace or [])
    _trace(result, "table_retrieval", "Selected candidate tables for the query.")
    if rewrite_result:
        result.llm_io.append(
            {
                "stage": "embedding_query_rewrite",
                "backend": rewrite_result.backend,
                "model": rewrite_result.model,
                "prompt": rewrite_result.prompt,
                "llm_call": rewrite_result.llm_call,
                "raw_response": rewrite_result.raw,
                "tool_args": rewrite_result.tool_args,
                "embedding_query": rewrite_result.embedding_query,
                "target_tables": rewrite_result.target_tables,
                "note": rewrite_result.note,
            }
        )
        if rewrite_result.note:
            result.gen_note = rewrite_result.note

    if entity_matches is None:
        entity_matches = entity_resolver.resolve_entities(
            question,
            candidate_tables=r.expanded_tables,
            preferred_columns=[],
            joined_only=True,
        )
    result.entity_matches = entity_matches or []
    if result.entity_matches:
        _trace(result, "fuzzy_entities", "Resolved fuzzy values from real database values.")
    else:
        _trace(result, "fuzzy_entities", "No fuzzy values needed normalization.")
    entity_context = _entity_context(result.entity_matches)

    plan_result = planner.create_plan(
        user_question=question,
        skill_md_context=r.skill_md_context,
        schema_context=r.schema_context,
        allowed_join_graph=r.allowed_join_graph,
        backend=backend,
        history_context=history_context,
        entity_context=entity_context,
    )
    result.planner_prompt = plan_result.prompt
    result.planner_raw = plan_result.raw or ""
    result.plan = plan_result.plan
    result.llm_io.append(
        {
            "stage": "planner",
            "backend": plan_result.backend,
            "model": plan_result.model,
            "prompt": plan_result.prompt,
            "llm_call": plan_result.llm_call,
            "raw_response": plan_result.raw,
            "plan": plan_result.plan,
            "note": plan_result.note,
        }
    )
    if plan_result.note:
        result.gen_note = (result.gen_note + " | " + plan_result.note).strip(" |")

    _apply_entity_filters_to_plan(result.plan, result.entity_matches)
    _validate_and_repair_plan(result, backend, history_context, entity_context)
    _trace(result, "planning", "Created and validated the SQL plan.")

    if result.plan and result.plan_validation and result.plan_validation.valid:
        sql_result = sql_writer.write_sql(
            user_question=question,
            schema_context=r.schema_context,
            validated_plan=result.plan,
            backend=backend,
            entity_context=entity_context,
        )
        result.sql_prompt = sql_result.prompt
        result.sql = sql_result.sql
        result.sql_raw = sql_result.raw or ""
        result.llm_io.append(
            {
                "stage": "sql_generation",
                "backend": sql_result.backend,
                "model": sql_result.model,
                "prompt": sql_result.prompt,
                "llm_call": sql_result.llm_call,
                "raw_response": sql_result.raw,
                "sql": sql_result.sql,
                "note": sql_result.note,
            }
        )
        if sql_result.note:
            result.gen_note = (result.gen_note + " | " + sql_result.note).strip(" |")
        _validate_and_repair_sql(result, backend)
        _trace(result, "sql_generation", "Generated and validated SQL.")

    sql_to_use = result.sql or gold_sql
    if not result.sql and gold_sql:
        result.gen_note = (result.gen_note + " | using provided reference SQL for validate/run.").strip(" |")
        result.sql = gold_sql
        result.validation = validator.validate(gold_sql)

    if result.plan_validation and result.plan_validation.warnings:
        result.warnings.extend(result.plan_validation.warnings)
    if result.validation and result.validation.warnings:
        result.warnings.extend(result.validation.warnings)

    if sql_to_use and result.validation is None:
        result.validation = validator.validate(sql_to_use)

    if sql_to_use and execute and result.validation and result.validation.ok:
        cols, rows, err = run_query(sql_to_use)
        result.columns, result.rows, result.run_error = cols, rows, err
        result.answer = _answer_rows(question, cols, rows, err)
        _trace(result, "execution", f"Executed query and received {len(rows or [])} rows.")
    elif result.validation and not result.validation.ok:
        result.run_error = "SQL không vượt qua kiểm tra an toàn; truy vấn chưa được chạy."
        result.answer = result.run_error
    elif result.plan_validation and not result.plan_validation.valid:
        result.run_error = "Kế hoạch SQL không hợp lệ; chưa tạo SQL."
        result.answer = result.run_error
    else:
        result.answer = "Chưa tạo được SQL có thể chạy."

    _log_result(result)
    return result
