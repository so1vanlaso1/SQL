"""Validate Gemma's structured SQL plan in code."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import schema_catalog


@dataclass
class PlanValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_QUALIFIED_RE = re.compile(rf"\b({_IDENT})\.({_IDENT})\b")
_DANGEROUS_RE = re.compile(r"\b(drop|delete|update|insert|alter|truncate|create|attach|detach|pragma)\b", re.I)
_ALLOWED_FORMULA_CHARS_RE = re.compile(r"^[A-Za-z0-9_.,()\s+\-*/<>=!'%]+$")


def _tables(catalog: dict) -> set[str]:
    return set(catalog["tables"].keys())


def _columns(catalog: dict, table: str) -> set[str]:
    return set(catalog["tables"][table]["columns"].keys())


def _column_exists(catalog: dict, ref: str) -> bool:
    if "." not in ref:
        return False
    table, col = ref.split(".", 1)
    return table in catalog["tables"] and col in catalog["tables"][table]["columns"]


def _column_refs(value: Any) -> list[str]:
    refs = []
    if value is None:
        return refs
    text = str(value)
    for table, col in _QUALIFIED_RE.findall(text):
        refs.append(f"{table}.{col}")
    return refs


def _validate_refs(catalog: dict, refs: Iterable[str], errors: list[str], label: str) -> None:
    for ref in refs:
        if not _column_exists(catalog, ref):
            if "." in ref:
                table, _ = ref.split(".", 1)
                if table in catalog["tables"]:
                    available = ", ".join(sorted(_columns(catalog, table)))
                    errors.append(f"{label}: {ref} does not exist. Available columns in {table}: {available}")
                else:
                    errors.append(f"{label}: {ref} uses unknown table {table}")
            else:
                errors.append(f"{label}: {ref} must be table-qualified")


def validate_plan(plan: dict, allowed_tables: list[str], catalog: dict | None = None) -> PlanValidationResult:
    catalog = catalog or schema_catalog.load_catalog()
    errors: list[str] = []
    warnings: list[str] = []
    real_tables = _tables(catalog)
    allowed_set = set(allowed_tables)

    required = plan.get("required_tables") or []
    if not isinstance(required, list):
        errors.append("required_tables must be a list")
        required = []
    for table in required:
        if table not in real_tables:
            errors.append(f"Unknown required table: {table}")
        elif table not in allowed_set:
            errors.append(f"Required table {table} was not selected by retrieval/join expansion")

    allowed_joins = schema_catalog.join_pairs(catalog)
    for idx, join in enumerate(plan.get("join_plan") or []):
        left = join.get("left")
        right = join.get("right")
        if not left or not right:
            errors.append(f"join_plan[{idx}] must include left and right")
            continue
        _validate_refs(catalog, [left, right], errors, f"join_plan[{idx}]")
        if (left, right) not in allowed_joins:
            errors.append(f"join_plan[{idx}] uses non-allowed join: {left} = {right}")

    for idx, filt in enumerate(plan.get("filters") or []):
        column = filt.get("column")
        if not column:
            errors.append(f"filters[{idx}] missing column")
        else:
            _validate_refs(catalog, [column], errors, f"filters[{idx}]")

    metric_names = set()
    for idx, metric in enumerate(plan.get("metrics") or []):
        name = metric.get("name")
        formula = str(metric.get("formula") or "")
        if name:
            metric_names.add(name)
        if not formula:
            errors.append(f"metrics[{idx}] missing formula")
            continue
        if _DANGEROUS_RE.search(formula) or ";" in formula or "--" in formula or "/*" in formula:
            errors.append(f"metrics[{idx}] contains unsafe SQL tokens")
        if not _ALLOWED_FORMULA_CHARS_RE.match(formula):
            errors.append(f"metrics[{idx}] contains unsupported characters")
        _validate_refs(catalog, _column_refs(formula), errors, f"metrics[{idx}]")

    for idx, ref in enumerate(plan.get("group_by") or []):
        _validate_refs(catalog, [ref], errors, f"group_by[{idx}]")

    for idx, item in enumerate(plan.get("order_by") or []):
        field = item.get("field") if isinstance(item, dict) else item
        if field and field not in metric_names and "." in str(field):
            _validate_refs(catalog, [str(field)], errors, f"order_by[{idx}]")
        direction = str(item.get("direction", "ASC")).upper() if isinstance(item, dict) else "ASC"
        if direction not in {"ASC", "DESC"}:
            errors.append(f"order_by[{idx}] direction must be ASC or DESC")

    limit = plan.get("limit")
    if limit is not None:
        try:
            if int(limit) <= 0:
                errors.append("limit must be positive")
        except (TypeError, ValueError):
            errors.append("limit must be an integer or null")

    return PlanValidationResult(valid=not errors, errors=errors, warnings=warnings)
