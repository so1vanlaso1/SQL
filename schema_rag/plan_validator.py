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


def _metric_name_formula(metric: dict) -> tuple[str | None, str]:
    """Normalize a metric entry to (name, formula).

    Accepts both the validator-native shape ``{"name", "formula"}`` and the shape an
    LLM planner naturally emits, ``{"field", "aggregation", "alias"}``. In the latter
    case the formula is synthesized as ``AGG(field)`` so a plan that is semantically
    correct is not rejected over a key-name mismatch.
    """
    name = metric.get("name") or metric.get("alias")
    formula = str(metric.get("formula") or "").strip()
    if not formula:
        field_ref = str(metric.get("field") or metric.get("column") or metric.get("col") or "").strip()
        aggregation = str(metric.get("aggregation") or metric.get("agg") or "").strip()
        if field_ref and aggregation:
            formula = f"{aggregation}({field_ref})"
        elif field_ref:
            formula = field_ref
    return name, formula


def _filter_column(filt: dict) -> str | None:
    """A filter's target column under any of the common key names."""
    return filt.get("column") or filt.get("field") or filt.get("col")


def _join_columns(join: dict) -> tuple[str | None, str | None]:
    """Resolve a join entry to its (left, right) qualified column refs.

    Accepts either qualified columns directly in ``left``/``right``
    (e.g. ``"a.x"`` / ``"b.y"``), or the shape an LLM planner commonly emits:
    bare table names in ``left``/``right`` with the column equality carried in an
    ``on`` / ``on_condition`` / ``condition`` string (e.g. ``"a.x = b.y"``).
    """
    left = str(join.get("left") or "").strip()
    right = str(join.get("right") or "").strip()
    if "." in left and "." in right:
        return left, right
    # shape: split table/column keys, e.g. left_table + left_column.
    lt, lc = join.get("left_table"), join.get("left_column")
    rt, rc = join.get("right_table"), join.get("right_column")
    if lt and lc and rt and rc:
        return f"{lt}.{lc}", f"{rt}.{rc}"
    # shape: bare table names in left/right + equality carried in an on-condition.
    on = join.get("on") or join.get("on_condition") or join.get("condition")
    refs = _column_refs(on)
    if len(refs) >= 2:
        return refs[0], refs[1]
    return None, None


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
    # table-pair -> (left_col, right_col), so a join given as bare table names can be
    # resolved to its real FK columns (the json_schema planner emits table names).
    join_by_tables: dict[tuple[str, str], tuple[str, str]] = {}
    for j in catalog.get("joins", []):
        lt = str(j["left"]).split(".", 1)[0]
        rt = str(j["right"]).split(".", 1)[0]
        join_by_tables.setdefault((lt, rt), (j["left"], j["right"]))
        join_by_tables.setdefault((rt, lt), (j["right"], j["left"]))

    for idx, join in enumerate(plan.get("join_plan") or []):
        lcol, rcol = _join_columns(join)
        if not lcol or not rcol:
            # Fall back to resolving bare table names (left="a", right="b") via the
            # catalog's foreign keys.
            lt = str(join.get("left") or join.get("left_table") or "").strip()
            rt = str(join.get("right") or join.get("right_table") or "").strip()
            resolved = join_by_tables.get((lt, rt))
            if resolved:
                lcol, rcol = resolved
        if not lcol or not rcol:
            errors.append(
                f"join_plan[{idx}] must name a real join "
                f"(table.col in left/right, two related tables, or an on_condition like 'a.x = b.y')"
            )
            continue
        _validate_refs(catalog, [lcol, rcol], errors, f"join_plan[{idx}]")
        # join_pairs() already contains both directions, but check the reverse too
        # so a planner that orders the equality either way still validates.
        if (lcol, rcol) not in allowed_joins and (rcol, lcol) not in allowed_joins:
            errors.append(f"join_plan[{idx}] uses non-allowed join: {lcol} = {rcol}")

    for idx, filt in enumerate(plan.get("filters") or []):
        column = _filter_column(filt)
        if not column:
            errors.append(f"filters[{idx}] missing column")
        else:
            _validate_refs(catalog, [column], errors, f"filters[{idx}]")

    metric_names = set()
    for idx, metric in enumerate(plan.get("metrics") or []):
        name, formula = _metric_name_formula(metric)
        if name:
            metric_names.add(name)
        if not formula:
            errors.append(f"metrics[{idx}] missing formula (or field/aggregation)")
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
        field_str = str(field).strip() if field else ""
        # order_by may reference a metric alias (bare word like "total_qty"), a qualified
        # column, or an aggregate expression such as "COUNT(a.b)". Validate only the
        # qualified column refs we can extract from it; a bare alias is fine because the
        # SQL validator + EXPLAIN are the authoritative gate on the final query.
        if field_str and field_str not in metric_names:
            refs = _column_refs(field_str)
            if refs:
                _validate_refs(catalog, refs, errors, f"order_by[{idx}]")
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
