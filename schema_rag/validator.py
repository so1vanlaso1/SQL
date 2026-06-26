"""Validate generated SQL before execution.

The validator is deliberately code-only. LLMs may propose plans and SQL, but this
module decides whether a statement is safe enough to bind, explain, and run.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

from . import config, schema_catalog, schema_def


@dataclass
class ValidationResult:
    ok: bool
    referenced_tables: List[str] = field(default_factory=list)
    unknown_tables: List[str] = field(default_factory=list)
    unknown_columns: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    explain: List[str] = field(default_factory=list)


_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_DANGEROUS_RE = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM)\b", re.I)


def _catalog_sets() -> tuple[Set[str], dict[str, set[str]]]:
    try:
        catalog = schema_catalog.load_catalog()
        tables = set(catalog["tables"].keys())
        cols = {t: set(meta["columns"].keys()) for t, meta in catalog["tables"].items()}
        return tables, cols
    except Exception:
        tables = set(schema_def.all_table_names())
        cols = {t: set(schema_def.columns_of(t)) for t in tables}
        return tables, cols


def _strip_trailing_semicolon(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def _has_statement_chaining(sql: str) -> bool:
    stripped = sql.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1]
    return ";" in stripped


def _starts_readonly(sql: str) -> bool:
    return bool(re.match(r"^\s*(SELECT|WITH)\b", sql, flags=re.IGNORECASE))


def _parse_with_sqlglot(sql: str, res: ValidationResult) -> None:
    try:
        import sqlglot
        from sqlglot import exp
    except Exception:
        res.warnings.append("sqlglot unavailable; using regex + SQLite binding checks only.")
        return
    try:
        parsed = sqlglot.parse(sql, read=config.SQL_DIALECT)
    except Exception as exc:  # noqa: BLE001
        res.errors.append(f"SQL parse error: {exc}")
        return
    if len(parsed) != 1:
        res.errors.append("Only one SQL statement is allowed.")
        return
    root = parsed[0]
    if not isinstance(root, (exp.Select, exp.With, exp.Union)):
        res.errors.append("Only SELECT/WITH/UNION read queries are allowed.")
    for select in root.find_all(exp.Select):
        has_top_level_star = any(isinstance(expr, exp.Star) for expr in select.expressions)
        if has_top_level_star and not root.args.get("limit"):
            res.errors.append("SELECT * without LIMIT is not allowed.")


def _alias_map(sql: str) -> tuple[dict[str, str], list[str]]:
    aliases: dict[str, str] = {}
    referenced: list[str] = []
    pattern = re.compile(
        rf"\b(?:FROM|JOIN)\s+({_IDENT})(?:\s+(?:AS\s+)?({_IDENT}))?",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(sql):
        table = match.group(1)
        alias = match.group(2)
        referenced.append(table)
        if alias and alias.upper() not in {"ON", "WHERE", "JOIN", "GROUP", "ORDER", "LIMIT"}:
            aliases[alias] = table
        aliases[table] = table
    return aliases, referenced


def _static_identifier_check(sql: str, res: ValidationResult) -> None:
    known_tables, known_cols = _catalog_sets()
    aliases, referenced = _alias_map(sql)

    for table in referenced:
        if table in known_tables:
            if table not in res.referenced_tables:
                res.referenced_tables.append(table)
        elif table not in res.unknown_tables:
            res.unknown_tables.append(table)

    for prefix, col in re.findall(rf"\b({_IDENT})\.({_IDENT}|\*)\b", sql):
        if col == "*":
            continue
        table = aliases.get(prefix, prefix)
        if table in known_tables and col not in known_cols[table]:
            res.unknown_columns.append(f"{prefix}.{col}")

    res.unknown_tables = sorted(set(res.unknown_tables))
    res.unknown_columns = sorted(set(res.unknown_columns))
    if res.unknown_tables:
        res.errors.append("Unknown tables: " + ", ".join(res.unknown_tables))
    if res.unknown_columns:
        res.errors.append("Unknown columns: " + ", ".join(res.unknown_columns))


def _binding_check(sql: str, db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(db_path)
    try:
        con.execute("EXPLAIN " + sql)
        return []
    except sqlite3.Error as exc:
        return [str(exc)]
    finally:
        con.close()


def explain_query(sql: str, db_path: Path | None = None) -> tuple[list[str], list[str], list[str]]:
    """Return (plan_lines, warnings, errors) from SQLite EXPLAIN QUERY PLAN."""
    db_path = Path(db_path or config.DB_PATH)
    if not db_path.exists():
        return [], [], []
    warnings: list[str] = []
    errors: list[str] = []
    plan_lines: list[str] = []
    try:
        catalog = schema_catalog.load_catalog()
    except Exception:
        catalog = {"tables": {}}
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        for row in rows:
            detail = str(row[-1])
            plan_lines.append(detail)
            scan = re.search(r"\bSCAN\s+([A-Za-z_][A-Za-z0-9_]*)\b", detail, flags=re.IGNORECASE)
            if scan:
                table_or_alias = scan.group(1)
                row_count = catalog.get("tables", {}).get(table_or_alias, {}).get("row_count")
                if row_count and int(row_count) > config.EXPLAIN_MAX_SCAN_ROWS:
                    errors.append(
                        f"EXPLAIN shows full scan of {table_or_alias} ({row_count} rows), above limit {config.EXPLAIN_MAX_SCAN_ROWS}."
                    )
                elif row_count:
                    warnings.append(f"EXPLAIN shows scan of {table_or_alias} ({row_count} rows).")
                else:
                    warnings.append(f"EXPLAIN shows scan: {detail}")
    except sqlite3.Error as exc:
        errors.append(str(exc))
    finally:
        con.close()
    return plan_lines, warnings, errors


def _looks_like_raw_select(sql: str) -> bool:
    text = sql.upper()
    if re.search(r"\bGROUP\s+BY\b", text) or re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", text):
        return False
    return True


def validate(sql: str, db_path: Path | None = None, require_limit_for_raw: bool = True) -> ValidationResult:
    db_path = Path(db_path or config.DB_PATH)
    res = ValidationResult(ok=True)
    sql = sql.strip()

    if not sql:
        res.errors.append("SQL is empty.")
    if _has_statement_chaining(sql):
        res.errors.append("Semicolon statement chaining is not allowed.")
    if not _starts_readonly(sql):
        res.errors.append("Only SELECT/WITH queries are allowed.")
    if _DANGEROUS_RE.search(sql):
        res.errors.append("Dangerous SQL keyword detected.")

    normalized = _strip_trailing_semicolon(sql)
    _parse_with_sqlglot(normalized, res)
    _static_identifier_check(normalized, res)

    if require_limit_for_raw and _looks_like_raw_select(normalized) and not re.search(r"\bLIMIT\s+\d+\b", normalized, re.I):
        res.errors.append(f"Raw row SELECT queries must include LIMIT <= {config.RAW_SELECT_LIMIT}.")
    limit_match = re.search(r"\bLIMIT\s+(\d+)\b", normalized, re.I)
    if limit_match and int(limit_match.group(1)) > config.MAX_RESULT_ROWS:
        res.errors.append(f"LIMIT {limit_match.group(1)} exceeds max result rows {config.MAX_RESULT_ROWS}.")

    bind_errors = _binding_check(normalized, db_path)
    res.errors.extend(f"bind error: {err}" for err in bind_errors)

    plan, explain_warnings, explain_errors = explain_query(normalized, db_path)
    res.explain = plan
    res.warnings.extend(explain_warnings)
    res.errors.extend(explain_errors)

    if res.errors:
        res.ok = False
    return res
