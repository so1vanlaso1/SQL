"""Generic fuzzy entity resolution over live database values.

The resolver is schema-driven: it discovers eligible columns from the catalog
and searches actual distinct values. Neo4j is used when configured, with a local
SQLite fallback so chat remains usable without graph infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import sqlite3
from typing import Any, Iterable

from . import config, schema_catalog
from .vn_text import normalize_vietnamese_text

try:  # rapidfuzz is preferred, but the app should still import without it.
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - exercised only when dependency is absent
    fuzz = None


@dataclass
class EntityValue:
    table: str
    column: str
    value: str
    normalized: str
    data_type: str = ""
    row_count: int = 0


_STOPWORDS = {
    "a",
    "ai",
    "bao",
    "bang",
    "bi",
    "cac",
    "cai",
    "can",
    "chi",
    "cho",
    "co",
    "con",
    "cua",
    "da",
    "den",
    "duoc",
    "gi",
    "gom",
    "hay",
    "la",
    "lai",
    "loc",
    "moi",
    "nao",
    "nay",
    "nhieu",
    "nhung",
    "qua",
    "rieng",
    "so",
    "tat",
    "theo",
    "thi",
    "thu",
    "tong",
    "trong",
    "ve",
    "voi",
}


# Backwards-compatible alias. The implementation now lives in vn_text so retrieval,
# BM25, alias matching, and entity resolution all share one normalizer.
normalize_text = normalize_vietnamese_text


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _score(phrase: str, value: str) -> float:
    if not phrase or not value:
        return 0.0
    if phrase == value:
        return 1.0
    if phrase in value:
        # Prefer prefix/substring matches strongly enough that partial names
        # gather all variants in the same field.
        return 0.96 if value.startswith(phrase) else 0.92
    if fuzz is not None:
        return max(
            fuzz.WRatio(phrase, value),
            fuzz.partial_ratio(phrase, value),
            fuzz.token_set_ratio(phrase, value),
        ) / 100.0
    from difflib import SequenceMatcher

    return SequenceMatcher(None, phrase, value).ratio()


def _looks_identifier_like(text: str) -> bool:
    return bool(re.search(r"\d|_", text)) or bool(re.match(r"^[A-Za-z]{1,6}[-_ ]?\d+", text))


def _eligible_columns(
    tables: Iterable[str] | None = None,
    phrase: str = "",
    joined_only: bool = True,
) -> list[dict[str, Any]]:
    catalog = schema_catalog.load_catalog()
    selected = set(tables or [])
    id_like = _looks_identifier_like(phrase)
    out: list[dict[str, Any]] = []
    for table, meta in catalog.get("tables", {}).items():
        if selected and table not in selected:
            continue
        if joined_only and not table.startswith("jt_"):
            continue
        for column, col_meta in meta.get("columns", {}).items():
            qualified = f"{table}.{column}"
            if column in config.FUZZY_EXCLUDED_COLUMNS or qualified in config.FUZZY_EXCLUDED_COLUMNS:
                continue
            data_type = str(col_meta.get("data_type") or "").upper()
            sample_values = list(col_meta.get("common_values") or [])
            sample_values.extend((row.get(column) for row in meta.get("sample_rows", []) if isinstance(row, dict)))
            has_text_value = any(isinstance(v, str) and v.strip() for v in sample_values)
            text_like = "CHAR" in data_type or "TEXT" in data_type or "CLOB" in data_type or has_text_value
            code_like = id_like and re.search(r"(^|_)id$|ma_|_ma$|code|sku", column, flags=re.I)
            if text_like or code_like:
                out.append(
                    {
                        "table": table,
                        "column": column,
                        "data_type": data_type,
                        "row_count": int(meta.get("row_count") or 0),
                    }
                )
    return out


def _candidate_phrases(question: str) -> list[str]:
    normalized = normalize_text(question)
    tokens = normalized.split()
    phrases: list[str] = []
    for size in range(min(6, len(tokens)), 0, -1):
        for idx in range(0, len(tokens) - size + 1):
            window = tokens[idx : idx + size]
            if all(tok in _STOPWORDS for tok in window):
                continue
            phrase = " ".join(window)
            if len(phrase) < 3:
                continue
            phrases.append(phrase)
    return list(dict.fromkeys(phrases))


def _sqlite_values(columns: list[dict[str, Any]], limit_per_column: int = 5000) -> list[EntityValue]:
    if not config.DB_PATH.exists():
        return []
    con = sqlite3.connect(config.DB_PATH)
    values: list[EntityValue] = []
    try:
        for col in columns:
            table = col["table"]
            column = col["column"]
            try:
                rows = con.execute(
                    f"""
                    SELECT {_quote_ident(column)} AS value, COUNT(*) AS n
                    FROM {_quote_ident(table)}
                    WHERE {_quote_ident(column)} IS NOT NULL
                    GROUP BY {_quote_ident(column)}
                    ORDER BY n DESC
                    LIMIT ?
                    """,
                    (limit_per_column,),
                ).fetchall()
            except sqlite3.Error:
                continue
            for value, n in rows:
                text = str(value).strip()
                if not text:
                    continue
                values.append(
                    EntityValue(
                        table=table,
                        column=column,
                        value=text,
                        normalized=normalize_text(text),
                        data_type=str(col.get("data_type") or ""),
                        row_count=int(n or 0),
                    )
                )
    finally:
        con.close()
    return values


def _neo4j_values(columns: list[dict[str, Any]], phrase: str) -> list[EntityValue]:
    if not (config.NEO4J_ENABLED and config.NEO4J_URI and config.NEO4J_USER):
        return []
    try:
        from neo4j import GraphDatabase
    except Exception:
        return []
    pairs = {(col["table"], col["column"]) for col in columns}
    if not pairs:
        return []
    needle = normalize_text(phrase)
    query = """
    MATCH (e:EntityValue)
    WHERE e.normalized CONTAINS $needle OR $needle CONTAINS e.normalized
    RETURN e.table AS table, e.column AS column, e.value AS value,
           e.normalized AS normalized, e.data_type AS data_type,
           coalesce(e.row_count, 0) AS row_count
    LIMIT 1000
    """
    try:
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        try:
            with driver.session(database=config.NEO4J_DATABASE) as session:
                records = session.run(query, needle=needle).data()
        finally:
            driver.close()
    except Exception:
        return []
    values: list[EntityValue] = []
    for row in records:
        key = (str(row.get("table") or ""), str(row.get("column") or ""))
        if key not in pairs:
            continue
        values.append(
            EntityValue(
                table=key[0],
                column=key[1],
                value=str(row.get("value") or ""),
                normalized=str(row.get("normalized") or ""),
                data_type=str(row.get("data_type") or ""),
                row_count=int(row.get("row_count") or 0),
            )
        )
    return values


def _field_weight(table: str, column: str, preferred_tables: set[str], preferred_columns: set[str]) -> float:
    weight = 0.0
    if table in preferred_tables:
        weight += 0.06
    if column in preferred_columns or f"{table}.{column}" in preferred_columns:
        weight += 0.08
    return weight


def resolve_entities(
    question: str,
    candidate_tables: list[str] | None = None,
    preferred_columns: list[str] | None = None,
    joined_only: bool = True,
) -> list[dict[str, Any]]:
    """Resolve likely literal values mentioned in a question.

    The returned shape is intentionally simple JSON so it can be logged, stored,
    shown in the UI, and injected into LLM prompts.
    """
    phrases = _candidate_phrases(question)
    if not phrases:
        return []
    preferred_tables = set(candidate_tables or [])
    preferred_cols = set(preferred_columns or [])
    min_score = config.FUZZY_MIN_SCORE
    groups: list[dict[str, Any]] = []

    for phrase in phrases:
        columns = _eligible_columns(candidate_tables, phrase=phrase, joined_only=joined_only)
        values = _neo4j_values(columns, phrase) or _sqlite_values(columns)
        candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in values:
            raw_score = _score(phrase, item.normalized)
            if raw_score < min_score:
                continue
            adjusted = min(1.0, raw_score + _field_weight(item.table, item.column, preferred_tables, preferred_cols))
            candidates.setdefault((item.table, item.column), []).append(
                {
                    "value": item.value,
                    "score": round(raw_score, 4),
                    "adjusted_score": round(adjusted, 4),
                    "row_count": item.row_count,
                }
            )

        for (table, column), matches in candidates.items():
            matches = sorted(matches, key=lambda m: (m["adjusted_score"], m["row_count"]), reverse=True)
            if not matches:
                continue
            best = float(matches[0]["adjusted_score"])
            kept = [
                m
                for m in matches
                if float(m["adjusted_score"]) >= best - 0.05 or phrase in normalize_text(m["value"])
            ]
            # Avoid broad one-word dimension terms matching every value in a
            # field, e.g. "tuyen" should not become a filter for every route.
            if (
                len(phrase.split()) == 1
                and len(kept) >= config.FUZZY_MAX_MATCHES
                and not any(normalize_text(m["value"]) == phrase for m in kept)
            ):
                continue
            kept = kept[: config.FUZZY_MAX_MATCHES]
            groups.append(
                {
                    "requested_text": phrase,
                    "table": table,
                    "column": column,
                    "qualified_column": f"{table}.{column}",
                    "score": round(best, 4),
                    "values": [m["value"] for m in kept],
                    "matches": kept,
                    "match_count": len(kept),
                    "ambiguity_group": f"{phrase}:{table}.{column}",
                }
            )

    groups = sorted(groups, key=lambda g: (g["score"], len(str(g["requested_text"]).split()), g["match_count"]), reverse=True)
    chosen: list[dict[str, Any]] = []
    used_phrases: set[str] = set()
    used_columns: set[str] = set()
    for group in groups:
        phrase = str(group["requested_text"])
        qcol = str(group["qualified_column"])
        if any(phrase in used or used in phrase for used in used_phrases):
            continue
        if qcol in used_columns:
            continue
        chosen.append(group)
        used_phrases.add(phrase)
        used_columns.add(qcol)
        if len(chosen) >= 3:
            break
    return chosen


def build_neo4j_index(joined_only: bool = True) -> dict[str, int | str]:
    """Populate Neo4j EntityValue nodes from the live SQLite catalog."""
    if not (config.NEO4J_ENABLED and config.NEO4J_URI and config.NEO4J_USER):
        return {"status": "disabled", "count": 0}
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        return {"status": f"neo4j import failed: {exc}", "count": 0}

    columns = _eligible_columns(joined_only=joined_only)
    values = _sqlite_values(columns)
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    try:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            session.run("CREATE INDEX entity_value_normalized IF NOT EXISTS FOR (e:EntityValue) ON (e.normalized)")
            session.run("CREATE INDEX entity_value_table IF NOT EXISTS FOR (e:EntityValue) ON (e.table)")
            session.run("CREATE INDEX entity_value_column IF NOT EXISTS FOR (e:EntityValue) ON (e.column)")
            for item in values:
                session.run(
                    """
                    MERGE (e:EntityValue {table: $table, column: $column, value: $value})
                    SET e.normalized = $normalized,
                        e.data_type = $data_type,
                        e.row_count = $row_count
                    """,
                    table=item.table,
                    column=item.column,
                    value=item.value,
                    normalized=item.normalized,
                    data_type=item.data_type,
                    row_count=item.row_count,
                )
    finally:
        driver.close()
    return {"status": "ok", "count": len(values)}


def entity_context_text(matches: list[dict[str, Any]] | None) -> str:
    if not matches:
        return "(none)"
    lines = []
    for match in matches:
        values = ", ".join(repr(v) for v in match.get("values", []))
        lines.append(
            f"- {match.get('requested_text')} -> {match.get('qualified_column')} "
            f"IN ({values}) score={match.get('score')}"
        )
    return "\n".join(lines)
