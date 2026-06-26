"""Database management helpers for the local DBMS UI."""
from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

from . import config, index_schema, schema_catalog, skill_cards


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def schema_signature() -> dict[str, str]:
    with connect() as con:
        rows = con.execute(
            """
            SELECT name, type, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    return {str(row["name"]): str(row["sql"] or "") for row in rows}


def list_tables() -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        out = []
        for row in rows:
            name = str(row["name"])
            count = None
            try:
                count = con.execute(f"SELECT COUNT(*) FROM {quote_ident(name)}").fetchone()[0]
            except sqlite3.Error:
                pass
            out.append({"name": name, "type": row["type"], "row_count": count, "chat_enabled": name.startswith("jt_")})
        return out


def table_schema(table: str) -> dict[str, Any]:
    with connect() as con:
        cols = [dict(row) for row in con.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()]
        indexes = [dict(row) for row in con.execute(f"PRAGMA index_list({quote_ident(table)})").fetchall()]
    return {"table": table, "columns": cols, "indexes": indexes}


def table_rows(table: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    with connect() as con:
        cur = con.execute(f"SELECT * FROM {quote_ident(table)} LIMIT ? OFFSET ?", (limit, offset))
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = [list(row) for row in cur.fetchall()]
        count = con.execute(f"SELECT COUNT(*) FROM {quote_ident(table)}").fetchone()[0]
    return {"table": table, "columns": columns, "rows": rows, "row_count": count, "limit": limit, "offset": offset}


def backup_database() -> Path | None:
    if not config.DB_PATH.exists():
        return None
    backup_dir = config.DATA_DIR / "db_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    path = backup_dir / f"sales_{time.strftime('%Y%m%d_%H%M%S')}.db"
    shutil.copy2(config.DB_PATH, path)
    return path


def _changed_tables(before: dict[str, str], after: dict[str, str]) -> list[str]:
    changed = []
    for name, sql in after.items():
        if name.startswith("sys_chat_"):
            continue
        if before.get(name) != sql:
            changed.append(name)
    return sorted(changed)


def _removed_tables(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(name for name in before if name not in after and not name.startswith("sys_chat_"))


def refresh_skills_for_tables(table_names: list[str]) -> list[str]:
    if not table_names:
        return []
    catalog = schema_catalog.extract_catalog()
    schema_catalog.save_catalog(catalog)
    written: list[str] = []
    for table in table_names:
        if table not in catalog.get("tables", {}):
            continue
        content = skill_cards.generate_table_skill_with_gemma(catalog, table, config.SKILL_DIR)
        path = config.SKILL_DIR / f"{table}.skill.md"
        path.write_text(content, encoding="utf-8")
        written.append(str(path))
    index_schema.build_index(generate_skills=False)
    return written


def execute_sql(sql: str) -> dict[str, Any]:
    sql = sql.strip()
    if not sql:
        raise ValueError("SQL không được để trống.")
    before = schema_signature()
    backup_path = backup_database()
    is_read = sql.lstrip().upper().startswith(("SELECT", "WITH", "PRAGMA"))
    result: dict[str, Any] = {
        "columns": [],
        "rows": [],
        "row_count": 0,
        "changed_tables": [],
        "removed_tables": [],
        "skill_files": [],
        "backup_path": str(backup_path) if backup_path else "",
    }
    with connect() as con:
        if is_read:
            cur = con.execute(sql)
            result["columns"] = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchmany(config.MAX_RESULT_ROWS + 1)
            result["rows"] = [list(row) for row in rows[: config.MAX_RESULT_ROWS]]
            result["row_count"] = len(result["rows"])
        else:
            con.executescript(sql)
            con.commit()
            result["row_count"] = con.total_changes
    after = schema_signature()
    changed = _changed_tables(before, after)
    removed = _removed_tables(before, after)
    result["changed_tables"] = changed
    result["removed_tables"] = removed
    for table in removed:
        path = config.SKILL_DIR / f"{table}.skill.md"
        if path.exists():
            path.unlink()
    if changed or removed:
        result["skill_files"] = refresh_skills_for_tables(changed)
        if removed and not changed:
            catalog = schema_catalog.extract_catalog()
            schema_catalog.save_catalog(catalog)
            index_schema.build_index(generate_skills=False)
    return result
