"""Extract a real schema catalog from the SQLite database.

The catalog is the runtime source of truth for tables, columns, indexes, row
counts, samples, and common values. Descriptions and aliases are enriched from
schema_def when available, but identifiers come from the live database.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from . import config, joined_tables, schema_def


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _schema_meta(table: str) -> dict:
    if table in joined_tables.BY_NAME:
        jt = joined_tables.BY_NAME[table]
        return {
            "name": table,
            "description": jt.purpose,
            "aliases": [
                table,
                table.replace("_", " "),
                "bang da join",
                "joined table",
                "feature table",
            ],
            "columns": [],
            "foreign_keys": [],
        }
    try:
        return schema_def.get_table(table)
    except KeyError:
        return {"name": table, "description": "", "aliases": [], "columns": [], "foreign_keys": []}


def _column_desc(meta: dict, column: str) -> str:
    for col in meta.get("columns", []):
        if col.get("name") == column:
            return str(col.get("desc", ""))
    return ""


def _common_values(con: sqlite3.Connection, table: str, column: str, limit: int = 5) -> list[Any]:
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
            (limit,),
        ).fetchall()
    except sqlite3.Error:
        return []
    return [row["value"] for row in rows]


def _sample_rows(con: sqlite3.Connection, table: str, limit: int) -> list[dict]:
    try:
        rows = con.execute(f"SELECT * FROM {_quote_ident(table)} LIMIT ?", (limit,)).fetchall()
    except sqlite3.Error:
        return []
    return [dict(row) for row in rows]


def extract_catalog(
    db_path: Path | None = None,
    sample_limit: int | None = None,
    common_value_limit: int = 5,
) -> dict:
    db_path = Path(db_path or config.DB_PATH)
    sample_limit = config.SKILL_SAMPLE_LIMIT if sample_limit is None else sample_limit
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        table_rows = con.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        tables: dict[str, dict] = {}
        for table_row in table_rows:
            table = str(table_row["name"])
            meta = _schema_meta(table)
            columns: dict[str, dict] = {}
            pk_cols: list[str] = []
            for col in con.execute(f"PRAGMA table_info({_quote_ident(table)})").fetchall():
                name = str(col["name"])
                is_pk = bool(col["pk"])
                if is_pk:
                    pk_cols.append(name)
                columns[name] = {
                    "name": name,
                    "data_type": str(col["type"] or ""),
                    "primary_key": is_pk,
                    "nullable": not bool(col["notnull"]),
                    "default": col["dflt_value"],
                    "description": _column_desc(meta, name),
                    "common_values": _common_values(con, table, name, common_value_limit),
                }

            foreign_keys = []
            for fk in con.execute(f"PRAGMA foreign_key_list({_quote_ident(table)})").fetchall():
                foreign_keys.append(
                    {
                        "column": str(fk["from"]),
                        "references_table": str(fk["table"]),
                        "references_column": str(fk["to"]),
                    }
                )

            indexes = []
            for idx in con.execute(f"PRAGMA index_list({_quote_ident(table)})").fetchall():
                idx_name = str(idx["name"])
                idx_cols = [
                    str(row["name"])
                    for row in con.execute(f"PRAGMA index_info({_quote_ident(idx_name)})").fetchall()
                ]
                indexes.append({"name": idx_name, "unique": bool(idx["unique"]), "columns": idx_cols})

            row_count = con.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()[0]
            tables[table] = {
                "name": table,
                "description": str(meta.get("description", "")),
                "aliases": list(meta.get("aliases", [])),
                "columns": columns,
                "primary_key": pk_cols,
                "foreign_keys": foreign_keys,
                "indexes": indexes,
                "row_count": int(row_count),
                "sample_rows": _sample_rows(con, table, sample_limit),
            }
    finally:
        con.close()

    joins = []
    for table in tables.values():
        for fk in table["foreign_keys"]:
            joins.append(
                {
                    "left": f"{table['name']}.{fk['column']}",
                    "right": f"{fk['references_table']}.{fk['references_column']}",
                    "join_type": "many_to_one",
                }
            )
    common_join_keys = {
        "don_hang_id",
        "khach_hang_id",
        "nha_phan_phoi_id",
        "nhan_vien_id",
        "tuyen_id",
        "san_pham_id",
        "danh_muc_id",
        "vung_id",
        "ngay",
        "thang",
        "ngay_dat_hang",
        "thang_dat_hang",
    }
    jt_names = [name for name in tables if name.startswith("jt_")]
    for i, left_name in enumerate(jt_names):
        left_cols = set(tables[left_name]["columns"])
        for right_name in jt_names[i + 1 :]:
            shared = sorted((left_cols & set(tables[right_name]["columns"])) & common_join_keys)
            for col in shared:
                joins.append(
                    {
                        "left": f"{left_name}.{col}",
                        "right": f"{right_name}.{col}",
                        "join_type": "feature_key",
                    }
                )

    return {"dialect": "sqlite", "database": str(db_path), "tables": tables, "joins": joins}


def save_catalog(catalog: dict, path: Path | None = None) -> Path:
    path = Path(path or config.CATALOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_catalog(path: Path | None = None, rebuild_if_missing: bool = True) -> dict:
    path = Path(path or config.CATALOG_PATH)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if not rebuild_if_missing:
        raise FileNotFoundError(f"Schema catalog not found at {path}")
    catalog = extract_catalog()
    save_catalog(catalog, path)
    return catalog


def table_names(catalog: dict | None = None) -> list[str]:
    catalog = catalog or load_catalog()
    return list(catalog["tables"].keys())


def columns_of(table: str, catalog: dict | None = None) -> list[str]:
    catalog = catalog or load_catalog()
    return list(catalog["tables"][table]["columns"].keys())


def join_pairs(catalog: dict | None = None) -> set[tuple[str, str]]:
    catalog = catalog or load_catalog()
    pairs = set()
    for join in catalog["joins"]:
        pairs.add((join["left"], join["right"]))
        pairs.add((join["right"], join["left"]))
    return pairs


if __name__ == "__main__":
    out = save_catalog(extract_catalog())
    print(f"[catalog] wrote {out}")
