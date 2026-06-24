"""Generate and load compact per-table skill.md files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from . import config, schema_catalog


def _clean(value: Any) -> str:
    text = " ".join(str(value).split())
    return text if text else "n/a"


def _json_preview(rows: list[dict], limit: int = 3) -> str:
    clipped = rows[:limit]
    return json.dumps(clipped, ensure_ascii=False, indent=2)


def _joins_for_table(catalog: dict, table: str) -> list[str]:
    joins = []
    prefix = f"{table}."
    for edge in catalog.get("joins", []):
        if str(edge["left"]).startswith(prefix) or str(edge["right"]).startswith(prefix):
            joins.append(f"{edge['left']} = {edge['right']} ({edge.get('join_type', 'join')})")
    return joins


def _business_note(table: dict, joins: list[str]) -> str:
    aliases = ", ".join(table.get("aliases", [])[:8])
    fields = ", ".join(list(table.get("columns", {}).keys())[:10])
    parts = [f"Use this table when the question relates to {_clean(table.get('description'))}."]
    if aliases:
        parts.append(f"Useful aliases and keywords: {aliases}.")
    if fields:
        parts.append(f"Important identifiers and fields: {fields}.")
    if joins:
        parts.append("Use only the listed common joins when connecting this table.")
    return " ".join(parts)


def render_table_skill(catalog: dict, table_name: str) -> str:
    table = catalog["tables"][table_name]
    columns = table.get("columns", {})
    pk = ", ".join(table.get("primary_key", [])) or "None declared"
    joins = _joins_for_table(catalog, table_name)

    lines: list[str] = [f"# Table: {table_name}", ""]
    lines.extend(["## Meaning", _clean(table.get("description")), ""])
    lines.extend(["## Primary key", pk, ""])
    lines.append("## Important fields")
    for name, col in columns.items():
        desc = _clean(col.get("description")) if col.get("description") else ""
        type_bits = [col.get("data_type") or "UNKNOWN"]
        if col.get("primary_key"):
            type_bits.append("primary key")
        if not col.get("nullable", True):
            type_bits.append("not null")
        suffix = f": {desc}" if desc else ""
        lines.append(f"- {name} ({', '.join(type_bits)}){suffix}")
    lines.append("")

    lines.append("## Common joins")
    if joins:
        lines.extend(f"- {join}" for join in joins)
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Common values")
    for name, col in columns.items():
        vals = col.get("common_values") or []
        if vals:
            lines.append(f"- {name}: {', '.join(_clean(v) for v in vals[:5])}")
    if lines[-1] == "## Common values":
        lines.append("- None sampled")
    lines.append("")

    lines.append("## Sample data")
    lines.append("```json")
    lines.append(_json_preview(table.get("sample_rows", []), config.SKILL_SAMPLE_LIMIT))
    lines.append("```")
    lines.append("")

    lines.append("## Business notes")
    lines.append(_business_note(table, joins))
    lines.append("")
    return "\n".join(lines)


def embedding_text(catalog: dict, table_name: str) -> str:
    table = catalog["tables"][table_name]
    columns = table.get("columns", {})
    joins = _joins_for_table(catalog, table_name)
    aliases = ", ".join(table.get("aliases", []))
    fields = ", ".join(columns.keys())
    common_value_bits = []
    for name, col in columns.items():
        vals = col.get("common_values") or []
        if vals:
            common_value_bits.append(f"{name}: {', '.join(_clean(v) for v in vals[:5])}")
    return "\n".join(
        part
        for part in [
            f"{table_name} table. {_clean(table.get('description'))}",
            f"Aliases and keywords: {aliases}" if aliases else "",
            f"Fields: {fields}",
            "Common joins: " + "; ".join(joins) if joins else "",
            "Common values: " + "; ".join(common_value_bits) if common_value_bits else "",
            _business_note(table, joins),
        ]
        if part
    )


def build_skill_cards(
    catalog: dict | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    catalog = catalog or schema_catalog.load_catalog()
    out_dir = Path(out_dir or config.SKILL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for table_name in catalog["tables"]:
        path = out_dir / f"{table_name}.skill.md"
        path.write_text(render_table_skill(catalog, table_name), encoding="utf-8")
        written.append(path)
    return written


def read_skill_cards(table_names: Iterable[str], skill_dir: Path | None = None) -> str:
    skill_dir = Path(skill_dir or config.SKILL_DIR)
    blocks = []
    for table in table_names:
        path = skill_dir / f"{table}.skill.md"
        if path.exists():
            blocks.append(path.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(blocks)
