"""Generate and load compact per-table skill.md files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import requests

from . import config, joined_tables, schema_catalog


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


def _skill_chat_url() -> str:
    cleaned = str(config.GEMMA_SKILL_API_URL).rstrip("/")
    if cleaned.endswith("/chat/completions") or cleaned.endswith("/v1/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def _strip_markdown_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _source_skill_context(catalog: dict, source_tables: Iterable[str], out_dir: Path) -> str:
    blocks: list[str] = []
    for source in source_tables:
        path = out_dir / f"{source}.skill.md"
        if path.exists():
            blocks.append(path.read_text(encoding="utf-8"))
        elif source in catalog.get("tables", {}):
            blocks.append(render_table_skill(catalog, source))
    return "\n\n---\n\n".join(blocks)


def _generate_joined_skill_with_gemma(catalog: dict, table_name: str, out_dir: Path) -> str:
    meta = catalog["tables"][table_name]
    jt = joined_tables.BY_NAME.get(table_name)
    if not jt:
        raise ValueError(f"{table_name} is not a registered joined table")
    source_context = _source_skill_context(catalog, jt.sources, out_dir)
    table_payload = {
        "name": table_name,
        "purpose": jt.purpose,
        "source_tables": jt.sources,
        "columns": meta.get("columns", {}),
        "row_count": meta.get("row_count", 0),
        "sample_rows": meta.get("sample_rows", [])[: config.SKILL_SAMPLE_LIMIT],
    }
    system = (
        "Bạn là data engineer viết tài liệu skill.md cho hệ thống text-to-SQL. "
        "Chỉ trả về Markdown, không bọc code fence, không giải thích ngoài file."
    )
    user = f"""Hãy tạo file skill.md bằng tiếng Việt cho bảng joined/materialized sau.

Yêu cầu bắt buộc:
- Bắt đầu bằng '# Table: {table_name}'.
- Có các mục: Meaning, Primary key, Important fields, Common joins, Common values, Sample data, Business notes.
- Giải thích bảng này dùng cho câu hỏi nghiệp vụ nào.
- Vì đây là bảng đã join sẵn, ưu tiên nói rằng pipeline nên truy vấn trực tiếp bảng này trước khi tự join nhiều bảng nguồn.
- Mô tả rõ các khóa/dimension quan trọng như ngày, tháng, khách hàng, nhà phân phối, nhân viên, sản phẩm nếu có.
- Không tạo cột không tồn tại.
- Không viết SQL.

Thông tin bảng joined:
{json.dumps(table_payload, ensure_ascii=False, indent=2, default=str)}

Skill.md của các bảng nguồn đã dùng để tạo bảng joined:
{source_context}
"""
    payload = {
        "model": config.GEMMA_SKILL_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 2200,
    }
    headers = {"Content-Type": "application/json"}
    if config.REMOTE_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.REMOTE_LLM_API_KEY}"
    try:
        response = requests.post(
            _skill_chat_url(),
            headers=headers,
            json=payload,
            timeout=config.REMOTE_LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content", "") if isinstance(message, dict) else str(choice.get("text", ""))
        if isinstance(content, list):
            content = "".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        md = _strip_markdown_response(str(content))
        if not md.startswith(f"# Table: {table_name}"):
            raise RuntimeError("Gemma4 skill response did not start with the required table heading")
        return md.rstrip() + "\n"
    except Exception as exc:  # noqa: BLE001
        if config.ALLOW_TEMPLATE_SKILL_FALLBACK:
            return render_table_skill(catalog, table_name)
        raise RuntimeError(
            f"Gemma4 skill generation failed for {table_name}. "
            "Set ALLOW_TEMPLATE_SKILL_FALLBACK=1 to use template fallback."
        ) from exc


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


def generate_table_skill_with_gemma(
    catalog: dict,
    table_name: str,
    out_dir: Path | None = None,
    source_tables: Iterable[str] | None = None,
) -> str:
    """Generate one skill.md with Gemma4 for any table, including user-created jt_ tables."""
    if config.ALLOW_TEMPLATE_SKILL_FALLBACK:
        return render_table_skill(catalog, table_name)
    out_dir = Path(out_dir or config.SKILL_DIR)
    meta = catalog["tables"][table_name]
    jt = joined_tables.BY_NAME.get(table_name)
    purpose = jt.purpose if jt else str(meta.get("description") or f"Bang {table_name} duoc tao trong DBMS.")
    sources = tuple(source_tables or (jt.sources if jt else ()))
    source_context = _source_skill_context(catalog, sources, out_dir)
    table_payload = {
        "name": table_name,
        "purpose": purpose,
        "source_tables": sources,
        "columns": meta.get("columns", {}),
        "row_count": meta.get("row_count", 0),
        "sample_rows": meta.get("sample_rows", [])[: config.SKILL_SAMPLE_LIMIT],
    }
    system = (
        "Ban la data engineer viet tai lieu skill.md cho he thong text-to-SQL. "
        "Chi tra ve Markdown, khong boc code fence, khong giai thich ngoai file."
    )
    user = f"""Hay tao file skill.md bang tieng Viet cho bang sau. Neu ten bang bat dau bang jt_ thi day la bang joined/materialized cho chat text-to-SQL.

Yeu cau bat buoc:
- Bat dau bang '# Table: {table_name}'.
- Co cac muc: Meaning, Primary key, Important fields, Common joins, Common values, Sample data, Business notes.
- Giai thich bang nay dung cho cau hoi nghiep vu nao.
- Neu la bang jt_, noi ro pipeline nen truy van truc tiep bang nay truoc khi tu join nhieu bang nguon.
- Mo ta cac khoa/dimension quan trong nhu ngay, thang, khach hang, nha phan phoi, nhan vien, san pham neu co.
- Khong tao cot khong ton tai.
- Khong viet SQL.

Thong tin bang:
{json.dumps(table_payload, ensure_ascii=False, indent=2, default=str)}

Skill.md cua cac bang nguon, neu co:
{source_context}
"""
    payload = {
        "model": config.GEMMA_SKILL_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 2200,
    }
    headers = {"Content-Type": "application/json"}
    if config.REMOTE_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.REMOTE_LLM_API_KEY}"
    try:
        response = requests.post(
            _skill_chat_url(),
            headers=headers,
            json=payload,
            timeout=config.REMOTE_LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content", "") if isinstance(message, dict) else str(choice.get("text", ""))
        if isinstance(content, list):
            content = "".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        md = _strip_markdown_response(str(content))
        if not md.startswith(f"# Table: {table_name}"):
            raise RuntimeError("Gemma4 skill response did not start with the required table heading")
        return md.rstrip() + "\n"
    except Exception as exc:  # noqa: BLE001
        if config.ALLOW_TEMPLATE_SKILL_FALLBACK:
            return render_table_skill(catalog, table_name)
        raise RuntimeError(
            f"Gemma4 skill generation failed for {table_name}. "
            "Set ALLOW_TEMPLATE_SKILL_FALLBACK=1 to use template fallback."
        ) from exc


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
    use_gemma_for_joined: bool = False,
) -> list[Path]:
    catalog = catalog or schema_catalog.load_catalog()
    out_dir = Path(out_dir or config.SKILL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    table_names = list(catalog["tables"])
    # Base/source skills are written first so Gemma can use them as context for jt_ tables.
    ordered = [t for t in table_names if not t.startswith("jt_")] + [t for t in table_names if t.startswith("jt_")]
    for table_name in ordered:
        path = out_dir / f"{table_name}.skill.md"
        if use_gemma_for_joined and table_name in joined_tables.BY_NAME:
            content = generate_table_skill_with_gemma(catalog, table_name, out_dir)
        else:
            content = render_table_skill(catalog, table_name)
        path.write_text(content, encoding="utf-8")
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
