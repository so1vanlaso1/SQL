"""Build the vector index from extracted schema catalog and table skill cards."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from . import config, schema_catalog, skill_cards
from .embedder import get_embedder
from .vectorstore import VectorStore


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


def _column_chunk_text(table: dict, col_name: str, col: dict) -> str:
    text = (
        f"Column {table['name']}.{col_name}: {_one_line(col.get('description', ''))}. "
        f"Table meaning: {_one_line(table.get('description', ''))}."
    )
    values = col.get("common_values") or []
    if values:
        text += " Common values: " + ", ".join(str(v) for v in values[:5]) + "."
    return text


def _row_chunk_text(table: dict, row: dict) -> str:
    desc_by_col = {name: _one_line(col.get("description", "")) for name, col in table["columns"].items()}
    fields = []
    for name in table["columns"]:
        value = row.get(name)
        shown = "NULL" if value is None else repr(value)
        fields.append(f"{name}={shown} ({desc_by_col.get(name, '')})")
    return "\n".join(
        [
            f"Sample row from table {table['name']}: {_one_line(table.get('description', ''))}",
            "Fields: " + "; ".join(fields),
        ]
    )


def build_chunks(
    db_path: Path | None = None,
    row_sample_limit: int | None = None,
) -> Tuple[List[str], List[str], List[dict]]:
    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []
    limit = config.ROW_SAMPLE_LIMIT if row_sample_limit is None else row_sample_limit

    catalog = schema_catalog.extract_catalog(
        db_path=db_path,
        sample_limit=max(limit, config.SKILL_SAMPLE_LIMIT),
    )
    schema_catalog.save_catalog(catalog)
    skill_cards.build_skill_cards(catalog)

    for table_name, table in catalog["tables"].items():
        ids.append(f"table::{table_name}")
        docs.append(skill_cards.embedding_text(catalog, table_name))
        metas.append({"kind": "table", "table": table_name})

        for col_name, col in table["columns"].items():
            ids.append(f"col::{table_name}.{col_name}")
            docs.append(_column_chunk_text(table, col_name, col))
            metas.append({"kind": "column", "table": table_name, "column": col_name})

        for idx, row in enumerate(table.get("sample_rows", [])[:limit], start=1):
            ids.append(f"row::{table_name}.{idx}")
            docs.append(_row_chunk_text(table, row))
            metas.append({"kind": "row", "table": table_name, "row_number": idx})

    return ids, docs, metas


def build_index() -> VectorStore:
    embedder = get_embedder()
    ids, docs, metas = build_chunks()
    row_chunks = sum(1 for m in metas if m["kind"] == "row")
    table_chunks = sum(1 for m in metas if m["kind"] == "table")
    print(
        f"[index] embedding {len(docs)} chunks "
        f"({table_chunks} table skill cards + columns + {row_chunks} row samples, "
        f"max {config.ROW_SAMPLE_LIMIT}/table) ..."
    )
    vectors = embedder.encode(docs)
    store = VectorStore(dim=int(vectors.shape[1]), model_name=getattr(embedder, "model_name", "unknown"))
    store.add(ids, vectors, docs, metas)
    store.save(config.INDEX_DIR)
    print(f"[index] saved {len(ids)} vectors (dim={store.dim}) -> {config.INDEX_DIR}")
    print(f"[index] wrote schema catalog -> {config.CATALOG_PATH}")
    print(f"[index] wrote table skill cards -> {config.SKILL_DIR}")
    return store


if __name__ == "__main__":
    build_index()
