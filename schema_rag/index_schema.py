"""Build the vector index from extracted schema catalog and table skill cards."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from . import alias_map, bm25_index, config, schema_catalog, skill_cards
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
    chat_only_joined: bool = True,
    use_gemma_for_joined: bool = False,
    generate_skills: bool = True,
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
    if generate_skills:
        skill_cards.build_skill_cards(catalog, use_gemma_for_joined=use_gemma_for_joined)

    index_tables = [
        (table_name, table)
        for table_name, table in catalog["tables"].items()
        if not chat_only_joined or table_name.startswith("jt_")
    ]

    for table_name, table in index_tables:
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


def build_index(use_gemma_for_joined: bool = False, generate_skills: bool = True) -> VectorStore:
    embedder = get_embedder()
    # Hybrid retrieval ranks over base + jt_ tables, so index all tables unless the
    # legacy jt_-only mode is configured.
    chat_only_joined = config.RETRIEVE_JOINED_ONLY
    ids, docs, metas = build_chunks(
        use_gemma_for_joined=use_gemma_for_joined,
        generate_skills=generate_skills,
        chat_only_joined=chat_only_joined,
    )
    row_chunks = sum(1 for m in metas if m["kind"] == "row")
    table_chunks = sum(1 for m in metas if m["kind"] == "table")
    scope = "joined-only" if chat_only_joined else "base + joined"
    print(
        f"[index] embedding {len(docs)} chunks "
        f"({table_chunks} {scope} table skill cards + columns + {row_chunks} row samples, "
        f"max {config.ROW_SAMPLE_LIMIT}/table) ..."
    )
    vectors = embedder.encode(docs)
    store = VectorStore(dim=int(vectors.shape[1]), model_name=getattr(embedder, "model_name", "unknown"))
    store.add(ids, vectors, docs, metas)
    store.save(config.INDEX_DIR)
    print(f"[index] saved {len(ids)} vectors (dim={store.dim}) -> {config.INDEX_DIR}")
    print(f"[index] wrote schema catalog -> {config.CATALOG_PATH}")
    print(f"[index] wrote table skill cards -> {config.SKILL_DIR}")

    # ---- hybrid retrieval artifacts (alias map + BM25 lexical index) ----------
    catalog = schema_catalog.load_catalog()
    alias_path = alias_map.build_and_save(catalog)
    print(f"[index] wrote alias map -> {alias_path}")
    bm25_index.build_bm25_index(ids, docs, metas)
    return store


if __name__ == "__main__":
    build_index()
