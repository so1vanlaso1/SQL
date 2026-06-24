"""Schema retriever.

    question
      -> embed
      -> vector search schema chunks
      -> aggregate chunk hits up to candidate tables (the SEED tables)
      -> expand seeds via the FK graph (adds bridging tables + join conditions)
      -> assemble a compact "schema pack" (DDL of only the selected tables + joins)

The schema pack is exactly what you hand to the SQL LLM - small, focused, and with the
real join paths spelled out so the model doesn't have to guess relationships.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import json

from . import config, fk_graph, schema_catalog, schema_def, skill_cards
from .embedder import get_embedder
from .vectorstore import Hit, VectorStore


@dataclass
class RetrievalResult:
    question: str
    seed_tables: List[str]                       # chosen purely by vector similarity
    table_scores: Dict[str, float]               # table -> aggregated similarity
    expanded_tables: List[str]                    # seeds + FK bridges
    bridge_tables: List[str]                      # tables added only to connect seeds
    join_edges: List[dict]                        # [{left,right,on}]
    chunk_hits: List[Hit] = field(default_factory=list)
    schema_pack: str = ""
    skill_md_context: str = ""
    schema_context: str = ""
    allowed_join_graph: List[dict] = field(default_factory=list)


def _load_store() -> VectorStore:
    if not VectorStore.exists(config.INDEX_DIR):
        raise FileNotFoundError(
            f"No schema index at {config.INDEX_DIR}. Run:  python -m schema_rag.cli index"
        )
    return VectorStore.load(config.INDEX_DIR)


def _aggregate_to_tables(hits: List[Hit]) -> Dict[str, float]:
    """Best chunk score per table; table, column, and row chunks all count."""
    scores: Dict[str, float] = {}
    for h in hits:
        tbl = h.metadata.get("table")
        if tbl is None:
            continue
        scores[tbl] = max(scores.get(tbl, -1e9), h.score)
    return scores


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


def _description_comments(table: dict) -> str:
    lines = [f"-- Bảng {table['name']}: {_one_line(table['description'])}"]
    for col in table["columns"]:
        lines.append(f"-- Trường {table['name']}.{col['name']}: {_one_line(col['desc'])}")
    return "\n".join(lines)


def build_schema_pack(tables: List[str], join_edges: List[dict]) -> str:
    blocks = ["-- Relevant tables (schema subset) --"]
    for t in tables:
        blocks.append(_description_comments(schema_def.get_table(t)))
        blocks.append(schema_def.ddl_for(t))
    blocks.append("")
    blocks.append("-- Join paths (use these exact relationships for JOINs) --")
    if join_edges:
        for e in join_edges:
            blocks.append(f"{e['on']}")
    else:
        blocks.append("(no foreign-key joins needed between the selected tables)")
    return "\n".join(blocks)


def _schema_context_from_catalog(catalog: dict, tables: List[str]) -> str:
    subset = {"tables": {}, "dialect": catalog.get("dialect", config.SQL_DIALECT)}
    for table in tables:
        if table not in catalog["tables"]:
            continue
        meta = catalog["tables"][table]
        subset["tables"][table] = {
            "columns": {
                name: {
                    "data_type": col.get("data_type"),
                    "primary_key": col.get("primary_key", False),
                    "nullable": col.get("nullable", True),
                }
                for name, col in meta.get("columns", {}).items()
            },
            "primary_key": meta.get("primary_key", []),
            "row_count": meta.get("row_count", 0),
            "indexes": meta.get("indexes", []),
        }
    return json.dumps(subset, ensure_ascii=False, indent=2)


def _allowed_joins(catalog: dict, tables: List[str]) -> List[dict]:
    selected = set(tables)
    joins = []
    for edge in catalog.get("joins", []):
        left_table = str(edge["left"]).split(".", 1)[0]
        right_table = str(edge["right"]).split(".", 1)[0]
        if left_table in selected and right_table in selected:
            joins.append(edge)
    return joins


def retrieve(
    question: str,
    top_k_chunks: int | None = None,
    max_seed_tables: int | None = None,
    max_expand_tables: int | None = None,
    store: VectorStore | None = None,
) -> RetrievalResult:
    top_k_chunks = top_k_chunks or config.TOP_K_CHUNKS
    max_seed_tables = max_seed_tables or config.MAX_SEED_TABLES
    max_expand_tables = max_expand_tables or config.MAX_EXPAND_TABLES

    store = store or _load_store()
    embedder = get_embedder()

    qvec = embedder.encode([question])[0]
    hits = store.search(qvec, k=top_k_chunks)

    table_scores = _aggregate_to_tables(hits)
    seed_tables = sorted(table_scores, key=lambda t: table_scores[t], reverse=True)[:max_seed_tables]

    expansion = fk_graph.expand(seed_tables, max_tables=max_expand_tables)
    expanded_tables: List[str] = expansion["tables"]          # type: ignore[assignment]
    join_edges: List[dict] = expansion["join_edges"]          # type: ignore[assignment]
    bridges: List[str] = expansion["added_bridges"]           # type: ignore[assignment]

    catalog = schema_catalog.load_catalog()
    pack = build_schema_pack(expanded_tables, join_edges)
    skill_context = skill_cards.read_skill_cards(expanded_tables)
    schema_context = _schema_context_from_catalog(catalog, expanded_tables)
    allowed_join_graph = _allowed_joins(catalog, expanded_tables)

    return RetrievalResult(
        question=question,
        seed_tables=seed_tables,
        table_scores=table_scores,
        expanded_tables=expanded_tables,
        bridge_tables=bridges,
        join_edges=join_edges,
        chunk_hits=hits,
        schema_pack=pack,
        skill_md_context=skill_context,
        schema_context=schema_context,
        allowed_join_graph=allowed_join_graph,
    )
