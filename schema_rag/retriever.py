"""Schema retriever.

    question
      -> normalize (Vietnamese có dấu -> không dấu)
      -> hybrid candidate search: exact alias + BM25 + vector
      -> fuse signals with Reciprocal Rank Fusion (+ alias/column boosts)
      -> aggregate to candidate tables (the SEED tables)
      -> expand seeds via the FK graph (adds bridging tables + join conditions)
      -> assemble a compact "schema pack" (DDL of only the selected tables + joins)

The schema pack is exactly what you hand to the SQL LLM - small, focused, and with the
real join paths spelled out so the model doesn't have to guess relationships.

Two modes (config.RETRIEVE_JOINED_ONLY):
  * False (default): rank over base + jt_ tables, then FK-expand into a mini-schema.
  * True (legacy):    vector-only over the pre-joined jt_ wide tables, no FK expansion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import json

from . import (
    alias_map as alias_map_mod,
    bm25_index as bm25_mod,
    config,
    fk_graph,
    rrf,
    schema_catalog,
    schema_def,
    skill_cards,
    vn_text,
)
from .embedder import get_embedder
from .vectorstore import Hit, VectorStore


@dataclass
class RetrievalResult:
    question: str
    embedding_query: str                         # text actually embedded for vector retrieval
    seed_tables: List[str]                       # chosen by fused hybrid ranking
    table_scores: Dict[str, float]               # table -> fused score
    expanded_tables: List[str]                    # seeds + FK bridges
    bridge_tables: List[str]                      # tables added only to connect seeds
    join_edges: List[dict]                        # [{left,right,on}]
    chunk_hits: List[Hit] = field(default_factory=list)        # vector hits
    bm25_hits: List[Hit] = field(default_factory=list)         # lexical hits
    alias_hits: List[dict] = field(default_factory=list)       # exact alias/synonym matches
    candidate_columns: List[str] = field(default_factory=list)  # qualified columns surfaced by alias match
    signal_tables: Dict[str, List[str]] = field(default_factory=dict)  # per-signal ranked tables (debug)
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


def _rank_tables_from_hits(hits: List[Hit]) -> List[str]:
    """Best-first list of unique tables from chunk hits (best chunk score per table)."""
    best: Dict[str, float] = {}
    for h in hits:
        tbl = h.metadata.get("table")
        if tbl is None:
            continue
        best[tbl] = max(best.get(tbl, -1e9), h.score)
    return sorted(best, key=lambda t: best[t], reverse=True)


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


def _ddl_from_catalog(table_name: str, meta: dict) -> str:
    lines = [f"CREATE TABLE {table_name} ("]
    col_lines = []
    for name, col in meta.get("columns", {}).items():
        decl = f"    {name} {col.get('data_type') or 'TEXT'}"
        if col.get("primary_key"):
            decl += " PRIMARY KEY"
        col_lines.append(decl)
    lines.append(",\n".join(col_lines))
    lines.append(");")
    return "\n".join(lines)


def build_schema_pack(tables: List[str], join_edges: List[dict]) -> str:
    catalog = schema_catalog.load_catalog()
    blocks = ["-- Relevant tables (schema subset) --"]
    for t in tables:
        if t in catalog.get("tables", {}):
            meta = catalog["tables"][t]
            blocks.append(f"-- Bảng {t}: {_one_line(meta.get('description', ''))}")
            for col_name, col in meta.get("columns", {}).items():
                desc = _one_line(col.get("description", ""))
                if desc:
                    blocks.append(f"-- Trường {t}.{col_name}: {desc}")
            blocks.append(_ddl_from_catalog(t, meta))
        else:
            blocks.append(_description_comments(schema_def.get_table(t)))
            blocks.append(schema_def.ddl_for(t))
    blocks.append("")
    blocks.append("-- Join paths (use these exact relationships for JOINs) --")
    if join_edges:
        for e in join_edges:
            blocks.append(str(e.get("on") or f"{e['left']} = {e['right']}"))
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


def _hybrid_candidates(
    normalized_query: str,
    vector_hits: List[Hit],
    boost_tables: list[str] | None,
) -> tuple[Dict[str, float], List[Hit], List[dict], List[str], Dict[str, List[str]]]:
    """Run BM25 + alias signals and fuse them with the vector ranking via RRF.

    Returns (table_scores, bm25_hits, alias_hits, candidate_columns, signal_tables).
    """
    vector_ranking = _rank_tables_from_hits(vector_hits)
    rankings: List[List[str]] = [vector_ranking]
    signal_tables: Dict[str, List[str]] = {"vector": vector_ranking}

    bm25_hits: List[Hit] = []
    if config.ENABLE_BM25:
        index = bm25_mod.load_bm25_index()
        if index is not None:
            bm25_hits = index.search(normalized_query, k=config.BM25_TOP_K)
            bm25_ranking = _rank_tables_from_hits(bm25_hits)
            if bm25_ranking:
                rankings.append(bm25_ranking)
                signal_tables["bm25"] = bm25_ranking

    alias_hits: List[dict] = []
    candidate_columns: List[str] = []
    boosts: Dict[str, float] = {}
    if config.ENABLE_ALIAS_MATCH:
        alias_hits = alias_map_mod.lookup(normalized_query)
        alias_tables: List[str] = []
        for hit in alias_hits:
            phrase_len = int(hit.get("phrase_len", 1))
            if hit.get("type") == "table":
                table = str(hit.get("identifier"))
                boosts[table] = boosts.get(table, 0.0) + config.ALIAS_MATCH_BOOST * phrase_len
                if table not in alias_tables:
                    alias_tables.append(table)
            elif hit.get("type") == "column":
                qualified = str(hit.get("identifier"))
                table = str(hit.get("table") or qualified.split(".", 1)[0])
                boosts[table] = boosts.get(table, 0.0) + config.COLUMN_MATCH_BOOST * phrase_len
                if qualified not in candidate_columns:
                    candidate_columns.append(qualified)
                if table not in alias_tables:
                    alias_tables.append(table)
        if alias_tables:
            signal_tables["alias"] = alias_tables

    # The LLM rewriter's confident table guesses act as a soft alias-strength boost.
    for table in boost_tables or []:
        boosts[table] = boosts.get(table, 0.0) + config.ALIAS_MATCH_BOOST

    table_scores = rrf.fuse(rankings, k=config.RRF_K, boosts=boosts)
    return table_scores, bm25_hits, alias_hits, candidate_columns, signal_tables


def retrieve(
    question: str,
    history_context: str = "",
    embedding_query: str | None = None,
    selected_tables: list[str] | None = None,
    top_k_chunks: int | None = None,
    max_seed_tables: int | None = None,
    max_expand_tables: int | None = None,
    store: VectorStore | None = None,
    joined_only: bool = True,
    boost_tables: list[str] | None = None,
) -> RetrievalResult:
    top_k_chunks = top_k_chunks or config.TOP_K_CHUNKS
    max_seed_tables = max_seed_tables or config.MAX_SEED_TABLES
    max_expand_tables = max_expand_tables or config.MAX_EXPAND_TABLES

    catalog = schema_catalog.load_catalog()
    hits: List[Hit] = []
    bm25_hits: List[Hit] = []
    alias_hits: List[dict] = []
    candidate_columns: List[str] = []
    signal_tables: Dict[str, List[str]] = {}

    if selected_tables is not None:
        retrieval_text = embedding_query or question
        chosen = [
            t
            for t in dict.fromkeys(selected_tables)
            if t in catalog.get("tables", {}) and (not joined_only or t.startswith("jt_"))
        ]
        if not chosen:
            raise ValueError("No valid selected tables were provided.")
        seed_tables = chosen
        table_scores = {t: 1.0 for t in chosen}
        if joined_only:
            expanded_tables = chosen
            bridges: List[str] = []
            join_edges = _allowed_joins(catalog, expanded_tables)
        else:
            expansion = fk_graph.expand(chosen, max_tables=max_expand_tables)
            expanded_tables = expansion["tables"]          # type: ignore[assignment]
            join_edges = expansion["join_edges"]           # type: ignore[assignment]
            bridges = expansion["added_bridges"]            # type: ignore[assignment]
    else:
        store = store or _load_store()
        embedder = get_embedder()

        retrieval_text = embedding_query or question
        if history_context and embedding_query is None:
            retrieval_text = f"Lịch sử hội thoại liên quan:\n{history_context}\n\nCâu hỏi hiện tại:\n{question}"
        qvec = embedder.encode([retrieval_text])[0]
        hits = store.search(qvec, k=top_k_chunks)

        if joined_only:
            # Legacy path: vector-only over the pre-joined jt_ tables, no FK expansion.
            table_scores = _aggregate_to_tables(hits)
            table_scores = {t: s for t, s in table_scores.items() if t.startswith("jt_")}
            seed_tables = sorted(table_scores, key=lambda t: table_scores[t], reverse=True)[:max_seed_tables]
            signal_tables = {"vector": seed_tables}
            expanded_tables = seed_tables
            bridges = []
            join_edges = _allowed_joins(catalog, expanded_tables)
        else:
            # Hybrid path: alias + BM25 + vector -> RRF -> FK-graph mini-schema.
            normalized_query = vn_text.normalize_vietnamese_text(retrieval_text)
            table_scores, bm25_hits, alias_hits, candidate_columns, signal_tables = _hybrid_candidates(
                normalized_query, hits, boost_tables
            )
            table_scores = {t: s for t, s in table_scores.items() if t in catalog.get("tables", {})}
            seed_tables = sorted(table_scores, key=lambda t: table_scores[t], reverse=True)[:max_seed_tables]
            expansion = fk_graph.expand(seed_tables, max_tables=max_expand_tables)
            expanded_tables = expansion["tables"]          # type: ignore[assignment]
            join_edges = expansion["join_edges"]           # type: ignore[assignment]
            bridges = expansion["added_bridges"]            # type: ignore[assignment]

    pack = build_schema_pack(expanded_tables, join_edges)
    skill_context = skill_cards.read_skill_cards(expanded_tables)
    schema_context = _schema_context_from_catalog(catalog, expanded_tables)
    allowed_join_graph = _allowed_joins(catalog, expanded_tables)

    return RetrievalResult(
        question=question,
        embedding_query=retrieval_text,
        seed_tables=seed_tables,
        table_scores=table_scores,
        expanded_tables=expanded_tables,
        bridge_tables=bridges,
        join_edges=join_edges,
        chunk_hits=hits,
        bm25_hits=bm25_hits,
        alias_hits=alias_hits,
        candidate_columns=candidate_columns,
        signal_tables=signal_tables,
        schema_pack=pack,
        skill_md_context=skill_context,
        schema_context=schema_context,
        allowed_join_graph=allowed_join_graph,
    )
