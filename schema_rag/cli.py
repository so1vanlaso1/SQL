"""Command-line interface for the schema-RAG pipeline.

    python -m schema_rag.cli build-db        # create + populate the SQLite DB
    python -m schema_rag.cli index           # embed schema -> build vector index
    python -m schema_rag.cli ask "..."       # run the pipeline on a question
    python -m schema_rag.cli demo            # run several example questions
    python -m schema_rag.cli setup           # build-db + index in one go
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import build_db, chat_memory, config, entity_resolver, index_schema, webapp
from .pipeline import PipelineResult, ask


def _print_retrieval(res: PipelineResult, show_chunks: bool = False) -> None:
    r = res.retrieval
    print("\n" + "=" * 78)
    print(f"QUESTION: {r.question}")
    print("=" * 78)
    if r.embedding_query != r.question:
        print("\n[0] Gemma embedding rewrite:")
        print(f"      {r.embedding_query}")

    print("\n[1] Vector retrieval -> candidate (seed) tables:")
    for t in r.seed_tables:
        print(f"      {t:<22} score={r.table_scores.get(t, 0):.3f}")

    if show_chunks:
        print("\n    raw chunk hits:")
        for h in r.chunk_hits[:10]:
            print(f"      {h.score:.3f}  {h.doc_id}")

    print("\n[2] FK-graph expansion -> tables actually used:")
    print(f"      {', '.join(r.expanded_tables)}")
    if r.bridge_tables:
        print(f"      (bridge tables added to connect seeds: {', '.join(r.bridge_tables)})")

    print("\n[3] Join paths from real FK relationships:")
    if r.join_edges:
        for e in r.join_edges:
            join_text = e.get("on") or f"{e.get('left')} = {e.get('right')}"
            print(f"      {join_text}")
    else:
        print("      (none needed)")

    print("\n[4] Schema pack handed to the SQL LLM:")
    print("      " + r.schema_pack.replace("\n", "\n      "))


def _print_sql_stage(res: PipelineResult) -> None:
    print(f"\n[5] Gemma planner (backend = {res.backend}):")
    if res.gen_note:
        print(f"      note: {res.gen_note}")
    if res.plan:
        print("      plan:")
        import json

        print("      " + json.dumps(res.plan, ensure_ascii=False, indent=2).replace("\n", "\n      "))
    else:
        print("      (no plan produced - configure PIPELINE_LLM_BACKEND or pass --gold)")

    if res.plan_validation:
        status = "OK" if res.plan_validation.valid else "PROBLEMS"
        print(f"\n[6] Plan validation: {status}")
        for e in res.plan_validation.errors:
            print(f"      error: {e}")
        for w in res.plan_validation.warnings:
            print(f"      warning: {w}")

    print(f"\n[7] Qwen SQL generation:")
    if res.sql:
        print("      SQL:")
        print("      " + res.sql.replace("\n", "\n      "))
    else:
        print("      (no SQL produced)")
        return

    if res.validation:
        v = res.validation
        status = "OK" if v.ok else "PROBLEMS"
        print(f"\n[8] SQL validation + EXPLAIN: {status}")
        if v.referenced_tables:
            print(f"      tables referenced: {', '.join(v.referenced_tables)}")
        if v.unknown_tables:
            print(f"      UNKNOWN tables:   {', '.join(v.unknown_tables)}")
        if v.unknown_columns:
            print(f"      UNKNOWN columns:  {', '.join(v.unknown_columns)}")
        for e in v.errors:
            print(f"      error: {e}")
        for w in v.warnings:
            print(f"      warning: {w}")
        if v.explain:
            print("      explain:")
            for line in v.explain[:8]:
                print(f"        {line}")

    if res.rows is not None:
        print(f"\n[9] Execution: {len(res.rows)} row(s) (showing up to {len(res.rows)})")
        if res.run_error:
            print(f"      run error: {res.run_error}")
        elif res.columns:
            print("      " + " | ".join(res.columns))
            print("      " + "-" * 60)
            for row in res.rows[:15]:
                print("      " + " | ".join(str(x) for x in row))
            if len(res.rows) > 15:
                print(f"      ... ({len(res.rows) - 15} more)")
        if res.answer:
            print(f"\n[10] Answer: {res.answer}")
    elif res.run_error:
        print(f"\n[9] Execution skipped: {res.run_error}")


def cmd_ask(args) -> None:
    res = ask(args.question, backend=args.backend, execute=not args.no_execute, gold_sql=args.gold)
    _print_retrieval(res, show_chunks=args.show_chunks)
    _print_sql_stage(res)
    print()


# A few representative questions for the simulated distribution DB.
DEMO_QUESTIONS: List[dict] = [
    {
        "q": "Which distributors have customers with falling order frequency?",
        "gold": (
            "SELECT npp.ten_nha_phan_phoi, COUNT(DISTINCT kh.khach_hang_id) AS khach_hang_giam_tan_suat\n"
            "FROM nha_phan_phoi npp\n"
            "JOIN nha_phan_phoi_khach_hang map ON map.nha_phan_phoi_id = npp.nha_phan_phoi_id\n"
            "JOIN khach_hang kh ON kh.khach_hang_id = map.khach_hang_id\n"
            "JOIN (\n"
            "  SELECT khach_hang_id,\n"
            "         SUM(CASE WHEN ngay_dat_hang >= '2025-01-01' THEN 1 ELSE 0 END) AS don_gan_day,\n"
            "         SUM(CASE WHEN ngay_dat_hang <  '2025-01-01' THEN 1 ELSE 0 END) AS don_truoc_do\n"
            "  FROM don_hang_ban\n"
            "  WHERE ngay_dat_hang >= '2024-01-01' AND trang_thai != 'CANCELLED'\n"
            "  GROUP BY khach_hang_id\n"
            ") freq ON freq.khach_hang_id = kh.khach_hang_id\n"
            "WHERE freq.don_gan_day < freq.don_truoc_do\n"
            "GROUP BY npp.nha_phan_phoi_id, npp.ten_nha_phan_phoi\n"
            "ORDER BY khach_hang_giam_tan_suat DESC;"
        ),
    },
    {
        "q": "Total sales amount by product category.",
        "gold": (
            "SELECT dm.ten_danh_muc, ROUND(SUM(ct.thanh_tien), 2) AS tong_doanh_so\n"
            "FROM chi_tiet_don_hang_ban ct\n"
            "JOIN san_pham sp ON sp.san_pham_id = ct.san_pham_id\n"
            "JOIN danh_muc_san_pham dm ON dm.danh_muc_id = sp.danh_muc_id\n"
            "GROUP BY dm.danh_muc_id, dm.ten_danh_muc\n"
            "ORDER BY tong_doanh_so DESC;"
        ),
    },
    {
        "q": "Which staff generated the most revenue?",
        "gold": (
            "SELECT nv.ten_nhan_vien, ROUND(SUM(dh.tong_tien), 2) AS doanh_so\n"
            "FROM nhan_vien nv\n"
            "JOIN don_hang_ban dh ON dh.nhan_vien_id = nv.nhan_vien_id\n"
            "WHERE dh.trang_thai != 'CANCELLED'\n"
            "GROUP BY nv.nhan_vien_id, nv.ten_nhan_vien\n"
            "ORDER BY doanh_so DESC\n"
            "LIMIT 10;"
        ),
    },
    {
        "q": "How many visits ended with no order by distributor?",
        "gold": (
            "SELECT npp.ten_nha_phan_phoi, COUNT(*) AS so_lan_khong_co_don\n"
            "FROM lich_su_vieng_tham vt\n"
            "JOIN nha_phan_phoi npp ON npp.nha_phan_phoi_id = vt.nha_phan_phoi_id\n"
            "WHERE vt.ket_qua = 'NO_ORDER'\n"
            "GROUP BY npp.nha_phan_phoi_id, npp.ten_nha_phan_phoi\n"
            "ORDER BY so_lan_khong_co_don DESC;"
        ),
    },
    {
        "q": "Which provinces have the most customers?",
        "gold": (
            "SELECT vt.tinh_thanh, COUNT(*) AS so_khach_hang\n"
            "FROM khach_hang kh\n"
            "JOIN vi_tri vt ON vt.vi_tri_id = kh.vi_tri_id\n"
            "GROUP BY vt.tinh_thanh\n"
            "ORDER BY so_khach_hang DESC;"
        ),
    },
]


def cmd_demo(args) -> None:
    for item in DEMO_QUESTIONS:
        res = ask(item["q"], backend=args.backend, execute=not args.no_execute, gold_sql=item["gold"])
        _print_retrieval(res, show_chunks=args.show_chunks)
        _print_sql_stage(res)
    print("\nDone. (Reference 'gold' SQL is used to demo validate+run when no LLM backend is set.)\n")


def cmd_build_db(args) -> None:
    build_db.build()


def cmd_index(args) -> None:
    index_schema.build_index(use_gemma_for_joined=args.gemma_skills)


def cmd_setup(args) -> None:
    build_db.build()
    chat_memory.init()
    index_schema.build_index(use_gemma_for_joined=True)
    print("\nReady. Try:  python -m schema_rag.cli web")


def cmd_web(args) -> None:
    webapp.serve(host=args.host, port=args.port)


def cmd_entity_index(args) -> None:
    result = entity_resolver.build_neo4j_index(joined_only=not args.all_tables)
    print(result)


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="schema_rag", description="Schema-RAG text-to-SQL pipeline (SQLite).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("build-db", help="create + populate the SQLite database")
    sp.set_defaults(func=cmd_build_db)

    sp = sub.add_parser("index", help="embed schema and build the vector index")
    sp.add_argument("--gemma-skills", action="store_true", help="generate registered jt_ table skills with Gemma4")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("setup", help="build-db + index")
    sp.set_defaults(func=cmd_setup)

    sp = sub.add_parser("web", help="start the local chat UI")
    sp.add_argument("--host", default="127.0.0.1", help="host to bind")
    sp.add_argument("--port", default=8000, type=int, help="port to bind")
    sp.set_defaults(func=cmd_web)

    sp = sub.add_parser("entity-index", help="build the optional Neo4j fuzzy entity index")
    sp.add_argument("--all-tables", action="store_true", help="index all tables instead of only joined chat tables")
    sp.set_defaults(func=cmd_entity_index)

    for name in ("ask", "demo"):
        sp = sub.add_parser(name, help="run the pipeline")
        if name == "ask":
            sp.add_argument("question", help="natural-language question")
            sp.add_argument("--gold", default=None, help="optional reference SQL to validate/run")
        sp.add_argument("--backend", default=None, choices=["none", "remote", "api", "llamacpp", "ollama", "openai"],
                        help="pipeline LLM backend (default from env PIPELINE_LLM_BACKEND, else 'none')")
        sp.add_argument("--no-execute", action="store_true", help="do not run the SQL, just validate")
        sp.add_argument("--show-chunks", action="store_true", help="show raw vector chunk hits")
        sp.set_defaults(func=cmd_ask if name == "ask" else cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
