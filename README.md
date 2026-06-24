# Vietnamese Schema RAG Demo

This project demonstrates schema RAG for a Vietnamese-style FMCG database. It is not
document RAG. The retriever chooses candidate tables/columns, then the FK graph adds
real join paths.

## Database

The demo builds `data/sales.db` with 20 tables and Vietnamese table names:

```text
cong_ty
vung
nha_phan_phoi
vi_tri
tuyen_ban_hang
nhan_vien
phan_cong_tuyen
loai_khach_hang
khach_hang
nha_phan_phoi_khach_hang
danh_muc_san_pham
san_pham
bang_gia_san_pham
khuyen_mai
khuyen_mai_san_pham
lich_su_vieng_tham
don_hang_ban
chi_tiet_don_hang_ban
don_giao_hang
hang_tra_ve
```

The shape follows the SQL dumps in `data/huhuhhuhuhu`: customers, distributors,
routes, staff, customer visits, sales orders, sales order items, products, promotions,
delivery orders, and returns.

## Pipeline

```text
user question
-> embed question with ibm-granite/granite-embedding-311m-multilingual-r2
-> vector search table skill cards, column chunks, and capped row samples
-> aggregate hits into seed tables
-> expand selected tables through the FK graph
-> pack selected skill.md files, real schema JSON, and allowed joins
-> Gemma planner creates structured JSON plan
-> code validates plan against real tables, columns, and join graph
-> Qwen SQL writer converts validated plan to SQL
-> code validates SQL, runs EXPLAIN, executes with timeout and row cap
-> return answer, rows, optional SQL, and request log
```

Generated offline artifacts:

```text
data/schema_catalog.json       extracted live database schema/catalog
data/table_skills/*.skill.md   one compact table card per table
data/schema_index/*            local vector index
data/query_logs/*.json         request traces
```

## Setup

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\Activate.ps1
python -m schema_rag.cli web
```

Git Bash / WSL / Linux / macOS:

```bash
bash setup.sh
python -m schema_rag.cli web
```

Then open:

```text
http://127.0.0.1:8000
```

## Commands

Rebuild database and index:

```bash
python -m schema_rag.cli setup
```

Run CLI examples:

```bash
python -m schema_rag.cli demo
```

Ask one question:

```bash
python -m schema_rag.cli ask "Which distributors have customers with falling order frequency?"
```

Run with llama.cpp router mode and two GGUF models:

```bash
llama-server --models-dir ./models --models-max 1 --sleep-idle-seconds 300
# set PIPELINE_LLM_BACKEND=llamacpp in .env
python -m schema_rag.cli ask "Which customer type generated the highest sales in HCM in 2025?" --backend llamacpp
```

Start chat UI:

```bash
python -m schema_rag.cli web
```

The UI returns:

- seed tables chosen by vector search
- expanded tables after FK graph connection
- join paths from real foreign keys
- database summary
- row count, metrics, and sample rows for every chosen table

## Embedding Model

`.env` defaults to:

```text
EMBEDDER=auto
EMBED_MODEL=ibm-granite/granite-embedding-311m-multilingual-r2
ROW_SAMPLE_LIMIT=10
SKILL_SAMPLE_LIMIT=3
PIPELINE_LLM_BACKEND=none
LLAMACPP_BASE_URL=http://localhost:8080
GEMMA_PLANNER_MODEL=unsloth/gemma-4-E4B-it-GGUF:UD_Q4_K_XL
QWEN_SQL_MODEL=qwen-sql
```

`EMBEDDER=auto` tries the real Granite embedding model and falls back to a deterministic
hash embedder only if the local environment cannot load the model. For best retrieval,
build the index and run the app with the same embedder setting.

`ROW_SAMPLE_LIMIT` caps how many example rows are embedded from each table during
indexing. `SKILL_SAMPLE_LIMIT` caps rows written into each generated `skill.md` card.

The models are not the safety system. The app validates plans, SQL identifiers,
read-only status, semicolon chaining, result limits, SQLite binding, and EXPLAIN output
in code before execution.
