"""Central configuration. Everything is overridable via environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path

# ---- Optional .env loading (no hard dependency on python-dotenv) -------------
def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv()

# ---- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = Path(os.environ.get("DB_PATH", DATA_DIR / "sales.db"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", DATA_DIR / "schema_index"))
SKILL_DIR = Path(os.environ.get("SKILL_DIR", DATA_DIR / "table_skills"))
CATALOG_PATH = Path(os.environ.get("CATALOG_PATH", DATA_DIR / "schema_catalog.json"))
LOG_DIR = Path(os.environ.get("LOG_DIR", DATA_DIR / "query_logs"))
LLM_IO_LOG_DIR = Path(os.environ.get("LLM_IO_LOG_DIR", DATA_DIR / "llm_io_logs"))
SKILL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
LLM_IO_LOG_DIR.mkdir(exist_ok=True)

# ---- Embedding model --------------------------------------------------------
# "auto"  -> try sentence-transformers with EMBED_MODEL, else fall back to hashing
# "st"    -> force sentence-transformers (errors out if unavailable)
# "hashing" -> deterministic dependency-free fallback (lower quality, but always runs)
EMBEDDER = os.environ.get("EMBEDDER", "auto").lower()
EMBED_MODEL = os.environ.get("EMBED_MODEL", "ibm-granite/granite-embedding-311m-multilingual-r2")
# "auto" -> cuda if torch sees a usable GPU, else cpu. Force with "cuda" / "cpu".
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "auto").lower()

# ---- Retrieval knobs --------------------------------------------------------
TOP_K_CHUNKS = int(os.environ.get("TOP_K_CHUNKS", "15"))   # raw chunk hits to pull
MAX_SEED_TABLES = int(os.environ.get("MAX_SEED_TABLES", "6"))  # candidate tables before FK expansion
MAX_EXPAND_TABLES = int(os.environ.get("MAX_EXPAND_TABLES", "12"))  # safety cap after FK expansion
ROW_SAMPLE_LIMIT = int(os.environ.get("ROW_SAMPLE_LIMIT", "10"))  # row examples embedded per table
SKILL_SAMPLE_LIMIT = int(os.environ.get("SKILL_SAMPLE_LIMIT", "3"))  # rows shown in each skill.md

# ---- Pipeline safety knobs --------------------------------------------------
SQL_DIALECT = os.environ.get("SQL_DIALECT", "sqlite").lower()
QUERY_TIMEOUT_SECONDS = int(os.environ.get("QUERY_TIMEOUT_SECONDS", "30"))
MAX_RESULT_ROWS = int(os.environ.get("MAX_RESULT_ROWS", "500"))
RAW_SELECT_LIMIT = int(os.environ.get("RAW_SELECT_LIMIT", "100"))
EXPLAIN_MAX_SCAN_ROWS = int(os.environ.get("EXPLAIN_MAX_SCAN_ROWS", "100000"))
PLANNER_REPAIR_ATTEMPTS = int(os.environ.get("PLANNER_REPAIR_ATTEMPTS", "1"))
SQL_REPAIR_ATTEMPTS = int(os.environ.get("SQL_REPAIR_ATTEMPTS", "1"))

# ---- LLMs (optional - the RAG retrieval works without them) ------------------
# Full pipeline backend options:
#   none    -> retrieval only, build prompts but do not call an LLM
#   remote  -> call the per-stage OpenAI-compatible chat-completions endpoints below
#   openai  -> call one generic OpenAI-compatible base URL for both stages
#   ollama  -> legacy local Ollama path
#   llamacpp -> legacy local llama.cpp router path (kept for compatibility; setup no longer installs it)
PIPELINE_LLM_BACKEND = os.environ.get("PIPELINE_LLM_BACKEND", os.environ.get("SQL_LLM_BACKEND", "none")).lower()

# Remote OpenAI-compatible chat-completions endpoints. These are full endpoint URLs,
# not base URLs. The Gemma planner and Qwen SQL writer intentionally call different
# servers.
GEMMA_PLANNER_API_URL = os.environ.get("GEMMA_PLANNER_API_URL", "http://192.168.0.5:30185/v1/chat/completions")
QWEN_SQL_API_URL = os.environ.get("QWEN_SQL_API_URL", "http://192.168.0.5:30186/v1/chat/completions")
GEMMA_SKILL_API_URL = os.environ.get("GEMMA_SKILL_API_URL", GEMMA_PLANNER_API_URL)
REMOTE_LLM_API_KEY = os.environ.get("REMOTE_LLM_API_KEY", "")
REMOTE_LLM_TIMEOUT_SECONDS = int(os.environ.get("REMOTE_LLM_TIMEOUT_SECONDS", "600"))

# Model names sent in the request payload. For single-model servers these can be
# simple aliases; the route is selected by GEMMA_PLANNER_API_URL/QWEN_SQL_API_URL.
GEMMA_PLANNER_MODEL = os.environ.get("GEMMA_PLANNER_MODEL", "gemma4-planner")
QWEN_SQL_MODEL = os.environ.get("QWEN_SQL_MODEL", "qwen-sql")
GEMMA_SKILL_MODEL = os.environ.get("GEMMA_SKILL_MODEL", GEMMA_PLANNER_MODEL)
ALLOW_TEMPLATE_SKILL_FALLBACK = os.environ.get("ALLOW_TEMPLATE_SKILL_FALLBACK", "0").lower() in {"1", "true", "yes"}
CHAT_HISTORY_TURNS = int(os.environ.get("CHAT_HISTORY_TURNS", "6"))

# Legacy llama.cpp router settings. They are no longer used by setup.sh/setup.ps1.
LLAMACPP_BASE_URL = os.environ.get("LLAMACPP_BASE_URL", "http://localhost:8888")
LLAMACPP_API_KEY = os.environ.get("LLAMACPP_API_KEY", "")
LLAMA_MANUAL_LOAD = os.environ.get("LLAMA_MANUAL_LOAD", "0").lower() not in {"0", "false", "no"}
LLAMA_MANUAL_UNLOAD = os.environ.get("LLAMA_MANUAL_UNLOAD", "0").lower() not in {"0", "false", "no"}

# Legacy direct SQL generation backend. Kept for CLI compatibility.
# Backends: "none" (just build the prompt), "ollama", "openai", "transformers"
SQL_LLM_BACKEND = os.environ.get("SQL_LLM_BACKEND", "none").lower()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

HF_SQL_MODEL = os.environ.get("HF_SQL_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
