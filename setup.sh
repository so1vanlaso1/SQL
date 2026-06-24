#!/usr/bin/env bash
# Setup for Linux / macOS / Git Bash / WSL / vast.ai.
#   bash setup.sh
# Creates a venv, installs deps, builds the SQLite DB, extracts the schema catalog,
# writes table skill cards, and builds the schema vector index.
set -euo pipefail
cd "$(dirname "$0")"

# --- pick a Python (3.12 preferred; avoid 3.14 for torch wheels) ---
PY_LABEL=""
PY_CMD=()

if command -v py >/dev/null 2>&1; then
  for v in 3.12 3.11 3.10; do
    if py "-$v" --version >/dev/null 2>&1; then
      PY_LABEL="py -$v"
      PY_CMD=(py "-$v")
      break
    fi
  done
fi

if [ ${#PY_CMD[@]} -eq 0 ]; then
  for cand in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
      PY_LABEL="$cand"
      PY_CMD=("$cand")
      break
    fi
  done
fi

[ ${#PY_CMD[@]} -eq 0 ] && { echo "No Python found. Install Python 3.12."; exit 1; }
echo "Using Python: $PY_LABEL ($("${PY_CMD[@]}" --version 2>&1))"

# --- venv ---
[ -d .venv ] || "${PY_CMD[@]}" -m venv .venv

# Use the venv interpreter directly. This is more reliable than relying on
# activation to expose a `python` command in Git Bash / Windows shells.
if [ -x .venv/Scripts/python.exe ]; then
  VENV_PY=".venv/Scripts/python.exe"
elif [ -x .venv/bin/python ]; then
  VENV_PY=".venv/bin/python"
else
  echo "Could not find the virtualenv Python executable."
  exit 1
fi

# --- deps ---
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt

# --- .env ---
[ -f .env ] || cp .env.example .env

# --- artifact dirs ---
mkdir -p data/table_skills data/schema_index data/query_logs models

# --- build DB + catalog + skill cards + index ---
"$VENV_PY" -m schema_rag.cli setup

echo ""
echo "Done."
echo ""
echo "Run retrieval-only UI:"
echo "  $VENV_PY -m schema_rag.cli web"
echo ""
echo "Run the two-model GGUF pipeline with llama.cpp router:"
echo "  1. Put GGUF files or preset model aliases under ./models, or edit GEMMA_PLANNER_MODEL/QWEN_SQL_MODEL in .env."
echo "  2. Start llama.cpp router mode:"
echo "     llama-server --models-dir ./models --models-max 1 --sleep-idle-seconds 300"
echo "  3. Set PIPELINE_LLM_BACKEND=llamacpp in .env."
echo "  4. Ask a question:"
echo "     $VENV_PY -m schema_rag.cli ask \"Which customer type generated the highest sales in HCM in 2025?\" --backend llamacpp"
