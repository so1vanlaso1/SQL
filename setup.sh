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

# --- GGUF model setup -------------------------------------------------------
# Downloads the exact planner and SQL-writer GGUF files and writes a llama.cpp
# router preset. The app requests the local aliases from GEMMA_PLANNER_MODEL and
# QWEN_SQL_MODEL; the preset maps those aliases to the downloaded files.
echo ""
echo "Setting up GGUF models for llama.cpp router mode ..."
"$VENV_PY" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def load_env(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def split_hf_id(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"HF model id must include repo:quant, got {value!r}")
    repo, quant = value.rsplit(":", 1)
    return repo, quant


def filename_for(repo: str, quant: str) -> str:
    base = repo.rsplit("/", 1)[-1]
    if base.endswith("-GGUF"):
        base = base[:-5]
    return f"{base}-{quant}.gguf"


root = Path.cwd()
env = load_env(root / ".env")
models_dir = root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

gemma_alias = env.get("GEMMA_PLANNER_MODEL", "gemma4-planner")
qwen_alias = env.get("QWEN_SQL_MODEL", "qwen3.5-sql")
gemma_hf = env.get("GEMMA_PLANNER_HF_ID", "unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL")
qwen_hf = env.get("QWEN_SQL_HF_ID", "unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL")

entries = []
for alias, hf_id in [(gemma_alias, gemma_hf), (qwen_alias, qwen_hf)]:
    repo, quant = split_hf_id(hf_id)
    filename = filename_for(repo, quant)
    print(f"[models] downloading {repo}:{quant} -> models/{filename}")
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=models_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    entries.append((alias, f"./models/{filename}"))

preset = ["version = 1", "", "[*]", "ctx-size = 8192", "threads = -1", ""]
for alias, model_path in entries:
    preset.extend([f"[{alias}]", f"model = {model_path}", "temp = 0", ""])

(models_dir / "models.ini").write_text("\n".join(preset), encoding="utf-8")
print(f"[models] wrote {models_dir / 'models.ini'}")
print("[models] planner alias:", gemma_alias)
print("[models] sql alias:", qwen_alias)
PY

# --- build DB + catalog + skill cards + index ---
"$VENV_PY" -m schema_rag.cli setup

echo ""
echo "Done."
echo ""
echo "Run retrieval-only UI:"
echo "  $VENV_PY -m schema_rag.cli web"
echo ""
echo "Run the two-model GGUF pipeline with llama.cpp router:"
echo "  1. Start llama.cpp router mode from this directory:"
echo "     llama-server --models-preset ./models/models.ini --models-max 1 --sleep-idle-seconds 300"
echo "  2. Set PIPELINE_LLM_BACKEND=llamacpp in .env."
echo "  3. Ask a question:"
echo "     $VENV_PY -m schema_rag.cli ask \"Which customer type generated the highest sales in HCM in 2025?\" --backend llamacpp"
