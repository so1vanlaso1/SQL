#!/usr/bin/env bash
# Setup for Linux / macOS / Git Bash / WSL / vast.ai.
#   bash setup.sh
#
# This setup keeps the embedding model local/GPU-capable, but does NOT install
# llama.cpp and does NOT download local Gemma/Qwen GGUF models. Gemma planner and
# Qwen SQL writer are called through remote OpenAI-compatible chat-completions APIs.
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

[ ${#PY_CMD[@]} -eq 0 ] && { echo "No Python found. Install Python 3.10-3.12."; exit 1; }
echo "Using Python: $PY_LABEL ($("${PY_CMD[@]}" --version 2>&1))"

# --- venv ---
[ -d .venv ] || "${PY_CMD[@]}" -m venv .venv

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

# --- GPU-matched torch for the local embedding model ------------------------
# The local Granite embedding model uses sentence-transformers/torch. If an NVIDIA
# GPU is visible, install a CUDA 12.8 torch wheel before requirements so embedding
# can run on the local GPU. Override with TORCH_INDEX_URL or skip with
# SKIP_TORCH_INSTALL=1.
if [ "${SKIP_TORCH_INSTALL:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  if "$VENV_PY" - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("torch") is None:
    sys.exit(1)
import torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "torch already present and CUDA is usable; skipping torch install."
  else
    echo "Installing GPU-matched torch from ${TORCH_INDEX_URL} for local embeddings ..."
    "$VENV_PY" -m pip install --force-reinstall torch --index-url "$TORCH_INDEX_URL"
  fi
fi

"$VENV_PY" -m pip install -r requirements.txt

# --- .env ---
[ -f .env ] || cp .env.example .env

# --- artifact dirs (no local LLM model dir needed) ---
mkdir -p data/table_skills data/schema_index data/query_logs data/llm_io_logs data/runtime_logs

# --- remote API runtime config ---------------------------------------------
upsert_env() {
  local key="$1"
  local value="$2"
  if grep -qE "^${key}=" .env; then
    sed -i.bak "s|^${key}=.*|${key}=${value}|" .env
    rm -f .env.bak
  else
    printf "\n%s=%s\n" "$key" "$value" >> .env
  fi
}

upsert_env PIPELINE_LLM_BACKEND remote
upsert_env GEMMA_PLANNER_API_URL "http://192.168.0.5:30185/v1/chat/completions"
upsert_env QWEN_SQL_API_URL "http://192.168.0.5:30186/v1/chat/completions"
upsert_env GEMMA_PLANNER_MODEL gemma4-planner
upsert_env QWEN_SQL_MODEL qwen-sql
upsert_env REMOTE_LLM_TIMEOUT_SECONDS 600
upsert_env EMBEDDER auto
if command -v nvidia-smi >/dev/null 2>&1; then
  upsert_env EMBED_DEVICE cuda
else
  upsert_env EMBED_DEVICE auto
fi

# Remove old local-model settings from previous .env files if present.
"$VENV_PY" - <<'PY'
from pathlib import Path
path = Path(".env")
remove = {
    "GEMMA_PLANNER_HF_ID",
    "QWEN_SQL_HF_ID",
    "LLAMA_MANUAL_LOAD",
    "LLAMA_MANUAL_UNLOAD",
}
lines = []
for raw in path.read_text(encoding="utf-8").splitlines():
    if "=" in raw and raw.split("=", 1)[0].strip() in remove:
        continue
    lines.append(raw)
path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
PY

# --- build DB + catalog + skill cards + local embedding index ---------------
"$VENV_PY" -m schema_rag.cli setup

wait_for_url() {
  local url="$1"
  local name="$2"
  local seconds="${3:-120}"
  "$VENV_PY" - "$url" "$name" "$seconds" <<'PY'
from __future__ import annotations
import sys, time
import requests
url, name, seconds = sys.argv[1], sys.argv[2], int(sys.argv[3])
deadline = time.time() + seconds
last = ""
while time.time() < deadline:
    try:
        resp = requests.get(url, timeout=2)
        if resp.status_code < 500:
            print(f"{name} is ready: {url}")
            raise SystemExit(0)
        last = f"HTTP {resp.status_code}"
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        last = f"{exc.__class__.__name__}: {exc}"
    time.sleep(2)
print(f"{name} did not become ready at {url}. Last error: {last}", file=sys.stderr)
raise SystemExit(1)
PY
}

start_chat_ui() {
  local web_host="${WEB_HOST:-0.0.0.0}"
  local web_port="${WEB_PORT:-8000}"
  local log_file="data/runtime_logs/web.log"
  local pid_file="data/runtime_logs/web.pid"
  if wait_for_url "http://127.0.0.1:${web_port}/api/schema" "chat UI" 5 >/dev/null 2>&1; then
    echo "Chat UI is already running at http://127.0.0.1:${web_port}"
    return
  fi
  echo "Starting chat UI ..."
  nohup "$VENV_PY" -m schema_rag.cli web --host "$web_host" --port "$web_port" > "$log_file" 2>&1 &
  echo "$!" > "$pid_file"
  echo "Chat UI log: $log_file"
  wait_for_url "http://127.0.0.1:${web_port}/api/schema" "chat UI" 60
}

if [ "${START_SERVICES:-1}" = "1" ]; then
  start_chat_ui
else
  echo "Skipping chat UI startup because START_SERVICES=0."
fi

echo ""
echo "Done."
echo ""
echo "Pipeline backend: remote"
echo "Gemma planner API: http://192.168.0.5:30185/v1/chat/completions"
echo "Qwen SQL API:     http://192.168.0.5:30186/v1/chat/completions"
echo "Embedding model:  local (${EMBED_DEVICE:-auto}; setup writes cuda when nvidia-smi is visible)"
echo "Chat UI:          http://127.0.0.1:${WEB_PORT:-8000}"
echo ""
echo "Runtime logs:     data/runtime_logs/web.log"
echo "Pipeline logs:    data/query_logs/*.json and data/llm_io_logs/*.json"
echo ""
echo "CLI test:"
echo "  $VENV_PY -m schema_rag.cli ask \"Which customer type generated the highest sales in HCM in 2025?\" --backend remote"
