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

# --- GPU-matched torch (install BEFORE requirements) -------------------------
# The default PyPI `torch` wheel now ships a CUDA 13.0 (cu130) build, which needs
# a 580+ driver. Vast.ai Blackwell boxes (e.g. RTX 5060 Ti, sm_120) commonly run
# a 570.x driver that only supports CUDA 12.8 -> torch reports the driver as "too
# old", torch.cuda.is_available() returns False, and the Granite embedder silently
# falls back to CPU. Installing a cu128 torch first pins a wheel that matches the
# driver AND has sm_120 kernels; the later `pip install -r requirements.txt`
# (sentence-transformers) then keeps this torch instead of pulling cu130.
# Override the channel with TORCH_INDEX_URL, or skip entirely with SKIP_TORCH_INSTALL=1.
if [ "${SKIP_TORCH_INSTALL:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  # Only (re)install when torch is missing or can't actually use the GPU. This avoids
  # re-downloading a multi-GB wheel on a healthy box, while still REPAIRING an existing
  # mismatched install (e.g. a cu130 wheel that a cu128 driver reports as "too old") --
  # a bare `pip install torch` would treat that broken wheel as already satisfied.
  if "$VENV_PY" - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("torch") is None:
    sys.exit(1)  # not installed -> needs install
import torch
sys.exit(0 if torch.cuda.is_available() else 1)  # unusable CUDA -> needs reinstall
PY
  then
    echo "torch already present and CUDA is usable; skipping torch install."
  else
    echo "Installing GPU-matched torch from ${TORCH_INDEX_URL} ..."
    "$VENV_PY" -m pip install --force-reinstall torch --index-url "$TORCH_INDEX_URL"
  fi
fi

"$VENV_PY" -m pip install -r requirements.txt

# --- .env ---
[ -f .env ] || cp .env.example .env

# --- artifact dirs ---
mkdir -p data/table_skills data/schema_index data/query_logs data/llm_io_logs data/runtime_logs models

# --- force full-pipeline runtime config -------------------------------------
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

# Router port is configurable so it can dodge an occupied 8080 on remote hosts.
LLAMA_PORT="${LLAMA_PORT:-8888}"

upsert_env PIPELINE_LLM_BACKEND llamacpp
upsert_env LLAMACPP_BASE_URL "http://127.0.0.1:${LLAMA_PORT}"
upsert_env GEMMA_PLANNER_MODEL gemma4-planner
upsert_env QWEN_SQL_MODEL qwen3.5-sql
upsert_env GEMMA_PLANNER_HF_ID unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
upsert_env QWEN_SQL_HF_ID unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL
upsert_env LLAMA_MANUAL_LOAD 1
# Keep models resident: with --models-max 2 both the planner and SQL model fit in
# 16 GB, so DON'T unload between stages/requests (avoids ~3s reload each call).
upsert_env LLAMA_MANUAL_UNLOAD 0

wait_for_url() {
  local url="$1"
  local name="$2"
  local seconds="${3:-120}"
  "$VENV_PY" - "$url" "$name" "$seconds" <<'PY'
from __future__ import annotations

import sys
import time

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
    except Exception as exc:  # noqa: BLE001
        last = f"{exc.__class__.__name__}: {exc}"
    time.sleep(2)
print(f"{name} did not become ready at {url}. Last error: {last}", file=sys.stderr)
raise SystemExit(1)
PY
}

# --- llama.cpp server -------------------------------------------------------
# vast.ai images often do not ship llama-server on PATH. Use an existing binary
# when available; otherwise clone and build llama.cpp locally.
LLAMA_SERVER_BIN=""

ensure_build_tools() {
  local missing=()
  for tool in git cmake; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done
  if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
    missing+=("g++")
  fi

  if [ ${#missing[@]} -eq 0 ]; then
    return
  fi

  if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
    echo "Installing build tools needed for llama.cpp: ${missing[*]}"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake build-essential
  else
    echo "Missing build tools for llama.cpp: ${missing[*]}"
    echo "Install git, cmake, and a C++ compiler, then rerun setup.sh."
    exit 1
  fi
}

resolve_llama_server() {
  if command -v llama-server >/dev/null 2>&1; then
    LLAMA_SERVER_BIN="$(command -v llama-server)"
    return
  fi
  if [ -x ./llama.cpp/build/bin/llama-server ]; then
    LLAMA_SERVER_BIN="./llama.cpp/build/bin/llama-server"
    return
  fi
  if [ -x ./llama.cpp/build/bin/server ]; then
    LLAMA_SERVER_BIN="./llama.cpp/build/bin/server"
    return
  fi
}

build_llama_cpp() {
  resolve_llama_server
  if [ -n "$LLAMA_SERVER_BIN" ]; then
    echo "Using llama.cpp server: $LLAMA_SERVER_BIN"
    return
  fi

  ensure_build_tools

  if [ ! -d llama.cpp/.git ]; then
    echo "Cloning llama.cpp ..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
  else
    echo "Updating llama.cpp ..."
    git -C llama.cpp pull --ff-only || true
  fi

  local cmake_args=(-S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF)
  if command -v nvidia-smi >/dev/null 2>&1; then
    cmake_args+=(-DGGML_CUDA=ON)
    echo "Building llama.cpp server with CUDA support ..."
    if ! cmake "${cmake_args[@]}"; then
      echo "CUDA configure failed; retrying CPU-only build."
      rm -rf llama.cpp/build
      cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
    fi
  else
    echo "Building llama.cpp server CPU-only ..."
    cmake "${cmake_args[@]}"
  fi
  cmake --build llama.cpp/build --config Release --target llama-server -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

  resolve_llama_server
  if [ -z "$LLAMA_SERVER_BIN" ]; then
    echo "llama.cpp build completed, but llama-server was not found under ./llama.cpp/build/bin."
    exit 1
  fi
  echo "Built llama.cpp server: $LLAMA_SERVER_BIN"
}

if [ "${SKIP_LLAMA_BUILD:-0}" = "1" ]; then
  resolve_llama_server
  LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
  echo "Skipping llama.cpp build because SKIP_LLAMA_BUILD=1."
else
  build_llama_cpp
fi

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

# [*] applies to every model the router loads (gemma planner + qwen sql).
#  - n-gpu-layers = 999 offloads ALL transformer layers to the GPU (0 = CPU-only).
#    The router loads one model at a time (--models-max 1), so a single 9B Q4 + KV
#    cache fits comfortably in the RTX 5060 Ti's 16 GB.
#  - ctx-size = 32768 because the web UI packs skill cards + row samples into prompts
#    that reached ~12k tokens; the old 8192 made the server reject the request with
#    "exceeds the available context size".
preset = [
    "version = 1",
    "",
    "[*]",
    "ctx-size = 32768",
    "n-gpu-layers = 999",
    "threads = -1",
    "",
]
for alias, model_path in entries:
    preset.extend([f"[{alias}]", f"model = {model_path}", "temp = 0", ""])

(models_dir / "models.ini").write_text("\n".join(preset), encoding="utf-8")
print(f"[models] wrote {models_dir / 'models.ini'}")
print("[models] planner alias:", gemma_alias)
print("[models] sql alias:", qwen_alias)
PY

# --- build DB + catalog + skill cards + index ---
"$VENV_PY" -m schema_rag.cli setup

# --- start full stack -------------------------------------------------------
start_llama_router() {
  local log_file="data/runtime_logs/llama-server.log"
  local pid_file="data/runtime_logs/llama-server.pid"
  if wait_for_url "http://127.0.0.1:${LLAMA_PORT}/v1/models" "llama.cpp router" 5 >/dev/null 2>&1; then
    echo "llama.cpp router is already running at http://127.0.0.1:${LLAMA_PORT}"
    return
  fi
  echo "Starting llama.cpp router ..."
  nohup "$LLAMA_SERVER_BIN" \
    --models-preset ./models/models.ini \
    --models-max 2 \
    --sleep-idle-seconds 300 \
    --host 127.0.0.1 \
    --port "$LLAMA_PORT" \
    > "$log_file" 2>&1 &
  echo "$!" > "$pid_file"
  echo "llama.cpp router log: $log_file"
  wait_for_url "http://127.0.0.1:${LLAMA_PORT}/v1/models" "llama.cpp router" 300
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
  start_llama_router
  start_chat_ui
else
  echo "Skipping service startup because START_SERVICES=0."
fi

echo ""
echo "Done."
echo ""
echo "Full pipeline is configured with PIPELINE_LLM_BACKEND=llamacpp."
echo "Chat UI:"
echo "  http://127.0.0.1:${WEB_PORT:-8000}"
echo ""
echo "Runtime logs:"
echo "  data/runtime_logs/llama-server.log"
echo "  data/runtime_logs/web.log"
echo ""
echo "Per-request pipeline and LLM I/O logs:"
echo "  data/query_logs/*.json"
echo "  data/llm_io_logs/*.json"
echo ""
echo "CLI test:"
echo "  $VENV_PY -m schema_rag.cli ask \"Which customer type generated the highest sales in HCM in 2025?\""
