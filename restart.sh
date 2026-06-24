#!/usr/bin/env bash
# Restart the full LLM stack (llama.cpp router + chat UI) with the latest code.
#
# Stops any router/web that is currently running, then starts both fresh so they
# pick up the newest code on disk and the current models/models.ini (ctx-size,
# n-gpu-layers). Safe to run repeatedly.
#
#   bash restart.sh           # restart with the code currently checked out
#   PULL=1 bash restart.sh    # git pull --ff-only first, then restart
#   LLAMA_PORT=9000 WEB_PORT=8001 bash restart.sh   # override ports
#
set -uo pipefail
cd "$(dirname "$0")"

# --- venv python ------------------------------------------------------------
if [ -x .venv/bin/python ]; then
  VENV_PY=".venv/bin/python"
elif [ -x .venv/Scripts/python.exe ]; then
  VENV_PY=".venv/Scripts/python.exe"
else
  echo "No .venv found. Run setup.sh first."
  exit 1
fi

# --- config (override via env) ----------------------------------------------
LLAMA_PORT="${LLAMA_PORT:-8888}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8000}"
LOG_DIR="data/runtime_logs"
mkdir -p "$LOG_DIR"

# --- optional: pull latest code ---------------------------------------------
if [ "${PULL:-0}" = "1" ]; then
  echo "== Pulling latest code =="
  git pull --ff-only || echo "  (git pull failed - commit/stash local changes first; continuing with on-disk code)"
fi

# --- resolve llama-server binary --------------------------------------------
if command -v llama-server >/dev/null 2>&1; then
  LLAMA_SERVER_BIN="$(command -v llama-server)"
elif [ -x ./llama.cpp/build/bin/llama-server ]; then
  LLAMA_SERVER_BIN="./llama.cpp/build/bin/llama-server"
elif [ -x ./llama.cpp/build/bin/server ]; then
  LLAMA_SERVER_BIN="./llama.cpp/build/bin/server"
else
  echo "llama-server binary not found (run setup.sh to build it)."
  exit 1
fi

# --- stop existing services -------------------------------------------------
stop_pid() {
  local pid_file="$1" name="$2" pid
  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file" 2>/dev/null)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      echo "  stopping $name (pid $pid)"
      kill "$pid" 2>/dev/null
    fi
    rm -f "$pid_file"
  fi
}

echo "== Stopping current LLM stack =="
stop_pid "$LOG_DIR/web.pid" "chat UI"
stop_pid "$LOG_DIR/llama-server.pid" "llama.cpp router"
# Fallback: catch strays (e.g. spawned model instances) not tracked by pid files.
# Patterns are specific so this never matches restart.sh itself.
pkill -f "schema_rag.cli web"        2>/dev/null || true
pkill -f "llama-server --models-preset" 2>/dev/null || true
sleep 3

# --- readiness helper -------------------------------------------------------
wait_for_url() {
  local url="$1" name="$2" seconds="${3:-120}"
  "$VENV_PY" - "$url" "$name" "$seconds" <<'PY'
import sys, time, requests
url, name, seconds = sys.argv[1], sys.argv[2], int(sys.argv[3])
deadline = time.time() + seconds
last = ""
while time.time() < deadline:
    try:
        r = requests.get(url, timeout=2)
        if r.status_code < 500:
            print(f"  {name} ready: {url}")
            raise SystemExit(0)
        last = f"HTTP {r.status_code}"
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        last = f"{exc.__class__.__name__}: {exc}"
    time.sleep(2)
print(f"  {name} did NOT become ready at {url}. Last error: {last}", file=sys.stderr)
raise SystemExit(1)
PY
}

# --- start router -----------------------------------------------------------
echo "== Starting llama.cpp router on 127.0.0.1:${LLAMA_PORT} =="
nohup "$LLAMA_SERVER_BIN" \
  --models-preset ./models/models.ini \
  --models-max 1 \
  --sleep-idle-seconds 300 \
  --host 127.0.0.1 \
  --port "$LLAMA_PORT" \
  > "$LOG_DIR/llama-server.log" 2>&1 &
echo "$!" > "$LOG_DIR/llama-server.pid"
wait_for_url "http://127.0.0.1:${LLAMA_PORT}/v1/models" "llama.cpp router" 300

# --- start chat UI ----------------------------------------------------------
echo "== Starting chat UI on ${WEB_HOST}:${WEB_PORT} =="
nohup "$VENV_PY" -m schema_rag.cli web --host "$WEB_HOST" --port "$WEB_PORT" \
  > "$LOG_DIR/web.log" 2>&1 &
echo "$!" > "$LOG_DIR/web.pid"
wait_for_url "http://127.0.0.1:${WEB_PORT}/api/schema" "chat UI" 60

echo ""
echo "== Done =="
echo "  router : pid $(cat "$LOG_DIR/llama-server.pid")  ->  127.0.0.1:${LLAMA_PORT}"
echo "  web    : pid $(cat "$LOG_DIR/web.pid")  ->  ${WEB_HOST}:${WEB_PORT}"
echo "  logs   : $LOG_DIR/llama-server.log , $LOG_DIR/web.log"
