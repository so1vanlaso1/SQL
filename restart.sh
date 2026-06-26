#!/usr/bin/env bash
# Restart the chat UI with the latest code.
#
# Gemma and Qwen are served by remote OpenAI-compatible APIs, so this script no
# longer starts a local llama.cpp router or local GGUF models.
#
#   bash restart.sh
#   PULL=1 bash restart.sh
#   WEB_PORT=8001 bash restart.sh
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

WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8000}"
LOG_DIR="data/runtime_logs"
mkdir -p "$LOG_DIR"

# --- optional: pull latest code ---------------------------------------------
if [ "${PULL:-0}" = "1" ]; then
  echo "== Pulling latest code =="
  git pull --ff-only || echo "  (git pull failed - commit/stash local changes first; continuing with on-disk code)"
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

echo "== Stopping current chat UI =="
stop_pid "$LOG_DIR/web.pid" "chat UI"
# Clean up older local-router runs created by previous versions of this project.
stop_pid "$LOG_DIR/llama-server.pid" "old llama.cpp router"

# Stop any leftover web processes. On Windows/Git Bash, native python.exe
# processes are not always handled reliably by pkill, so use PowerShell when it
# is available. Limit matching to python processes running this app.
if command -v powershell.exe >/dev/null 2>&1; then
  WEB_PORT="$WEB_PORT" powershell.exe -NoProfile -ExecutionPolicy Bypass -Command '
    $port = [int]$env:WEB_PORT
    Get-CimInstance Win32_Process |
      Where-Object {
        $_.Name -like "python*" -and
        $_.CommandLine -like "*schema_rag.cli*" -and
        $_.CommandLine -like "*web*"
      } |
      ForEach-Object {
        Write-Host ("  stopping chat UI process (pid {0})" -f $_.ProcessId)
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
      }

    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
      ForEach-Object {
        $proc = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $_.OwningProcess) -ErrorAction SilentlyContinue
        if ($proc -and $proc.Name -like "python*") {
          Write-Host ("  stopping python listener on port {0} (pid {1})" -f $port, $_.OwningProcess)
          Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        }
      }
  ' || true
else
  pkill -f "[s]chema_rag.cli web" 2>/dev/null || true
fi

pkill -f "[l]lama-server --models-preset" 2>/dev/null || true
sleep 2

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

# --- start chat UI ----------------------------------------------------------
echo "== Starting chat UI on ${WEB_HOST}:${WEB_PORT} =="
rm -f "$LOG_DIR/web.log" "$LOG_DIR/web.err.log"
nohup "$VENV_PY" -m schema_rag.cli web --host "$WEB_HOST" --port "$WEB_PORT" \
  > "$LOG_DIR/web.log" 2>&1 &
echo "$!" > "$LOG_DIR/web.pid"
wait_for_url "http://127.0.0.1:${WEB_PORT}/api/schema" "chat UI" 60

echo ""
echo "== Done =="
echo "  web             : pid $(cat "$LOG_DIR/web.pid")  ->  ${WEB_HOST}:${WEB_PORT}"
echo "  gemma planner   : http://192.168.0.5:30185/v1/chat/completions"
echo "  qwen sql writer : http://192.168.0.5:30186/v1/chat/completions"
echo "  logs            : $LOG_DIR/web.log"
