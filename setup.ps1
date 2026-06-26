# Setup for Windows PowerShell.
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
# This setup keeps the embedding model local/GPU-capable, but does NOT install
# llama.cpp and does NOT download local Gemma/Qwen GGUF models. Gemma planner and
# Qwen SQL writer are called through remote OpenAI-compatible chat-completions APIs.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# --- pick a Python (3.12 preferred for torch wheels) ---
$pyExe = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in @("3.12", "3.11", "3.10")) {
        & py "-$v" --version *> $null
        if ($LASTEXITCODE -eq 0) { $pyExe = @("py", "-$v"); break }
    }
}
if (-not $pyExe) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pyExe = @("python")
        Write-Host "WARNING: using default 'python'. If torch fails to install, install Python 3.12." -ForegroundColor Yellow
    } else {
        throw "No Python found. Install Python 3.10-3.12 from python.org."
    }
}
Write-Host "Using Python: $($pyExe -join ' ')" -ForegroundColor Cyan

# --- venv ---
if (-not (Test-Path ".venv")) {
    if ($pyExe.Length -eq 1) { & $pyExe[0] -m venv .venv }
    else { & $pyExe[0] $pyExe[1] -m venv .venv }
}
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "Could not find virtualenv Python at $venvPy" }

# --- deps ---
& $venvPy -m pip install --upgrade pip

# --- GPU-matched torch for the local embedding model ------------------------
if ($env:SKIP_TORCH_INSTALL -ne "1" -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $torchIndex = if ($env:TORCH_INDEX_URL) { $env:TORCH_INDEX_URL } else { "https://download.pytorch.org/whl/cu128" }
    $torchOk = $false
@'
import importlib.util, sys
if importlib.util.find_spec("torch") is None:
    sys.exit(1)
import torch
sys.exit(0 if torch.cuda.is_available() else 1)
'@ | & $venvPy -
    if ($LASTEXITCODE -eq 0) { $torchOk = $true }
    if ($torchOk) {
        Write-Host "torch already present and CUDA is usable; skipping torch install." -ForegroundColor Green
    } else {
        Write-Host "Installing GPU-matched torch from $torchIndex for local embeddings ..." -ForegroundColor Cyan
        & $venvPy -m pip install --force-reinstall torch --index-url $torchIndex
    }
}

& $venvPy -m pip install -r requirements.txt

# --- .env ---
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }

# --- artifact dirs (no local LLM model dir needed) ---
New-Item -ItemType Directory -Force -Path "data\table_skills", "data\schema_index", "data\query_logs", "data\llm_io_logs", "data\runtime_logs" | Out-Null

function Upsert-Env($Key, $Value) {
    $path = ".env"
    $lines = if (Test-Path $path) { Get-Content $path } else { @() }
    $found = $false
    $newLines = foreach ($line in $lines) {
        if ($line -match "^$([regex]::Escape($Key))=") {
            "$Key=$Value"
            $found = $true
        } else {
            $line
        }
    }
    if (-not $found) { $newLines += "$Key=$Value" }
    Set-Content -Path $path -Value $newLines -Encoding UTF8
}

Upsert-Env "PIPELINE_LLM_BACKEND" "remote"
Upsert-Env "GEMMA_PLANNER_API_URL" "http://192.168.0.5:30185/v1/chat/completions"
Upsert-Env "QWEN_SQL_API_URL" "http://192.168.0.5:30186/v1/chat/completions"
Upsert-Env "GEMMA_PLANNER_MODEL" "gemma4-planner"
Upsert-Env "QWEN_SQL_MODEL" "qwen-sql"
Upsert-Env "REMOTE_LLM_TIMEOUT_SECONDS" "600"
Upsert-Env "EMBEDDER" "auto"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { Upsert-Env "EMBED_DEVICE" "cuda" } else { Upsert-Env "EMBED_DEVICE" "auto" }

# Remove old local-model settings from previous .env files if present.
@'
from pathlib import Path
path = Path(".env")
remove = {"GEMMA_PLANNER_HF_ID", "QWEN_SQL_HF_ID", "LLAMA_MANUAL_LOAD", "LLAMA_MANUAL_UNLOAD"}
lines = []
for raw in path.read_text(encoding="utf-8").splitlines():
    if "=" in raw and raw.split("=", 1)[0].strip() in remove:
        continue
    lines.append(raw)
path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
'@ | & $venvPy -

# --- build DB + catalog + skill cards + local embedding index ---
& $venvPy -m schema_rag.cli setup

Write-Host "`nDone. Activate with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "Pipeline backend: remote" -ForegroundColor Green
Write-Host "Gemma planner API: http://192.168.0.5:30185/v1/chat/completions" -ForegroundColor Green
Write-Host "Qwen SQL API:     http://192.168.0.5:30186/v1/chat/completions" -ForegroundColor Green
Write-Host "Embedding model:  local GPU when CUDA is visible" -ForegroundColor Green
Write-Host "Run web UI:       python -m schema_rag.cli web" -ForegroundColor Green
Write-Host "CLI test:         python -m schema_rag.cli ask \"Which customer type generated the highest sales in HCM in 2025?\" --backend remote" -ForegroundColor Green
