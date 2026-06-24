# Setup for Windows PowerShell.
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
# Creates a venv (prefers Python 3.12 for torch wheels), installs deps,
# builds the SQLite DB, extracts the schema catalog, writes table skill cards,
# and builds the schema vector index.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# --- pick a Python (3.12 preferred; the Granite model needs torch wheels) ---
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
        throw "No Python found. Install Python 3.12 from python.org."
    }
}
Write-Host "Using Python: $($pyExe -join ' ')" -ForegroundColor Cyan

# --- venv ---
if (-not (Test-Path ".venv")) {
    if ($pyExe.Length -eq 1) {
        & $pyExe[0] -m venv .venv
    } else {
        & $pyExe[0] $pyExe[1] -m venv .venv
    }
}
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# --- deps ---
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r requirements.txt

# --- .env ---
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }

# --- artifact dirs ---
New-Item -ItemType Directory -Force -Path "data\table_skills", "data\schema_index", "data\query_logs", "models" | Out-Null

# --- build DB + catalog + skill cards + index ---
& $venvPy -m schema_rag.cli setup

Write-Host "`nDone. Activate with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "Retrieval/web UI:    python -m schema_rag.cli web" -ForegroundColor Green
Write-Host "`nFor llama.cpp router mode:" -ForegroundColor Cyan
Write-Host "  llama-server --models-dir ./models --models-max 1 --sleep-idle-seconds 300"
Write-Host "Then set PIPELINE_LLM_BACKEND=llamacpp in .env and run:"
Write-Host "  python -m schema_rag.cli ask `"Which customer type generated the highest sales in HCM in 2025?`" --backend llamacpp"
