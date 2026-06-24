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

# --- GGUF model setup ---
Write-Host "`nSetting up GGUF models for llama.cpp router mode ..." -ForegroundColor Cyan
@'
from __future__ import annotations

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
'@ | & $venvPy -

# --- build DB + catalog + skill cards + index ---
& $venvPy -m schema_rag.cli setup

Write-Host "`nDone. Activate with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "Retrieval/web UI:    python -m schema_rag.cli web" -ForegroundColor Green
Write-Host "`nFor llama.cpp router mode:" -ForegroundColor Cyan
Write-Host "  llama-server --models-preset ./models/models.ini --models-max 1 --sleep-idle-seconds 300"
Write-Host "Then set PIPELINE_LLM_BACKEND=llamacpp in .env and run:"
Write-Host "  python -m schema_rag.cli ask `"Which customer type generated the highest sales in HCM in 2025?`" --backend llamacpp"
