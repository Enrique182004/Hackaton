# Setup for the SAR Whiteboard Assistant on Windows 10/11.
# Run from the repo root in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\setup_windows.ps1 [-PullModel]
param([switch]$PullModel)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# 1. Python 3.10-3.12 via the py launcher (openai-whisper/torch do not support newer versions yet).
$pyVersion = $null
foreach ($v in "3.12", "3.11", "3.10") {
    & py "-$v" --version 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $pyVersion = $v; break }
}
if (-not $pyVersion) {
    throw "Python 3.10-3.12 not found. Install: winget install Python.Python.3.12"
}
Write-Host "Using Python $pyVersion"

# 2. Virtual environment.
if (-not (Test-Path .venv)) { & py "-$pyVersion" -m venv .venv }
$python = ".\.venv\Scripts\python.exe"

# 3. Python dependencies. PyAudio ships a prebuilt wheel on Windows, so no compiler is needed.
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

# 4. Tesseract OCR (for image uploads in the knowledge base).
if (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
    winget install --id UB-Mannheim.TesseractOCR -e --accept-package-agreements --accept-source-agreements
    Write-Warning "Add 'C:\Program Files\Tesseract-OCR' to PATH, then open a new terminal."
}

# 5. ffmpeg (required by openai-whisper to decode audio).
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
}

# 6. Ollama runtime for the local vision-language model.
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
    Write-Warning "Open a new terminal so 'ollama' is on PATH."
}

# 7. Pull the model (~3 GB). Optional here; can be run later.
if ($PullModel) {
    ollama pull qwen2.5vl:3b
} else {
    Write-Host "Skipping model pull. Run: ollama pull qwen2.5vl:3b"
}

Write-Host @"

Setup complete. Next:
  .\.venv\Scripts\python.exe scripts\hardware_check.py
  `$env:SAR_VLM_BACKEND='mock'; .\.venv\Scripts\python.exe -m biobench_sar.app   # no model needed
  .\.venv\Scripts\python.exe -m biobench_sar.app                                  # real VLM via Ollama
Open http://127.0.0.1:5000/ fullscreen (F11) on the projector, /precharge on your laptop.
"@
