# Setup for the SAR Whiteboard Assistant on Windows 10/11 (Windows PowerShell 5.1 or PowerShell 7).
# Run from the repo root in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\setup_windows.ps1 [-PullModel]
param([switch]$PullModel)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Windows PowerShell 5.1 turns a program's stderr into error records whenever it is redirected or captured
# (`py -3.12 --version 2>$null`, pip and winget warnings, a log file), and with "Stop" the first one ends the
# script. So programs run with "Continue" and are judged by their exit code; PowerShell 7 behaves the same here.
$script:NativeExitCode = 0

function Invoke-Native {
    # Runs a program with its output on the console; its exit code lands in $script:NativeExitCode (-1: did not start).
    param([Parameter(Mandatory = $true)][string]$FilePath, [string[]]$ArgumentList = @())
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $FilePath @ArgumentList
        $script:NativeExitCode = $LASTEXITCODE
    } catch {
        Write-Warning "Could not run ${FilePath}: $($_.Exception.Message)"
        $script:NativeExitCode = -1
    } finally {
        $ErrorActionPreference = $saved
    }
}

function Get-PythonVersion {
    # "3.12" for the Python that $Command runs, or $null when it does not run (or is the Microsoft Store stub).
    param([Parameter(Mandatory = $true)][string]$Command, [string[]]$Prefix = @())
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) { return $null }
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & $Command @Prefix -c "import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return ([string]($out | Select-Object -First 1)).Trim() }
    } catch {
        # not runnable: try the next candidate
    } finally {
        $ErrorActionPreference = $saved
    }
    return $null
}

function Test-PathEntry {
    # True when the ;-separated $PathValue already lists $Folder (any case, with or without a trailing \).
    param([string]$PathValue, [string]$Folder)
    if (-not $PathValue) { return $false }
    $wanted = $Folder.TrimEnd([char]'\')
    foreach ($entry in ($PathValue -split ";")) {
        if ($entry -and $entry.TrimEnd([char]'\') -ieq $wanted) { return $true }
    }
    return $false
}

function Add-ToUserPath {
    # Adds $Folder to the user's PATH (for new terminals; no admin rights needed) and to this session's PATH.
    param([Parameter(Mandatory = $true)][string]$Folder)
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (Test-PathEntry $userPath $Folder) {
        Write-Host "$Folder is already on your PATH."
    } else {
        $parts = @()
        if ($userPath) { $parts = @($userPath -split ";" | Where-Object { $_ }) }
        [Environment]::SetEnvironmentVariable("Path", (($parts + $Folder) -join ";"), "User")
        Write-Host "Added $Folder to your PATH (new terminals will find it)."
    }
    if (-not (Test-PathEntry $env:Path $Folder)) { $env:Path = "$env:Path;$Folder" }
}

function Update-SessionPath {
    # winget installers change PATH in the registry only; pick up their folders without opening a new terminal.
    foreach ($scope in "Machine", "User") {
        $value = [Environment]::GetEnvironmentVariable("Path", $scope)
        if (-not $value) { continue }
        foreach ($entry in ($value -split ";")) {
            if ($entry -and -not (Test-PathEntry $env:Path $entry)) { $env:Path = "$env:Path;$entry" }
        }
    }
}

function Find-TesseractFolder {
    # Where the UB-Mannheim installer puts tesseract.exe (for everyone, or just this user); it never adds it to PATH.
    $folders = @()
    foreach ($root in @($env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:ProgramW6432)) {
        if ($root) { $folders += (Join-Path $root "Tesseract-OCR") }
    }
    if ($env:LOCALAPPDATA) { $folders += (Join-Path $env:LOCALAPPDATA "Programs\Tesseract-OCR") }
    foreach ($folder in $folders) {
        if (Test-Path (Join-Path $folder "tesseract.exe")) { return $folder }
    }
    return $null
}

function Install-WithWinget {
    param([string]$Id, [string]$Name, [string]$Url)
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Warning "winget is not available, so $Name was not installed. Install it from $Url (or get winget: 'App Installer' in the Microsoft Store)."
        return
    }
    Invoke-Native winget @("install", "--id", $Id, "-e", "--accept-package-agreements", "--accept-source-agreements")
    if ($script:NativeExitCode -ne 0) {
        Write-Warning "winget could not install $Name (exit code $($script:NativeExitCode)). Install it from $Url."
    }
    Update-SessionPath
}

# 1. Python 3.10-3.12 (openai-whisper/torch do not support newer versions yet): the py launcher first, then
#    python/python3 on PATH (the Microsoft Store Python has no py launcher).
$supported = @("3.12", "3.11", "3.10")
$pyCommand = $null
$pyPrefix = @()
foreach ($v in $supported) {
    if ((Get-PythonVersion -Command "py" -Prefix @("-$v")) -eq $v) {
        $pyCommand = "py"
        $pyPrefix = @("-$v")
        break
    }
}
if (-not $pyCommand) {
    foreach ($candidate in "python", "python3") {
        $found = Get-PythonVersion -Command $candidate
        if ($found -and ($supported -contains $found)) {
            $pyCommand = $candidate
            break
        }
    }
}
if (-not $pyCommand) {
    throw "Python 3.10, 3.11 or 3.12 was not found. Install it with:  winget install -e --id Python.Python.3.12  then open a new PowerShell window and run this script again."
}
Write-Host "Using Python $(Get-PythonVersion -Command $pyCommand -Prefix $pyPrefix) ($pyCommand $($pyPrefix -join ' '))"

# 2. Virtual environment.
if (-not (Test-Path .venv)) {
    Invoke-Native $pyCommand ($pyPrefix + @("-m", "venv", ".venv"))
    if ($script:NativeExitCode -ne 0) { throw "Could not create the virtual environment (.venv); see the messages above." }
}
$python = ".\.venv\Scripts\python.exe"

# 3. Python dependencies. PyAudio ships a prebuilt wheel on Windows, so no compiler is needed.
Invoke-Native $python @("-m", "pip", "install", "--upgrade", "pip")
if ($script:NativeExitCode -ne 0) { Write-Warning "Could not upgrade pip; carrying on with the one in .venv." }
Invoke-Native $python @("-m", "pip", "install", "-r", "requirements.txt")
if ($script:NativeExitCode -ne 0) { throw "pip could not install requirements.txt; see the messages above." }

# 4. Room settings: start from the classroom kit (C930e webcam at 1080p, 1280x800 Epson projector).
if (-not (Test-Path sar.env) -and (Test-Path sar.env.example)) {
    Copy-Item sar.env.example sar.env
    Write-Host "Created sar.env from sar.env.example (the classroom kit); edit it if your room differs."
}

# 5. Tesseract OCR (reads the text in uploaded images). The assistant finds it in its install folder even off
#    PATH, but hardware_check and a terminal want it on PATH too.
if (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
    $tesseractDir = Find-TesseractFolder
    if (-not $tesseractDir) {
        Install-WithWinget "UB-Mannheim.TesseractOCR" "Tesseract OCR" "https://github.com/UB-Mannheim/tesseract/wiki"
        $tesseractDir = Find-TesseractFolder
    }
    if ($tesseractDir) {
        Add-ToUserPath $tesseractDir
    } else {
        Write-Warning "Tesseract OCR is not installed, so image uploads will not work. Install it from https://github.com/UB-Mannheim/tesseract/wiki, then add its folder (usually C:\Program Files\Tesseract-OCR) to PATH."
    }
}

# 6. ffmpeg (required by openai-whisper to decode audio).
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Install-WithWinget "Gyan.FFmpeg" "FFmpeg" "https://www.gyan.dev/ffmpeg/builds/"
}

# 7. Ollama runtime for the local vision-language model.
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Install-WithWinget "Ollama.Ollama" "Ollama" "https://ollama.com/download"
}

# 8. Pull the model (~3 GB). Optional here; can be run later.
if ($PullModel) {
    $ollama = $null
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) {
        $ollama = $command.Source
    } elseif ($env:LOCALAPPDATA -and (Test-Path (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"))) {
        $ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    }
    if ($ollama) {
        Invoke-Native $ollama @("pull", "qwen2.5vl:3b")
    }
    if (-not $ollama -or $script:NativeExitCode -ne 0) {
        Write-Warning "The model was not downloaded. Start Ollama, then run: ollama pull qwen2.5vl:3b"
    }
} else {
    Write-Host "Skipping model pull. Run: ollama pull qwen2.5vl:3b"
}

Write-Host @"

Setup complete. Next:
  .\.venv\Scripts\python.exe scripts\hardware_check.py
  `$env:SAR_VLM_BACKEND='mock'; .\.venv\Scripts\python.exe -m biobench_sar       # no model needed
  .\.venv\Scripts\python.exe -m biobench_sar                                      # real VLM via Ollama
Open http://127.0.0.1:5050/ fullscreen (F) on the projector, http://127.0.0.1:5050/dashboard on your laptop.
Room settings are in sar.env; docs\HARDWARE_DAY.md walks through the first run with the kit.
"@
