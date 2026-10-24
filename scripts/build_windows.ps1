# Build the Windows desktop app and its installer locally: the same steps as .github/workflows/build.yml.
# Run from anywhere in PowerShell, after scripts\setup_windows.ps1 has created .venv:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\build_windows.ps1 [-Whisper] [-SkipSelfTest] [-SkipInstaller]
# Output: dist\SAR Whiteboard Assistant\ (the app folder) and dist\SAR-Whiteboard-Assistant-Setup-<version>.exe
param([switch]$Whisper, [switch]$SkipSelfTest, [switch]$SkipInstaller)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Invoke-Step([string]$Name, [scriptblock]$Command) {
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE) { throw "$Name failed (exit code $LASTEXITCODE)" }
}

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "No .venv here. Run .\scripts\setup_windows.ps1 first." }

Invoke-Step "Install PyInstaller" { & $python -m pip install --upgrade "pyinstaller>=6.10,<7" }
$version = (& $python -c "import biobench_sar; print(biobench_sar.__version__)").Trim()
Write-Host "Version $version"

$fetchArgs = @("packaging\fetch_models.py")
if ($Whisper) { $fetchArgs += "--whisper" }
Invoke-Step "Fetch models" { & $python @fetchArgs }
Invoke-Step "PyInstaller" { & $python -m PyInstaller packaging\sar.spec --noconfirm }

$exe = "dist\SAR Whiteboard Assistant\SAR Whiteboard Assistant.exe"
if (-not $SkipSelfTest) { Invoke-Step "Self-test" { & $exe --self-test } }

if (-not $SkipInstaller) {
    $iscc = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $iscc) {
        $cmd = Get-Command iscc.exe -ErrorAction SilentlyContinue
        if ($cmd) { $iscc = $cmd.Source }
    }
    if (-not $iscc) { throw "Inno Setup 6 not found. Install it: winget install JRSoftware.InnoSetup" }
    Invoke-Step "Installer" { & $iscc "/DAppVersion=$version" packaging\windows\installer.iss }
    Write-Host "Installer: dist\SAR-Whiteboard-Assistant-Setup-$version.exe"
}
Write-Host "App folder: dist\SAR Whiteboard Assistant\"
