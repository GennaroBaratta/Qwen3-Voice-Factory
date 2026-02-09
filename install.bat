@echo off
setlocal
title Qwen3 Voice Factory Installer

echo ===================================================
echo   Qwen3 Voice Factory - Automatic Installer
echo   (RTX 5090 / CUDA 12.8 Ready)
echo ===================================================
echo.

:: 1. Ensure uv is installed
set "UV_EXE="
where uv >nul 2>nul
if errorlevel 1 (
    echo [1/4] Installing uv (Python manager)
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
)

for %%X in (uv.exe) do set "UV_EXE=%%~$PATH:X"
if not defined UV_EXE (
    for %%X in ("%USERPROFILE%\.local\bin\uv.exe" "%USERPROFILE%\.cargo\bin\uv.exe") do (
        if exist "%%~fX" set "UV_EXE=%%~fX"
    )
)

if not defined UV_EXE (
    echo ERROR: uv not found. Please restart your terminal or install uv manually.
    pause
    exit /b 1
)

:: 2. Create virtual environment if needed
if not exist ".venv\Scripts\python.exe" (
    echo [2/4] Creating virtual environment (Python 3.11)
    "%UV_EXE%" venv .venv --python 3.11
)

:INSTALL_PACKAGES
echo.
echo ===================================================
echo   Installing Libraries...
echo ===================================================

:: A. PyTorch Nightly (Must be first for RTX 5090)
echo [3/4] Installing PyTorch Nightly (Blackwell Support)...
"%UV_EXE%" pip install --python .venv\Scripts\python.exe --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

:: B. Project Dependencies
echo [4/4] Installing project dependencies...
"%UV_EXE%" sync --python .venv\Scripts\python.exe --no-dev

echo.
echo ===================================================
echo   INSTALLATION COMPLETE!
echo   You can now run 'start.bat'.
echo ===================================================
pause
