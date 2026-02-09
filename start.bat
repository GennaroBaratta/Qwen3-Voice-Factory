@echo off
title Qwen3 Voice Factory
echo Starting App...

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python environment not found!
    echo Please run 'install.bat' first.
    pause
    exit
)

.\.venv\Scripts\python.exe app.py
pause
