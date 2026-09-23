@echo off
setlocal
title SystemX RA2 Agent (launcher)
set "ROOT=%~dp0..\.."
if not exist "%ROOT%\.venv\Scripts\python.exe" (
    echo Run setup.bat in the SystemX folder first.
    pause
    exit /b 1
)
rem The controller starts the System One server in its own window when needed (Start button).
cd /d "%ROOT%"
"%ROOT%\.venv\Scripts\python.exe" -m addons.chronodivide %*
if errorlevel 1 pause
