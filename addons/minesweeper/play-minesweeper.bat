@echo off
setlocal
title SystemX Minesweeper Player
pushd "%~dp0..\.." || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat and addons\minesweeper\setup-minesweeper.bat first.
    popd
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m addons.minesweeper %*
set "playerExitCode=%errorlevel%"
popd
if not "%playerExitCode%"=="0" pause
exit /b %playerExitCode%
