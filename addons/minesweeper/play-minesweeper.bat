@echo off
setlocal
title SystemOne Minesweeper Player
pushd "%~dp0..\.." || exit /b 1
".venv\Scripts\python.exe" -m addons.minesweeper %*
set "playerExitCode=%errorlevel%"
popd
if not "%playerExitCode%"=="0" pause
exit /b %playerExitCode%
