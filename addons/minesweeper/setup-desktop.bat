@echo off
setlocal
title SystemOne Desktop Minesweeper Setup
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-desktop.ps1" %*
set "setupExitCode=%errorlevel%"
pause
exit /b %setupExitCode%
