@echo off
setlocal
title SystemX Desktop Minesweeper Setup
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-desktop.ps1" %*
set "setupExitCode=%errorlevel%"
pause
exit /b %setupExitCode%
