@echo off
setlocal
title SystemOne Minesweeper Setup
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-minesweeper.ps1" %*
set "setupExitCode=%errorlevel%"
echo.
if not "%setupExitCode%"=="0" echo Setup failed. Check the messages above.
pause
exit /b %setupExitCode%
