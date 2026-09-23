@echo off
setlocal
title SystemOne Server Setup
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set "setupExitCode=%errorlevel%"
echo.
if not "%setupExitCode%"=="0" echo Setup failed. Check the messages above.
pause
exit /b %setupExitCode%
