@echo off
setlocal
title SystemOne Von Server
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "serverExitCode=%errorlevel%"
if not "%serverExitCode%"=="0" (
    echo.
    echo SystemOne stopped with an error. Check the messages above.
    pause
)
exit /b %serverExitCode%
