@echo off
setlocal
title SystemX Server
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "serverExitCode=%errorlevel%"
if not "%serverExitCode%"=="0" (
    echo.
    echo SystemX stopped with an error. Check the messages above.
    pause
)
exit /b %serverExitCode%
