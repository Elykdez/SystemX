@echo off
setlocal
title SystemX RA2 Agent (autoplay)
rem Plays one skirmish after another: the controller starts the match from the setup screen,
rem plays it, exits at the score screen, and starts again here with the latest code.
rem Stop: close this window, or create .cache\chronodivide\autoplay-stop (checked between games).
set "ROOT=%~dp0..\.."
set "STOP=%ROOT%\.cache\chronodivide\autoplay-stop"
if not exist "%ROOT%\.venv\Scripts\python.exe" (
    echo Run setup.bat in the SystemX folder first.
    pause
    exit /b 1
)
if exist "%STOP%" del "%STOP%"
cd /d "%ROOT%"
echo Autoplay: leave the game on the skirmish setup screen, and hands off the mouse and keyboard.
:loop
if exist "%STOP%" goto done
"%ROOT%\.venv\Scripts\python.exe" -m addons.chronodivide ui --autoplay
if errorlevel 1 (
    echo The agent stopped with code %errorlevel%; autoplay ends here.
    goto done
)
timeout /t 3 /nobreak >nul
goto loop
:done
echo Autoplay stopped.
pause
