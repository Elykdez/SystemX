@echo off
setlocal
title SystemOne Desktop Minesweeper
pushd "%~dp0..\.." || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Run the root setup.bat first.
    popd
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import cv2, mss, PIL" >nul 2>&1
if errorlevel 1 (
    echo Run addons\minesweeper\setup-desktop.bat first.
    popd
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m addons.minesweeper.desktop_player %*
set "playerExitCode=%errorlevel%"
popd
if not "%playerExitCode%"=="0" pause
exit /b %playerExitCode%
