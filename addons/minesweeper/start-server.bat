@echo off
setlocal
rem Minesweeper is sequential: one accelerated int8 model worker is sufficient.
set "SYSTEMX_WORKERS=1"
set "SYSTEMX_QUANTIZATION=int8"
call "%~dp0..\..\start.bat" %*
exit /b %errorlevel%
