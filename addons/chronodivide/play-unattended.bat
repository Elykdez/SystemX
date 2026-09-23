@echo off
rem Unattended play: the controller starts in Execute + System One, arms itself for the match,
rem brings the game window to the front, and resumes by itself a few seconds after you
rem stop touching the mouse or keyboard. Start the skirmish in the browser yourself.
call "%~dp0play-chronodivide.bat" ui --autostart --unattended --front
