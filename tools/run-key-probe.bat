@echo off
REM Keyboard diagnostic probe launcher (portable: resolves paths from its own folder).
setlocal
set "PY=python"
if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
"%PY%" "%~dp0key-probe.py"
endlocal
