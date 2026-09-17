@echo off
REM GGUFRun GUI launcher (portable: resolves everything from its own folder).
REM Uses the system Python on PATH (or the default Python 3.14 install location).
setlocal
set "PY=python"
if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
start "" "%PY%" "%~dp0gguf-ui.py"
endlocal
