@echo off
REM GGUFRun keyboard-probe launcher (portable: resolves everything from its own folder).
REM Picks the first Python that can "import tkinter": "python" on PATH first,
REM then the usual per-user / per-machine install locations. No machine-specific paths.
setlocal
set "PY="
python -c "import tkinter" >nul 2>nul && set "PY=python"
if not defined PY for %%P in (
    "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
    "%LOCALAPPDATA%\Python\bin\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "C:\Python314\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
) do if not defined PY if exist "%%~P" "%%~P" -c "import tkinter" >nul 2>nul && set "PY=%%~P"
if not defined PY set "PY=python"
"%PY%" "%~dp0key-probe.py"
endlocal
