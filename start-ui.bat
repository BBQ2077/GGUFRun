@echo off
REM Portable launcher; uses the installed Python 3 interpreter.
setlocal
where py >nul 2>&1
if not errorlevel 1 (
  start "" py -3 "%~dp0gguf-ui.py"
) else (
  start "" python "%~dp0gguf-ui.py"
)
endlocal
