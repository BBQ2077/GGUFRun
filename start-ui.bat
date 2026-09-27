@echo off
REM Uses Python on PATH; install Python for Windows with Tcl/Tk support.
setlocal
where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%~dp0gguf-ui.py"
) else (
  python "%~dp0gguf-ui.py"
)
endlocal
