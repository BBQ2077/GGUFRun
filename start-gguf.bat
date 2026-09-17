@echo off
REM llama-server launcher (non-UI). Portable: every path is relative to this .bat's folder.
REM Runtime: always uses .\runtime\llama-server.exe (see README for where to get it).
REM
REM Usage:
REM   start-gguf.bat                            list models, start the first usable one
REM   start-gguf.bat <model-file|full-path>     start the given model
REM   start-gguf.bat <model> jinja              also add --jinja (recommended for Gemma)
REM   start-gguf.bat <model> dryrun             print the command only, do not launch
REM
REM Speculative decoding knobs (optional; omit to use llama.cpp defaults):
REM   set SPEC_N_MAX=3    before calling, or pass as the 3rd arg: n-max
REM   e.g.  start-gguf.bat model.gguf jinja 3
REM   NOTE: llama.cpp default is 3 (NOT 4). On small-VRAM cards 2-3 is often
REM         faster than 4 - extra draft length costs more than it gains.
REM
REM Drafter auto-pairing is family-matched (same family only: a cross-family drafter
REM makes llama-server crash with a hard assertion, not a friendly error).
REM Tip: start-ui.bat is nicer (model dropdown, remembers settings, alias follows model).
setlocal

set "BASE=%~dp0"
set "RT=%BASE%runtime"
if not exist "%RT%\llama-server.exe" (
  echo [ERROR] runtime\llama-server.exe not found under "%BASE%"
  echo         Put an official llama.cpp build ^(llama-server.exe + its DLLs^) into .\runtime\
  pause
  exit /b 1
)
set "PORT=18435"
set "EXTRA="
set "DRY="
set "MODEL=%~1"

if /i "%~1"=="jinja"  set "MODEL="
if /i "%~1"=="jinja"  set "EXTRA=--jinja"
if /i "%~1"=="dryrun" set "MODEL="
if /i "%~1"=="dryrun" set "DRY=1"
if /i "%~2"=="jinja"  set "EXTRA=--jinja"
if /i "%~3"=="jinja"  set "EXTRA=--jinja"
if /i "%~2"=="dryrun" set "DRY=1"
if /i "%~3"=="dryrun" set "DRY=1"
REM Optional 3rd arg = speculative n-max (overrides SPEC_N_MAX env var)
if not "%~3"=="" (
  if /i not "%~3"=="jinja" if /i not "%~3"=="dryrun" set "SPEC_N_MAX=%~3"
)

if "%MODEL%"=="" (
  echo [INFO] No model given - models in this folder:
  for %%F in ("%BASE%*.gguf") do (
    echo %%~nxF | findstr /i "dspark dflash draft mtp" >nul
    if errorlevel 1 (
      echo          %%~nxF
      if not defined MODEL set "MODEL=%%~fF"
    )
  )
  echo.
)

REM bare filename -> make it absolute against this folder (CWD may differ)
if not "%~1"=="" if exist "%BASE%%~1" set "MODEL=%BASE%%~1"

if "%MODEL%"=="" (
  echo [ERROR] no usable .gguf model found in %BASE%
  pause
  exit /b 1
)
if not exist "%MODEL%" (
  echo [ERROR] model not found: %MODEL%
  echo         usage: start-gguf.bat ^<model-file^> [jinja^|dryrun]
  pause
  exit /b 1
)
if not exist "%RT%\llama-server.exe" (
  echo [ERROR] llama-server.exe not found in %RT%
  pause
  exit /b 1
)

REM alias follows the model file name (the GUI reads general.name from the GGUF header)
for %%F in ("%MODEL%") do set "ALIAS=%%~nF"

REM ---- drafter auto-pairing: same family only ----
set "DRAFTER="
set "SPEC="
for %%K in (gemma bonsai qwen spark minicpm phi mistral glm llama) do (
  echo "%MODEL%" | findstr /i "%%K" >nul
  if not errorlevel 1 (
    for %%F in ("%BASE%*.gguf") do (
      if not defined DRAFTER (
        echo %%~nxF | findstr /i "%%K" >nul
        if not errorlevel 1 (
          echo %%~nxF | findstr /i "mtp dflash dspark draft eagle" >nul
          if not errorlevel 1 set "DRAFTER=%%~fF"
        )
      )
    )
  )
)
if not defined DRAFTER goto :nospec
set "SPEC=draft-simple"
echo "%DRAFTER%" | findstr /i "mtp" >nul && set "SPEC=draft-mtp"
echo "%DRAFTER%" | findstr /i "dflash" >nul && set "SPEC=draft-dflash"
echo "%DRAFTER%" | findstr /i "dspark" >nul && set "SPEC=draft-dspark"
echo "%DRAFTER%" | findstr /i "eagle" >nul && set "SPEC=draft-eagle3"
REM NOTE: keep these OUTSIDE any ( ) block. Batch expands %VAR% when a block is
REM parsed, so setting and using a var in the same block yields a stale/empty value.
set "SPECFLAGS=-md "%DRAFTER%" --spec-type %SPEC%"
REM Speculative knobs are OPTIONAL: only send what the user set, else use llama.cpp
REM defaults. Never hardcode n-max=4 - the llama.cpp default is 3, and on small-VRAM
REM cards 2-3 is often faster (extra draft length costs more than it gains).
if defined SPEC_N_MAX   set "SPECFLAGS=%SPECFLAGS% --spec-draft-n-max %SPEC_N_MAX%"
if defined SPEC_N_MIN   set "SPECFLAGS=%SPECFLAGS% --spec-draft-n-min %SPEC_N_MIN%"
if defined SPEC_P_SPLIT set "SPECFLAGS=%SPECFLAGS% --spec-draft-p-split %SPEC_P_SPLIT%"
if defined SPEC_P_MIN   set "SPECFLAGS=%SPECFLAGS% --spec-draft-p-min %SPEC_P_MIN%"
goto :specdone

:nospec
echo [INFO] no same-family drafter found - running Normal mode
set "SPECFLAGS="

:specdone

echo Starting llama-server on 127.0.0.1:%PORT% ...
echo Runtime : %RT%
echo Model   : %MODEL%
echo Alias   : %ALIAS%  ^| Spec : %SPEC%  ^| Jinja : %EXTRA%
echo Command :
echo llama-server.exe -m "%MODEL%" --alias %ALIAS% %EXTRA% -ngl all -ngld all -fa on -ctk q4_0 -ctv q4_0 -nkvo -c 65536 -np 1 --host 127.0.0.1 --port %PORT% --temp 0.8 --top-p 0.95 --top-k 40 --min-p 0.05 %SPECFLAGS%

if defined DRY (
  echo [dryrun] the command above was NOT executed.
  exit /b 0
)

cd /d "%RT%"
llama-server.exe -m "%MODEL%" --alias %ALIAS% %EXTRA% -ngl all -ngld all -fa on -ctk q4_0 -ctv q4_0 -nkvo -c 65536 -np 1 --host 127.0.0.1 --port %PORT% --temp 0.8 --top-p 0.95 --top-k 40 --min-p 0.05 %SPECFLAGS%

endlocal
