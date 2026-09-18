# Changelog

All notable changes to GGUFRun are recorded here.
Format: `MAJOR.MINOR` with date (YYYY-MM-DD).

## 2.0 — 2026-09-18

### Added

- **Parameter templates** (`presets.json`): save a named snapshot of the launch settings, bind it to
  a model, and it is re-applied automatically every time that model is selected — including the
  runtime folder that model last used. Bind / unbind / set-as-default buttons in the template row.
- **Offload controls**: `-ncmoe` / `--n-cpu-moe` (keep the first N layers' *expert* weights on the
  CPU) and `-ncffn` / `--n-cpu-ffn` (same for *dense FFN* weights). Empty = flag not sent.
- **Load mode / lazy mode** (`--load-mode` / `-lm`, `--lazy-mode` / `-lzm`) in a dedicated
  Advanced tab, with an explanation of the `auto` default (> 4 GiB tensors only).
- **LoRA tab**: `--lora <file.gguf>` plus a strength scale that is applied *after* startup through
  `POST /lora-adapters`, so you can retune strength without reloading the model.
- **Custom arguments tab**: append arbitrary flags to the end of the launch command, quote-aware.
- **Reasoning controls**: thinking mode (`-rea`) and reasoning budget (`--reasoning-budget`,
  `0` = stop thinking immediately, `-1` = unlimited) plus `--reasoning-effort`.
- **Lightweight web view** (`🌐 開網頁（輕量）`): opens Chrome/Edge with `--app` and a dedicated
  profile at `tools/_webview_profile/` (no tabs, no extensions, no shared cookies). Falls back to
  the default browser when neither is installed.
- **Built-in key debugger** (`🔍 按鍵偵錯`) — the same diagnostics as `tools/key-probe.py`, without
  leaving the app.
- **Runtime × architecture hints**: the model status line warns when the selected runtime is known
  not to support the selected model's architecture.
- **New tools**: `tools/_arch.py` (read GGUF header metadata without loading the model),
  `tools/_ttype.py` (tensor type histogram), `tools/_bench_ncmoe.py` (offload benchmark).
- **Duplicate-server guard** (take over / restart / cancel) and an `atexit` no-orphan fallback that
  only ever targets the exact PID this app spawned.

### Changed

- Advanced sampling window is now organised into 8 tabs (基本 / 重複懲罰 / DRY / 其他 / 投機解碼 /
  LoRA / 自訂指令 / 載入模式); every optional field is **empty by default** and is only sent when filled.
- Alias (`--alias`) follows the main model's filename (falls back to the GGUF's `general.name`
  when the filename has no safe characters).
- Model picker remembers per-model runtime and template, so switching models no longer inherits the
  previous model's runtime by accident.
- README rewritten as a bilingual document covering templates, offload, LoRA, the lightweight
  window, and a privacy/security section.

### Packaging / hygiene

- No user-specific paths in the tracked files: launchers resolve `python` from `PATH` (falling back
  to the per-user Python 3.14 install), and the benchmark tool takes its model and runtime from
  environment variables or the project folder. The only absolute paths left are generic Windows
  locations (e.g. the standard Chrome/Edge install dirs used to find a Chromium build).
- `.gitignore` extended so machine-specific state and heavy artefacts can never be committed by
  accident: `ui-settings.json`, `models.json`, `presets.json`, `tools/_webview_profile/`,
  `*.gguf`, `*-runtime/`, `*.log`, `__pycache__/`, `*.bak*`, `*.crdownload`, editor/OS junk.

## 1.x

Initial public release: portable Tkinter console for `llama-server` — model/DF picker, runtime
switcher, start/stop with live log, dry-run command preview, speculative-decoding controls,
sampling controls, remembered settings, model classification memory, Recycle-Bin-only deletion.
