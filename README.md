# GGUFRun — Local GGUF Inference Console (Portable)

> 本機 GGUF 推理控制台（可攜）｜A portable GUI console for `llama-server`.
>
> 選模型、選草稿模型（DF）、切換執行檔、啟停伺服器 —— 一個視窗搞定。
> Pick a model, pick a draft model (DF), switch runtime, start/stop the server — all in one window.

[English](#english) ｜ [繁體中文](#繁體中文)

---

## English

### What it is

GGUFRun is a small **Tkinter GUI front-end for `llama.cpp`'s `llama-server`**. It keeps a folder
of `.gguf` models tidy and lets you launch local inference without memorizing CLI flags.

It is **portable**: everything resolves relative to the script's own folder. Copy the folder
anywhere (another disk, a USB stick, another machine) and it still works — no absolute paths baked in.

### Features

- **Model picker** — auto-scans `*.gguf` in the app folder, plus manually added models remembered in `models.json`.
- **Draft model (DF) picker** — for speculative decoding. Auto-pairs with the main model, or pick one manually.
- **Runtime switcher** — scans every sub-folder containing `llama-server.exe`.
- **Start / stop the server** with one click; live log pane; port is configurable.
- **Preview the exact command** before launching (dry-run view).
- **Sampling controls** — Temperature / Top-P / Top-K / Min-P + 17 optional knobs,
  defaults matched to `llama.cpp` built-ins (`0.8 / 0.95 / 40 / 0.05`).
- **Auto-remembered settings** — model, DF, runtime, port, mode, sampling values persist in `ui-settings.json`.
- **Model classification memory** — whether a file is a *main model* or a *draft model* is remembered
  the first time you say so; filenames are only a hint, never a verdict.
- **Safe delete** — removing a model entry can send the file to the **Windows Recycle Bin**
  (recoverable). This app never performs an irreversible delete.

### Requirements

| Item | Notes |
|---|---|
| OS | Windows 10 / 11 (x64). Uses the Win32 API for process control and the Recycle Bin. |
| Python | 3.10+ with **Tkinter**. On Windows, the official python.org installer includes Tkinter by default. |
| llama.cpp | A `llama-server.exe` build for Windows (see below). |
| GPU | Optional. Runs on CPU, but a CUDA build + NVIDIA GPU is much faster. |

### Getting `llama-server` (beginners: read this)

GGUFRun is only the **control panel**. It does not include the engine that actually runs models.
This repo also does **not** ship that engine, because it is very large (the CUDA DLLs alone exceed
GitHub's 100 MB per-file limit). You download it yourself — it's a 3-step copy-paste job:

1. **Open the releases page:**
   👉 https://github.com/ggml-org/llama.cpp/releases

   (Always use the newest release at the top of that page. If you want a direct link to the newest
   one, use https://github.com/ggml-org/llama.cpp/releases/latest )

2. **Download two ZIP files** from the same release. In the *Assets* list, look for:

   | Your situation | Which file to download |
   |---|---|
   | NVIDIA GPU (recommended) | `llama-bXXXXX-bin-win-cuda-12.4-x64.zip` **and** `cudart-llama-bin-win-cuda-12.4-x64.zip` |
   | No GPU / not sure | `llama-bXXXXX-bin-win-cpu-x64.zip` |

   (`XXXXX` is the build number — it changes every release, that's normal.)

   > **Windows x64 only.** Also make sure you take the `win` ones, not the `linux`/`macos` ones.
   > With a newer GPU and up-to-date drivers you can also take the `cuda-13.4-x64` build instead
   > (performance is about the same).

3. **Extract both ZIPs into one folder** named `runtime`, next to `gguf-ui.py`.
   Both archives contain DLLs, so extract them into the **same** folder and let them merge.

After that, your folder should look like this:

```
GGUFRun/
├─ gguf-ui.py
├─ start-ui.bat
├─ start-gguf.bat
├─ runtime/                    <-- you just created this
│  ├─ llama-server.exe         <-- the important one
│  ├─ ggml-cuda.dll
│  ├─ cublas64_12.dll
│  └─ ... (all the other .exe and .dll files)
└─ YourModel-Q4_K_M.gguf       <-- your downloaded model
```

**How to check it worked:** open GGUFRun, and the **Runtime** dropdown should list `runtime/`.
If it is empty, `llama-server.exe` is not where the app expects it.

> The app also accepts **any** sub-folder that contains `llama-server.exe`, and the runtime
> dropdown lists them all — so you can keep several builds side by side (e.g. `runtime-cuda/`
> and `runtime-cpu/`) and switch between them.

### Where do I get a `.gguf` model?

Any model you can run, from wherever you prefer to get GGUF files (e.g. Hugging Face).
Put the `.gguf` file **directly next to `gguf-ui.py`** so the app can auto-detect it.
Pick whatever quantization suits your VRAM — there is no required naming convention.

> Grabbing a large file from a host that throttles single connections? `pardl.py` in this repo
> downloads it in parallel ranges: `python pardl.py <url> <out-file> [connections]`.

### Quick start

1. Put `llama-server.exe` (+ DLLs) into `runtime/` (see above).
2. Put one or more `.gguf` models into the same folder as `gguf-ui.py`.
3. Double-click **`start-ui.bat`**, or run:
   ```
   python gguf-ui.py
   ```
4. Pick a model, optionally pick a draft model, press **▶ Start**.
5. Open `http://127.0.0.1:18435/` (the port is shown / editable in the UI).

Command-line alternative (no window):

```
start-gguf.bat YourModel-Q4_K_M.gguf
start-gguf.bat YourModel-Q4_K_M.gguf jinja        # enable --jinja
start-gguf.bat YourModel-Q4_K_M.gguf jinja dryrun # print the command, do not launch
```

### Speculative decoding (draft models / DF)

When the mode is **Speculative (DF)**, the app passes your draft model to `llama-server`
along with a `--spec-type` inferred from its filename:

| Filename contains | `--spec-type` |
|---|---|
| `mtp` | `draft-mtp` |
| `dflash` | `draft-dflash` |
| `dspark` | `draft-dspark` |
| `eagle` | `draft-eagle3` |
| *(none of the above)* | `draft-simple` |

Notes:

- The draft model should belong to the **same family** as the main model. A mismatched pair can
  crash `llama-server` outright; the UI warns you when it can detect a mismatch.
- **A filename containing `mtp` does not automatically mean "draft model."** Many main models
  embed MTP internally and are still complete models. The app therefore treats `mtp` as a *hint*
  only: a large `mtp` file is offered as a main model, a small one as a draft candidate — and once
  you classify a file, that choice is remembered and filenames no longer override it.

#### Tuning draft length (`--spec-draft-n-max`)

`llama.cpp` drafts up to **3** tokens by default. Do **not** assume `4` is optimal — it is not
a rule, and it is not what `llama.cpp` ships with. Drafting too far ahead costs more than it
saves, and the effect grows the more constrained your hardware is.

The app's **Advanced (參數／進階) → "投機解碼"** tab exposes four knobs. **Leave them empty to
keep `llama.cpp`'s own defaults** (they are only sent when you fill them in):

| Setting | CLI flag | `llama.cpp` default |
|---|---|---|
| Draft length cap | `--spec-draft-n-max` | 3 |
| Draft length floor | `--spec-draft-n-min` | 0 |
| Draft split probability | `--spec-draft-p-split` | 0.10 |
| Min draft probability | `--spec-draft-p-min` | 0.00 |

Reference sweep (one 8 GB GPU, `n-max` 1–6): Normal 58.2 → n1 68.4 → n2 72.0 → **n3 72.2** →
n4 70.7 → n5 63.7 → n6 61.2 t/s. Peak at 3, i.e. exactly `llama.cpp`'s default. Sweep 1–6
yourself and keep the best — on small-VRAM cards the best value is usually **2–3**, not 4.

The CLI launcher takes it too:

```
start-gguf.bat YourModel.gguf jinja 3      # 3rd arg = --spec-draft-n-max
set SPEC_N_MAX=3 && start-gguf.bat YourModel.gguf
```

### How models are classified

1. **Memory first.** If a file was ever classified via **➕ Add model / ➕ Add DF**, the remembered
   `kind` wins, always.
2. **Otherwise, filename hints.** `draft` / `dspark` / `dflash` → draft model.
   `mtp` + small file (< 1 GiB) → draft model. Everything else → main model.

### Files

| File / folder | Purpose |
|---|---|
| `gguf-ui.py` | The GUI console |
| `start-ui.bat` | Launches the GUI using the system Python |
| `start-gguf.bat` | Headless CLI launcher: `start-gguf.bat <model.gguf> [jinja] [dryrun]` |
| `ui-settings.json` | Auto-saved UI state (created on first run; machine-specific, git-ignored) |
| `models.json` | Manually added models / DF and their remembered classification (git-ignored) |
| `runtime/` | Your `llama-server.exe` build (not in this repo) |
| `*.gguf` | Your models (not in this repo) |
| `pardl.py` | Parallel ranged downloader: `python pardl.py <url> <out-file> [connections]` (default 8). Splits a file into ranges, downloads them concurrently, resumes from finished `.parts/` chunks, then merges. Handy for big model files from hosts that cap a single connection's speed. |
| `tools/` | Helper scripts (see below) |

#### `tools/`

| File | What it does |
|---|---|
| `gguf_dspark_to_dflash.py` | Rewrites a legacy `arch=dspark` drafter GGUF into the current `arch=dflash` convention (renames metadata/tensors, shifts `target_layers`, injects the tokenizer from a donor GGUF). Copies tensor bytes as-is — nothing is requantized. Only needed if you have an older draft model the runtime no longer accepts. |
| `key-probe.py` | Keyboard diagnostic. Prints every key's `keysym` / `char` / `keycode` and writes `key-probe.log`. Used to figure out odd keys (e.g. numpad `.` with NumLock off). |
| `run-key-probe.bat` | Double-click launcher for `key-probe.py`. |

Generated at runtime and intentionally **git-ignored**: `ui-settings.json`, `models.json`,
`*.gguf`, `runtime/`, `*.log`, `__pycache__/`.

### `tools/`

| Script | Purpose |
|---|---|
| `gguf_dspark_to_dflash.py` | Utility to convert/patch draft-model metadata between `dspark` and `dflash` flavours. |
| `key-probe.py` + `run-key-probe.bat` | Small keyboard diagnostic: prints the raw keysym/keycode of each key press. Useful when a numpad key behaves oddly under different NumLock / IME states. |

### Troubleshooting

| Symptom | Fix |
|---|---|
| `start-ui.bat` opens nothing | Make sure `python` is on `PATH`, or that Python is installed to the default location. Install from python.org with the *tcl/tk* option enabled. |
| Runtime dropdown is empty | No sub-folder contains `llama-server.exe`. Create `runtime/` and put it there. |
| Server exits immediately | Check the log pane. Common causes: mismatched draft model family, unsupported model architecture, or insufficient VRAM — try lowering `-ngl`. |
| Numpad decimal point types nothing | Windows sends a different key code when NumLock is off. The app handles it; if it still misbehaves, run `tools/run-key-probe.bat` and check the log. |

### License

No license file is included yet. Add one of your choosing before publishing or redistributing.

---

## 繁體中文

### 這是什麼

GGUFRun 是 `llama.cpp` 的 `llama-server` 的一個小型 **Tkinter 圖形前端**。它把一整資料夾的
`.gguf` 模型整理好，讓你不用記一堆命令列參數就能啟動本機推理。

它是**可攜的**：所有路徑都以程式自己所在的資料夾為基準動態解析。整個資料夾複製到
任何地方（別的磁碟、隨身碟、別的電腦）都能直接跑，沒有寫死的絕對路徑。

### 功能

- **主模型選擇**：自動掃描程式資料夾內的 `*.gguf`，加上手動添加（記在 `models.json`）的模型。
- **草稿模型（DF）選擇**：用於投機解碼，可依主模型自動配對，也可手動指定。
- **執行檔切換**：自動掃描所有含 `llama-server.exe` 的子資料夾。
- **一鍵啟停伺服器**：即時日誌視窗；埠可自訂。
- **啟動前預覽完整指令**（dry-run）。
- **取樣參數控制**：Temperature / Top-P / Top-K / Min-P ＋ 17 項選配，
  預設值對齊 `llama.cpp` 內建預設（`0.8 / 0.95 / 40 / 0.05`）。
- **設定自動記憶**：主模型、DF、執行檔、埠、模式、取樣值都存進 `ui-settings.json`。
- **模型分類記憶**：一個檔案是「主模型」還是「草稿模型」，你分類過一次就會記住；
  檔名只是提示，永遠不會蓋過你的選擇。
- **安全刪除**：移除模型項目時可把檔案**丟進 Windows 資源回收桶**（可還原）。
  本程式不做任何無法挽回的永久刪除。

### 系統需求

| 項目 | 說明 |
|---|---|
| 作業系統 | Windows 10 / 11（x64）。行程控制與資源回收桶使用 Win32 API。 |
| Python | 3.10 以上，需含 **Tkinter**。Windows 官方安裝檔預設就含 Tkinter。 |
| llama.cpp | 一份 Windows 版 `llama-server.exe`（見下節）。 |
| 顯示卡 | 非必要。可純 CPU 執行，但 CUDA 版 ＋ NVIDIA 顯卡會快很多。 |

### 取得 `llama-server`（新手看這裡）

GGUFRun 只是**控制面板**，不含真正跑模型的引擎。本倉庫也不附這個引擎，因為它很大
（光是 CUDA 的 DLL 就超過 GitHub 單檔 100 MB 的上限）。請自己下載，三個步驟：

1. **打開官方 Releases 頁面：**
   👉 https://github.com/ggml-org/llama.cpp/releases

   （用最上面那個最新版就好。想直接跳到最新版可用
   https://github.com/ggml-org/llama.cpp/releases/latest ）

2. **在同一個版本下載兩個 ZIP。** 在 *Assets* 清單裡找：

   | 你的情況 | 要下載哪個檔 |
   |---|---|
   | 有 NVIDIA 顯卡（推薦） | `llama-bXXXXX-bin-win-cuda-12.4-x64.zip` **和** `cudart-llama-bin-win-cuda-12.4-x64.zip` |
   | 沒顯卡／不確定 | `llama-bXXXXX-bin-win-cpu-x64.zip` |

   （`XXXXX` 是版本號，每次改版都會變，正常現象。）

   > **只支援 Windows x64。** 也別抓錯成 `linux`／`macos` 的版本。
   > 顯卡較新、驅動夠新的話，也可以改抓 `cuda-13.4-x64` 的版本（效能差不多）。

3. **把兩個 ZIP 解壓縮到同一個資料夾**，命名為 `runtime`，放在 `gguf-ui.py` 旁邊。
   兩個壓縮檔裡都有 DLL，請解到**同一個**資料夾讓它們合併。

完成後你的資料夾應該長這樣：

```
GGUFRun/
├─ gguf-ui.py
├─ start-ui.bat
├─ start-gguf.bat
├─ runtime/                    <-- 你剛剛建立的
│  ├─ llama-server.exe         <-- 最重要的就是這顆
│  ├─ ggml-cuda.dll
│  ├─ cublas64_12.dll
│  └─ ...（其他 .exe 與 .dll）
└─ YourModel-Q4_K_M.gguf       <-- 你下載的模型
```

**怎麼確認成功：** 打開 GGUFRun，上面的「執行檔」下拉應該會出現 `runtime/`。
如果空的，就是 `llama-server.exe` 沒放對位置。

> 程式也接受**任何**含 `llama-server.exe` 的子資料夾，下拉會全部列出——
> 所以你可以同時放好幾套互相切換（例如 `runtime-cuda/` 與 `runtime-cpu/`）。

### `.gguf` 模型哪裡下載？

任何你跑得動的模型，從你習慣的地方取得 GGUF 檔即可（例如 Hugging Face）。
把 `.gguf` 檔**直接放在 `gguf-ui.py` 旁邊**，程式就會自動掃到。
量化等級挑適合你顯存的即可，**沒有規定的檔名格式**。

> 從限制單連線速度的來源抓大檔時，可用本倉庫的 `pardl.py` 多連線分段下載：
> `python pardl.py <網址> <輸出檔> [連線數]`。

### 快速開始

1. 把 `llama-server.exe`（含 DLL）放進 `runtime/`（見上）。
2. 把一或多個 `.gguf` 模型放進 `gguf-ui.py` 所在資料夾。
3. 雙擊 **`start-ui.bat`**，或執行：
   ```
   python gguf-ui.py
   ```
4. 選主模型、（可選）選草稿模型，按 **▶ 啟動**。
5. 開啟 `http://127.0.0.1:18435/`（埠在 UI 上可看可改）。

純命令列（不開視窗）：

```
start-gguf.bat YourModel-Q4_K_M.gguf
start-gguf.bat YourModel-Q4_K_M.gguf jinja        # 開啟 --jinja
start-gguf.bat YourModel-Q4_K_M.gguf jinja dryrun # 只印指令，不啟動
```

### 投機解碼（草稿模型／DF）

當模式選 **投機解碼（草稿模型 DF）** 時，程式會把草稿模型連同 `--spec-type` 一起傳給
`llama-server`，`--spec-type` 依檔名推斷：

| 檔名含 | `--spec-type` |
|---|---|
| `mtp` | `draft-mtp` |
| `dflash` | `draft-dflash` |
| `dspark` | `draft-dspark` |
| `eagle` | `draft-eagle3` |
| （都沒有） | `draft-simple` |

注意：

- 草稿模型必須與主模型**同家族**。配錯的話 `llama-server` 可能直接崩潰；UI 能判斷時會警告。
- **檔名含 `mtp` 不代表它就是草稿模型。** 很多主模型本身就內建 MTP，仍是完整模型。
  因此程式只把 `mtp` 當**提示**：大的 `mtp` 檔歸類為主模型，小的歸為草稿候選；
  而且只要你分類過一次，就會記住，之後檔名不再蓋過你的選擇。

### 模型如何分類

1. **記憶優先。** 只要曾用「➕ 添加模型 / ➕ 添加 DF」分類過，記住的 `kind` 永遠優先。
2. **其次才是檔名提示。** `draft` / `dspark` / `dflash` → 草稿模型；
   `mtp` 且檔案偏小（< 1 GiB）→ 草稿模型；其餘 → 主模型。

#### 草稿長度怎麼調（`--spec-draft-n-max`）

`llama.cpp` 預設一次草稿 **3** 個 token。**不要以為 4 最好**——那不是規則，也不是
`llama.cpp` 的出廠值。草稿抓太長，額外成本會吃掉收益，硬體越受限越明顯。

程式在 **「參數／進階」→「投機解碼」分頁**提供四個旋鈕。**留空＝沿用 `llama.cpp` 自己的
預設**（只有你填了才會送出）：

| 設定 | 參數 | `llama.cpp` 預設 |
|---|---|---|
| 草稿長度上限 | `--spec-draft-n-max` | 3 |
| 草稿長度下限 | `--spec-draft-n-min` | 0 |
| 草稿切分機率 | `--spec-draft-p-split` | 0.10 |
| 草稿最低機率 | `--spec-draft-p-min` | 0.00 |

實測掃描（某 8 GB 顯卡，`n-max` 1–6）：Normal 58.2 → n1 68.4 → n2 72.0 → **n3 72.2** →
n4 70.7 → n5 63.7 → n6 61.2 t/s。峰值落在 3，也就是 `llama.cpp` 的預設值。建議自己掃
1–6 取最佳——VRAM 小的卡上通常落在 **2～3**，而非 4。

命令列啟動器也吃這個參數：

```
start-gguf.bat 你的模型.gguf jinja 3      # 第 3 個參數 = --spec-draft-n-max
set SPEC_N_MAX=3 && start-gguf.bat 你的模型.gguf
```

### 檔案說明

| 檔案／資料夾 | 用途 |
|---|---|
| `gguf-ui.py` | 圖形控制台本體 |
| `start-ui.bat` | 用系統 Python 啟動控制台 |
| `start-gguf.bat` | 純命令列啟動：`start-gguf.bat <模型檔> [jinja] [dryrun]` |
| `ui-settings.json` | 自動儲存的 UI 狀態（首次執行時建立；含本機路徑，已被 git 忽略） |
| `models.json` | 手動添加的模型／DF 與其分類記憶（已被 git 忽略） |
| `runtime/` | 你的 `llama-server.exe`（不在本倉庫） |
| `*.gguf` | 你的模型（不在本倉庫） |
| `pardl.py` | 多連線分段下載器：`python pardl.py <網址> <輸出檔> [連線數]`（預設 8）。把檔案切成數段並行下載，可從已完成的 `.parts/` 分段續傳，最後合併。適合從限制單連線速度的來源抓大型模型檔。 |
| `tools/` | 輔助腳本（見下） |

執行時產生、且刻意**被 git 忽略**：`ui-settings.json`、`models.json`、`*.gguf`、
`runtime/`、`*.log`、`__pycache__/`。

### `tools/` 輔助工具

| 腳本 | 用途 |
|---|---|
| `gguf_dspark_to_dflash.py` | 把舊的 `arch=dspark` 草稿模型 GGUF 改寫成目前的 `arch=dflash` 格式（改 metadata／tensor 名稱、位移 `target_layers`、從捐贈模型注入 tokenizer）。Tensor 位元組原樣複製，**不重新量化**。只有當你的舊草稿模型已不被 runtime 接受時才需要。 |
| `key-probe.py` ＋ `run-key-probe.bat` | 小型鍵盤診斷：印出每次按鍵的 `keysym` / `char` / `keycode` 並寫入 `key-probe.log`。當數字鍵盤在不同 NumLock／輸入法狀態下行為怪異時很有用（例如 NumLock 關閉時的數字鍵盤 `.`）。 |

### 疑難排解

| 症狀 | 處理 |
|---|---|
| `start-ui.bat` 沒反應 | 確認 `python` 在 `PATH` 上，或 Python 安裝在預設位置。請用 python.org 安裝檔並勾選 tcl/tk。 |
| 執行檔下拉是空的 | 沒有任何子資料夾含 `llama-server.exe`。請建立 `runtime/` 並放進去。 |
| 伺服器一啟動就結束 | 看日誌視窗。常見原因：草稿模型家族不符、模型架構不支援、顯存不足——試著調低 `-ngl`。 |
| 數字鍵盤小數點打不出來 | Windows 在 NumLock 關閉時會送不同的鍵碼。程式已處理；若仍有問題，執行 `tools/run-key-probe.bat` 看日誌。 |

### 授權

目前未附授權檔。發布或再散布前請自行選擇並加入。
