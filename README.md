# GGUFRun — Windows 本機 LLM／生圖控制台

GGUFRun 使用 Python 標準庫的 Tkinter 管理 [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`；另有 Image 控制窗管理 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) `sd-server`，以及用瀏覽器操作的繁中生圖頁面。**此公開版只有程式碼，不含模型、LoRA、執行檔或 CUDA DLL。** 兩個服務預設只監聽本機 `127.0.0.1`。

## 共通準備

安裝 [Windows 版 Python 3](https://www.python.org/downloads/windows/)（須包含 Tkinter）；執行 `py -3 -m tkinter` 確認可以開視窗。`start-ui.bat` 使用 `py -3` 或 PATH 上的 `python`；主程式只用 Python 標準庫，不需 pip 安裝。以下路徑皆相對於 GGUFRun 專案根目錄，缺少的資料夾可自行建立。**只使用 LLM 不必下載 Image 的模型／runtime；只用 Image 也不必下載 LLM 的。**

## LLM 模式：下載什麼、放哪裡

**1. llama.cpp 執行環境**：從 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下載符合硬體的 **Windows 已編譯建置**（CPU 或適合顯卡的 CUDA 等版本），將 `llama-server.exe` 連同**同一個包內所需的 DLL** 放在 `RUNTIMES/llama-official/`。不要只下載 GitHub 的 Source code 壓縮檔，也不要混放不同建置的 DLL。

**2. 至少一個 LLM 主模型 GGUF**：點下面的檔案頁下載其中**一個**，放進 `LLM-MODELS/`。UI 會自動掃描該目錄的 `*.gguf`，無需額外的模型註冊步驟；也可從 UI 登記其他目錄的模型。

| 選擇示例 | 下載檔案 | runtime 注意事項 |
|---|---|---|
| Spark-X2.5-4B | [Spark-X2.5-4B-Q4_K_M.gguf](https://huggingface.co/abenzerps/Spark-X2.5-4B-GGUF/blob/main/Spark-X2.5-4B-Q4_K_M.gguf) | 模型頁要求 llama.cpp **b10828 或更新／相容建置**，舊版本可能不認得此架構。 |
| Ternary Bonsai 2 27B | [Ternary-Bonsai-2-27B-PTQ1_0.gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf) | **官方 llama.cpp 不能跑 PTQ1_0**；另需 [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) 的相容 Windows 建置，放入 `RUNTIMES/llama-bonsai/`，並在 GUI 選擇該 runtime。 |

其他 [GGUF 模型](https://huggingface.co/models?search=GGUF)也能用，但要先核對模型架構、runtime 版本及授權。草稿模型、LLM LoRA、視覺 mmproj 均非純文字聊天必備。雙擊 `start-ui.bat`（或 `py -3 gguf-ui.py`）選主模型和 runtime 啟動；預設網址是 `http://127.0.0.1:18435/`。命令列 `start-gguf.bat Spark-X2.5-4B-Q4_K_M.gguf dryrun` 可先預覽指令，去掉 `dryrun` 才會啟動；此批次檔**只使用 `llama-official/`**，Bonsai 請用 GUI 選專用 runtime。

**3. 載入後自動釋放 RAM**：mmap 載入時，放上 GPU 的權重會先從模型檔讀進 RAM 再複製到顯存，Windows 版 llama.cpp 之後不放掉這些頁面，顯存和 RAM 各留一份。GUI 在 server 第一次回報 RUNNING 時，對自己啟動的 llama-server 呼叫一次 `EmptyWorkingSet`（`ram_release.py`），日誌顯示 `[RAM] 模型已載入：釋放已上顯存的模型檔頁面，working set X → Y GiB`；CPU 層（如 `--n-cpu-moe` 的專家）用到時自動從檔案快取讀回。實測 RX 9070 + 32 GB RAM 跑一個 68 GB 的 MoE GGUF（自動放置、`-c 32768`）：載入後的 14.4 GiB 已上顯存權重不再佔 RAM，跑完兩個請求後可用 RAM 由 0.3 GB 變成 7.4 GB，生成速度相同。`--no-mmap`、`--mlock`、載入模式 `mlock`／`mmap+mlock`／`none`／`dio` 不做釋放；外部接管的 server 與 EXL3 不處理。不需要時刪除 `ram_release.py` 即可。

## EXL3 模式（選配，AMD RX 9070 系列）

GGUFRun 也能啟動 [ExLlamaV3](https://github.com/turboderp-org/exllamav3) 的 EXL3 量化模型。執行環境另外發布在 **[BBQ2077/exl3-windows-gfx1201](https://github.com/BBQ2077/exl3-windows-gfx1201)**：原生 Windows、免安裝，內附 Python／PyTorch ROCm 10.1 與預編譯核心，**只支援 gfx1201（RX 9070 XT／9070／9070 GRE、Radeon AI PRO R9700）**。不用 EXL3 就不必下載，一般 GGUF 不受影響。

1. 從該倉庫的 Releases 下載 A 包（Python 環境）與 B 包（程式），都解壓到 `RUNTIMES/exl3-gfx1201-b/`（舊的 `exl3-gfx1201/` 仍可用，介面會優先選 `-b`）。A 包用 `v2026.10.16`，B 包抓最新。
2. EXL3 模型放 `LLM-MODELS-EXL3/<模型資料夾>/`，一個模型一個資料夾。例如 [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3) 的 `SC_3.00bpw_H4_V4` 分支（12.1 GiB，16 GB 顯卡適用），搭配 DFlash2 草稿 [parth-thakre/Qwen3.8-27B-DFlash2-EXL3-3.0bpw](https://huggingface.co/parth-thakre/Qwen3.8-27B-DFlash2-EXL3-3.0bpw) 做投機解碼。
3. 主模型下拉選 `[EXL3] …` 即改用 EXL3 啟動；「更多設定」的 **視覺（看圖，EXL3）** 可選 `off`／`gpu`／`ram`，`ram` 把視覺塔放系統 RAM，幾乎不佔顯存。

`RUNTIMES/` 下可以同時放多套 EXL3 環境：介面會優先用 `exl3-gfx1201-b`（新版），沒有時才退回 `exl3-gfx1201`。
「EXL3 進階」區可調的項目會依該套 `server.py` 實際支援的參數自動顯示；舊版沒有的開關會自動略過，不會讓 server 啟動失敗。

**同時生成多路**：新版 runtime 的 `--parallel` 讓多個對話同時生成（1～8 路，預設 2），每一路有自己的串流與取消。
每路 Context 分開設定；路數 × 每路 Context 就是總 KV 容量，顯存不夠時先降每路 Context 或路數。

**加速與快取**（EXL3 進階，皆為選配）：

| 項目 | 作用 |
|---|---|
| 多列快路徑 | 預設開，一次算 8～64 列；可改回 32 列上限或關閉比對。 |
| 自動草稿長度 `--auto-draft` | 依目前路數實測每種草稿深度的速度，自動挑最快的一檔（含暫停草稿）。 |
| 自動草稿（猜中率）`--auto-draft-accept` | 每一輪依最近各位置的草稿猜中率與每輪耗時挑草稿長度，換內容幾輪內就跟上；勾了就不用勾上一項。 |
| KV 型別 q3／q2 | 只在 EXL3：K、V 可分開選，例如 K `q4_0`＋V `q2`（`-cq 4,2`），每路 64K 時 2.20bpw 可開 3 路草稿或 4 路不開草稿。 |
| 前文快取 RAM／SSD | `-ccs` 把舊前文搬進系統 RAM；`-dcd`／`-dcs` 再往 SSD 存，重開 server 後仍可用。 |
| 重複內容長草稿 `-hdt` | 輸出照抄前文時一次驗證一大段，適合改程式碼、複述文件。 |
| 純 n-gram 草稿 `-ngram` | 沒載草稿模型時，從前文找相同片段當草稿。 |
| 實驗項 `#490`／`#517` | 草稿查表放 RAM、量化 KV 分段搬運；預設關閉，效果依顯卡而異。 |

RX 9070 16 GB 實測：Qwen3.8-27B 3.0bpw＋DFlash2，單路寫程式碼約 104～122 tok/s；2.20bpw 模型 8 路同時生成
（每路 8192 context、各 600 tokens）合計約 206～246 tok/s，開草稿深度 3 時最高。下載指令、顯存與常見問題見執行環境倉庫的 README。

## Image 模式：下載什麼、放哪裡

**1. stable-diffusion.cpp 執行環境**：從 [stable-diffusion.cpp Releases](https://github.com/leejet/stable-diffusion.cpp/releases) 下載符合硬體的 Windows **已編譯**包，將 `sd-server.exe` 與同版所需 DLL 放在 `RUNTIMES/stable-diffusion-cuda12-master-908-88411ef/`（程式預設值，對應 [master-908-88411ef](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-908-88411ef)）。若下載別的版本，可放 `RUNTIMES/<自訂名稱>/`，再在 Image 控制窗選對應 runtime。CUDA 建置需要相容 NVIDIA GPU／驅動；不同版本可能不支援部分加速選項。

**2. Qwen-Image-2.1 範例模型組合**：文字生圖需下載**前三項**；只有使用 Qwen Edit 參考圖指令修圖時才需第四項。每項點來源檔案頁下載，放在右欄位置：

| Image 控制窗欄位 | 要下載的檔案 | 預設放置位置 |
|---|---|---|
| 生圖模型（diffusion） | [qwen-image-2.1-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/blob/main/qwen-image-2.1-Q4_K_M.gguf) | `IMAGE-MODELS/qwen-image-2.1-Q4_K_M.gguf` |
| LLM 文字編碼器 | [Qwen3VL-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/blob/main/Qwen3VL-8B-Instruct-Q4_K_M.gguf) | `IMAGE-MODELS/Qwen3VL-8B-Instruct-Q4_K_M.gguf` |
| VAE | [qwen_image_2.1_vae_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/vae/qwen_image_2.1_vae_bf16.safetensors) | `IMAGE-MODELS/qwen_image_2.1_vae_bf16.safetensors` |
| 視覺 mmproj（Qwen Edit 選配） | [mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/blob/main/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf) | `IMAGE-MODELS/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf` |

文字編碼器來源的實際檔名是 `Qwen3VL`（不是有連字號的 `Qwen3-VL`）；選擇其他檔名或放置目錄時，在 Image 控制窗改選**實際檔案路徑**，不要把不同權重只改名當作同一模型。生圖使用的 Qwen3VL 編碼器放在 `IMAGE-MODELS/`，不是 LLM 模式的 `LLM-MODELS/`。參數組合參考 [Unsloth 模型卡](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)及 [stable-diffusion.cpp 文件](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md)。換用其他生圖模型時依其說明選擇 checkpoint／CLIP-L／CLIP-G／T5XXL 等欄位，無需的欄位留白。

**3. 啟動**：在主視窗按「Image 模式」，或執行 `py -3 image-ui.py`；核對 runtime 與各模型欄位、啟動 Image Server（預設 `127.0.0.1:18436`），**從 Image 控制窗的按鈕**打開繁中生圖頁面，讓自動存圖服務連接上。填提示詞、加入佇列並按「開始／繼續」，PNG 自動儲存在 `image-output/`。佇列依賴該瀏覽器頁面保持開啟，關頁不會在背景繼續生成。

下載後大致如下（目錄可自行建立，軟體不會自動下載）：

```text
GGUFRun/
  start-ui.bat               gguf-ui.py          image-ui.py
  image_save_service.py      assets/image-web.html
  RUNTIMES/
    llama-official/           llama-server.exe + 同包 DLL（一般 LLM）
    llama-bonsai/            相容 fork 的 llama-server.exe + 同包 DLL（Bonsai 選配）
    stable-diffusion-cuda12-master-908-88411ef/  sd-server.exe + 同包 DLL（Image）
  LLM-MODELS/                自行下載的 LLM 主模型 GGUF
  IMAGE-MODELS/              自行下載的生圖權重／編碼器／VAE／選配 mmproj
  image-output/              圖片與 Image Server 日誌（執行後產生）
```

使用者設定儲存在 `models.json`、`ui-settings.json`、`presets.json`、`image-settings.json`、`image-presets.json`（都只在按儲存時寫入）；生成圖片與 Image Server 日誌儲存在 `image-output/`。

## 參數模板與儲存

控制窗與生圖頁都是**手動儲存**（跟 LLM 控制窗一樣）：改參數、套模板只影響這次，要留到下次開啟請按控制窗「💾 儲存參數」或生圖頁模板列的「💾 儲存」；未存時會顯示提示。只是測參數就不用存，關掉即還原。

控制窗內建兩個**完整模板**（存在 `image-presets.json`，第一次按儲存才寫檔），選了會一起換 runtime、全部模型欄位、額外指令與參考圖參數；選到對應的生圖模型檔也會自動套用。內建模板可改、可覆蓋，刪掉後不會再自動補回。自己另存模板時可選「完整模板」或只存「怎麼跑」（offload、取樣快取、VAE 分塊、Flash Attention…）。

| 內建模板 | runtime | 生圖模型 | LLM 編碼器 | VAE | 額外 |
|---|---|---|---|---|---|
| Qwen-Image-2.1-Turbo 官方 8 步 | `sd-master-f0aac52-bin-win-rocm-10.1.0-x64` | `qwen-image-2.1-turbo-Q4_K_M.gguf` | `Qwen3VL-8B-Instruct-Q4_K_M.gguf`（＋ mmproj） | `qwen_image_2.1_vae_bf16.safetensors` | 沿用你選的取樣快取 |
| MOSS-Image 4 步（data-space） | `sd-master-b13a482-moss4-gfx1201-rocm-10.1.0-x64` | `moss_image_sft_4step_int8_convrot.safetensors` | `Qwen3.5-9B-Q4_K_M.gguf` | `flux2-vae_small.safetensors` | `--tokenizer IMAGE-MODELS/tokenizer_qwen3.5.json` |

生圖頁另有自己的模板列（內建「頁面預設」「Qwen Turbo 8 步」「MOSS 4 步」，可另存／刪除自訂模板），管的是步數、CFG、取樣器、加速排程、LoRA 勾選與強度等每張圖的參數。開頁時若載入的是 Qwen Turbo 或 MOSS 4 步模型，會自動套用對應模板（不寫入存檔）。

## Qwen-Image-2.1-Turbo 官方 8 步（生圖頁「加速排程」）

[Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) 是官方 8 步加速版，權重自帶推薦排程（diffusers [PR #14950](https://github.com/huggingface/diffusers/pull/14950) 的 pipeline sigma）。生圖頁「加速排程」選「Qwen-Image-2.1-Turbo 官方 8 步」（模型檔名含 `turbo` 時自動選）後，每次請求固定 Euler、CFG 1、8 步，並以 `custom_sigmas` 送出官方清單：

```text
1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0
```

shift 1.0、不隨解析度位移；sigma 欄可自行微調，「還原官方」恢復。需要支援每次請求 `custom_sigmas` 的 sd-server（`sd-master-f0aac52` 起）。Hi-res 勾「第二輪沿用官方 turbo sigma」時，伺服器依第二輪去噪強度截取同一清單的尾段。本機 1024² 實測（RX 9070）：官方 sigma 與一般排程同樣 8 步，每步計算量相同（取樣約 19～23 秒），官方 sigma 的招牌與小黑板文字都正確、畫面較乾淨。

舊的 Viggle Turbo（LoRA／融合 GGUF、6 步排程）已移除，請改用官方 Turbo。舊存圖檔名的 `-6step`／`-turbo` 標籤仍可讀取；新圖改標 `-qwenturbo` 或 `-moss4`。

## MOSS-Image 4 步（實驗性，需 GGUFRun 自編 runtime）

[MOSS-Image-SFT-4-steps](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps) 是 [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) 的 4 步蒸餾版：它不走一般 sigma 排程，而是在 data-space 迴圈（`x = z + Q·x0`、`v = T(x, step_code)`、`x0 = z − v`）跑 4 個學到的 step code，官方 stable-diffusion.cpp 沒有這個取樣器。

需要準備：

- runtime：`RUNTIMES/sd-master-b13a482-moss4-gfx1201-rocm-10.1.0-x64/`（stable-diffusion.cpp master `b13a482`＋同目錄 `ggufrun-moss-dataspace.patch`，ROCm 10.1.0、gfx1201；來源與 SHA-256 見 `rocm-runtime.json`）。
- 底模：`IMAGE-MODELS/moss_image_sft_4step_int8_convrot.safetensors`，是 MOSS-Image-SFT int8 底模換上 4 步版 fp32 `time_embedding` 的衍生檔。
- `IMAGE-MODELS/moss-4step/dataspace.safetensors`（step code 與 latent mix Q），生圖頁每次請求以 `extra_sample_args` 帶 `dataspace=…` 啟用，只對 MOSS 模型生效，並自動關閉 spectrum 快取。
- LoRA：`IMAGE-MODELS/loras/MOSS-Image-SFT-4-steps-lora.safetensors`，**必須套用，沒套圖會整張糊掉**。生圖頁的「MOSS 4 步」模板（載入 MOSS 模型開頁時會自動套用）會在 LoRA 清單幫你勾好（強度 1，執行時套用、不合併進底模），要取消可自己取消；選 MOSS 4 步卻沒勾時，加入佇列會提醒。
- LLM `Qwen3.5-9B-Q4_K_M.gguf`、VAE `flux2-vae_small.safetensors`、`tokenizer_qwen3.5.json`；控制窗選「MOSS-Image 4 步」模板即全部帶入。

本機 1024² 實測：取樣約 22 秒（含載入約 37 秒），黑板菜單文字正確。

**圖生圖**：MOSS 只發布文生圖，沒有指令編輯版本，LoRA 也補不了。runtime 有實作 SDEdit 式 Img2Img：以原圖為起點，只跑後段 step code（去噪強度 1／0.75／0.5／0.25 分別從第 1／2／3／4 個 code 開始）。可用但 4 步蒸餾每一步都重算整張圖，強度 0.5、0.25 幾乎重畫，0.1 能保留構圖、換光線但文字會糊。適合同構圖換風格；要忠實改圖請用 Qwen-Image-2.1 Turbo 的 Img2Img 或 Qwen Edit（MOSS 不支援 Qwen Edit 參考圖，網頁會阻擋）。

**自己重編 runtime**：`tools/moss4/build-sd.bat`（不動系統環境，需要 VS2022 C++ 工作負載、git、uv）。它會在 `tools/moss4/.work/` 用 uv 建 Python 3.12 venv、從 AMD 官方源 `https://stable.repo.amd.com/rocm/whl-next/` 裝 `rocm[libraries,devel]==10.1.0`（含 HIP／clang），抓 stable-diffusion.cpp `b13a482` 並套上 `ggufrun-moss-dataspace.patch`，仿官方 CI 以 `-DSD_HIPBLAS=ON -DGPU_TARGETS=gfx1201` 編譯，最後由 `package_runtime.py` 放進 `RUNTIMES/` 並寫好 SHA-256 清單。編完 `.work` 約 4.5 GB 可刪。執行時 PATH 需含 runtime 目錄與 `RUNTIMES/rocm-sdk-10.1.0/bin`（控制窗會自動設定）。

**自己做 4 步底模**：把 [akhaliq/MOSS-Image-SFT-4-steps](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps) 的 `time_embedding.safetensors` 放到 `IMAGE-MODELS/moss-4step/`，執行 `python tools/moss4/make_moss_4step_base.py`，會從 `moss_image_sft_int8_convrot.safetensors` 產生 `moss_image_sft_4step_int8_convrot.safetensors`（約 5.4 GB）。

## 可選功能與授權

Image 網頁支援文字生圖、Img2Img、Qwen Edit 多參考圖與選配 Hi-res；支援哪些功能取決於模型與 sd-server 建置。**本公開版不附任何模型、LoRA 或衍生權重**（含 MOSS 4 步底模）；Qwen-Image-2.1-Turbo、MOSS-Image 及其 4 步權重請依各自模型卡授權自行下載。請分別遵守模型、runtime、LoRA 的原始授權；本專案 [MIT 授權](LICENSE) 僅涵蓋本專案程式碼。

測試：`py -3 -m unittest discover -s tests -p 'test_*.py'`；瀏覽器佇列測試另需 Node.js：`node --test tests/test_image_queue.cjs`。正常使用不需要 Node.js。
