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

使用者設定儲存在 `models.json`、`ui-settings.json`、`presets.json`、`image-settings.json`；生成圖片與 Image Server 日誌儲存在 `image-output/`。

## Viggle Qwen-Image-2.1 Turbo（生圖頁選配）

生圖頁有「Viggle Qwen-Image-2.1 Turbo LoRA（選填）」選單，預設關閉。選 Viggle Turbo 後，單張任務固定 Euler、CFG 1，並依輸出尺寸換算時間位移；「取樣步數」預設 6、可自訂 **4～150**：作者列出 4 步訓練節點、5／6／7 步高噪聲切分，以及密集文字建議的 **8 步**節點（[官方比較頁](https://huggingface.co/spaces/Viggle/Qwen-Image-2.1-viggle-turbo)的 5 個密集文字範例即用 8 步），9 步以上依作者「保留低噪聲節點、只細分高噪聲區段」原則推算，非逐一驗證的最佳值。可與 Hi-res 或去噪 Img2Img 併用；Hi-res 第二輪預設用獨立排程，勾選「第二輪沿用第一輪 turbo sigma」後改以放大後尺寸與第二輪步數重算同一套排程。關閉時步數恢復原值。

權重有兩種用法：掛 **LoRA**（`IMAGE-MODELS/loras/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128-fused-gguf.safetensors`；官方原始 r128 權重未隨附，重建衍生檔見 `tools/viggle-turbo/fuse_viggle.py`），或直接把作者轉好的 **融合 GGUF** 當生圖模型（此時開啟頁面會偵測模型名含 `viggle` 並自動啟用、免再掛 LoRA，避免重複套用）。兩種權重皆須自備；**本公開版不附任何 Viggle 權重**。[Viggle 模型卡](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)標示非商業研究與評估限制。

LoRA 一經伺服器套用即常駐，切回「關閉」不會歸還記憶體；網頁在選項旁有「♻ 釋放 LoRA 記憶體（重啟 Image Server）」按鈕，以相同啟動參數重啟後釋放。8 GB 顯卡請保留 CPU offload。

## 可選功能與授權

Image 網頁支援文字生圖、Img2Img、Qwen Edit 多參考圖與選配 Hi-res；支援哪些功能取決於模型與 sd-server 建置。Viggle Turbo LoRA 是選配，**本公開版不附權重也不附轉換後權重**；未取得合法且相容的權重時請維持關閉。[Viggle 模型卡](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)標示非商業研究與評估限制。請分別遵守模型、runtime、LoRA 的原始授權；本專案 [MIT 授權](LICENSE) 僅涵蓋本專案程式碼。

測試：`py -3 -m unittest discover -s tests -p 'test_*.py'`；瀏覽器佇列測試另需 Node.js：`node --test tests/test_image_queue.cjs`。正常使用不需要 Node.js。
