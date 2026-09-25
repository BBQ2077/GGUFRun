# GGUFRun — 本機 GGUF 推理控制台

Windows 本機 Tkinter 控制台：管理 `llama.cpp` 的 `llama-server`，另以 Image 模式管理 `stable-diffusion.cpp` 的 `sd-server`。LLM 和生圖模型、runtime 需自行依其原作者授權下載；本儲存庫只提供程式碼，不附帶模型、LoRA 或執行檔。

## 準備

- Windows 10/11、Python 3（含 Tkinter；建議 3.11 以上）。LLM 控制窗只用標準庫。
- 從 [llama.cpp](https://github.com/ggml-org/llama.cpp) 取得適合電腦的 Windows build，將 `llama-server.exe` 和所需 DLL 放在 `RUNTIMES/llama-official/`；要切換其他 build，可另放 `RUNTIMES/<名稱>/`。
- LLM GGUF 放在 `LLM-MODELS/`，或在介面中手動指定外部檔案。
- 生圖功能需另從 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) 取得包含 `sd-server.exe` 的相容 Windows build，放在 `RUNTIMES/stable-diffusion-cuda12-master-908-88411ef/`，或在 Image 控制窗選擇其他 runtime。所選版本須支援你啟用的參數與 Qwen-Image-2.1；不同 build 的功能可能不同。
- 生圖權重請自行取得，放於 `IMAGE-MODELS/` 或在 Image 控制窗選擇外部模型。文字編碼器、diffusion、VAE、視覺 mmproj 分欄選擇，依模型作者說明配對；預填的檔名只是選擇範例，不代表已附帶或已安裝。

> 安裝路徑不限；本程式以自身所在目錄為基準。

## 使用

1. 雙擊 `start-ui.bat`（或執行 `python gguf-ui.py`）。第一次開啟可能沒有可選模型或 runtime，須自行安裝上述檔案。
2. 在主視窗選 LLM、runtime 及選項，啟動本機服務；需要網頁則使用控制窗的開啟按鈕。
3. 點「🎨 Image 模式」開啟 `image-ui.py`；也可執行 `python image-ui.py`。選相容 runtime 和模型，保留 CPU offload（顯存有限時尤其重要），啟動 Image Server，再由該控制窗開啟繁中生圖網頁。網頁的自動存圖需經此控制窗的本機存圖服務。
4. 如只要 LLM 命令列模式，可用 `start-gguf.bat <模型檔名> [jinja] [dryrun]`；該批次檔使用 `RUNTIMES/llama-official/`。

設定檔會儲存在程式所在資料夾：LLM 使用 `ui-settings.json`、`models.json`、`presets.json`；Image 控制窗使用 `image-settings.json`。圖片與當次 server 日誌儲存在 `image-output/`。服務預設只監聽 `127.0.0.1`。

### Image 網頁

支援文字生圖、Img2Img 去噪、Qwen 多參考圖指令編輯，以及文字生圖的選配 Hi-res；單一頁面依序執行任務。尺寸快捷選項使用 32 的倍數（例如約 1080p 為 **1920×1088**，不是標準影片 1920×1080）；自訂寬高與 Hi-res 輸出亦檢查 32 對齊。伺服器仍可能改寫尺寸，存圖時會按回傳 PNG 的實際寬高保存；存圖服務仍驗證 PNG。

可選的 Viggle Qwen-Image-2.1 Turbo 模式只支援 txt2img 和 Qwen Edit：6 步 Euler、CFG 1、解析度依賴的 sigmas。權重**不在本儲存庫**；若取得作者的官方 LoRA，因目前 GGUF 使用融合的 `gate_up`，可使用 `tools/viggle-turbo/fuse_viggle.py` 將相容官方來源轉為程式指定檔名的 GGUF 對應版，放在 `IMAGE-MODELS/loras/`，然後重啟 Image Server。轉換工具需要 `numpy`；使用 LoRA 前請閱讀 [Viggle 原模型卡及授權條件](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)，**作者標示僅供非商業研究與評估**。本專案 MIT 程式碼授權不覆蓋第三方模型與權重。若無權重，請保持該選項關閉。

本專案程式碼依 [MIT License](LICENSE) 授權；第三方權重／runtime 遵循各原作者條件。
