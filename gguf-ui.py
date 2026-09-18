#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama-server GUI 控制台（本地模型管理｜可攜式：自動以本程式所在資料夾為基準）
- 自動掃描「本程式所在資料夾」的 *.gguf 當主模型；不在這資料夾的模型要用「➕ 添加模型」手動加
- 啟停本地 llama-server（自動偵測本資料夾下所有 runtime，可在 UI 切換）
- 模式：Normal / 投機解碼（草稿模型可「自動配對」，也可從下拉手動指定，含手動添加的 DF）
- 網頁按鈕：用瀏覽器開啟 http://127.0.0.1:<埠>/ （網址跟隨「埠」欄位變化）
- 選項：-ngl / -ngld、KV 型別 K/V（f16/q8_0/q4_0）、Flash-Attn、Context 大小、KV 放 RAM、思考模式、思考預算、埠
- 進階取樣：主視窗「⚙ 進階取樣設定」開啟獨立視窗（4 分頁 / 21 個參數）
  temp / top-p / top-k / min-p / repeat-penalty / repeat-last-n / presence-penalty / frequency-penalty
  DRY 系列（multiplier / base / allowed-length / penalty-last-n / sequence-breaker）
  XTC / typical / dynatemp / mirostat｜留空 = 不送出該參數
- 防重複：開啟時靜默掃描、每次「啟動」強制預檢（接管／終止重啟／取消）
- 防孤兒：關閉視窗自動收掉 server，程式結束 atexit 保底清理（只用 exact PID）
- 顯示狀態與即時日誌
只用標準庫 tkinter，不需額外安裝。建議用系統 python3 執行。

版本：2.0（2026-09-18）
"""
import os, re, sys, subprocess, threading, time, socket, json, shutil, urllib.request, atexit, webbrowser
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

try:                                    # 通用：以本程式所在資料夾為基準，不寫死路徑
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()
def find_runtimes():
    """掃描本資料夾下所有含 llama-server.exe 的子資料夾（可放多套 runtime 切換）。

    回傳 [(顯示名, 目錄)]，排序：runtime/ 優先，再來其他資料夾。
    """
    out, seen = [], set()
    for d, label in (("runtime", "官方 runtime/"),):
        p = os.path.join(BASE, d)
        if os.path.exists(os.path.join(p, "llama-server.exe")):
            out.append((label, p))
            seen.add(os.path.normcase(p))
    try:
        for n in sorted(os.listdir(BASE)):
            p = os.path.join(BASE, n)
            if (os.path.isdir(p) and os.path.normcase(p) not in seen
                    and os.path.exists(os.path.join(p, "llama-server.exe"))):
                out.append((n + "/", p))
    except Exception:
        pass
    return out


RUNTIMES = find_runtimes()
SRV = (os.path.join(RUNTIMES[0][1], "llama-server.exe") if RUNTIMES
       else os.path.join(BASE, "runtime", "llama-server.exe"))
# 本專案自己會用到的所有 runtime 目錄（判斷「殘留的測試 server」用）
RUNTIME_DIRS = tuple(os.path.normcase(d).lower() for _n, d in RUNTIMES) or \
               (os.path.normcase(os.path.dirname(SRV)).lower(),)
REGISTRY = os.path.join(BASE, "models.json")     # 手動添加的模型清單（長久儲存）


def _to_recycle_bin(path):
    """把檔案丟進 Windows 資源回收桶（可還原）；成功回 True。

    優先走 Shell API（SHFileOperationW），這樣才真的進回收桶。
    失敗時回 False，由呼叫端決定要不要改成永久刪除。
    """
    try:
        import ctypes
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [("hwnd", wintypes.HWND),
                        ("wFunc", wintypes.UINT),
                        ("pFrom", wintypes.LPCWSTR),
                        ("pTo", wintypes.LPCWSTR),
                        ("fFlags", ctypes.c_uint16),
                        ("fAnyOperationsAborted", wintypes.BOOL),
                        ("hNameMappings", ctypes.c_void_p),
                        ("lpszProgressTitle", wintypes.LPCWSTR)]

        FO_DELETE = 3
        FOF_ALLOWUNDO = 0x0040          # 進回收桶（可還原）
        FOF_NOCONFIRMATION = 0x0010
        FOF_SILENT = 0x0004
        FOF_NOERRORUI = 0x0400

        op = SHFILEOPSTRUCTW()
        op.hwnd = None
        op.wFunc = FO_DELETE
        # 這串最後要多一個 \0（雙重結尾）
        op.pFrom = os.path.abspath(path) + "\0\0"
        op.pTo = None
        op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
        res = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        if res == 0 and not op.fAnyOperationsAborted:
            return True
        return False
    except Exception:
        return False


def fix_path(p):
    """路徑修復：檔案不在原位、但同檔名就在本資料夾 → 自動改用本資料夾那份。

    資料夾搬移/改名之後，設定檔與 models.json 裡的舊路徑不必手改也能用。
    """
    if not isinstance(p, str) or not p:
        return p
    if os.path.exists(p):
        return p
    alt = os.path.join(BASE, os.path.basename(p))
    return alt if os.path.exists(alt) else p


_ALIAS_CACHE = {}
_GGUF_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_F = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


_GGUF_CACHE = {}


def gguf_meta(path, want, limit=64):
    """讀 GGUF 開頭的中介資料（只掃前 limit 筆，很快）。找不到回 None。"""
    ck = (os.path.normcase(os.path.abspath(path)), want)
    if ck in _GGUF_CACHE:
        return _GGUF_CACHE[ck]
    val = None
    try:
        import struct
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError("not gguf")
            struct.unpack("<I", f.read(4))
            struct.unpack("<Q", f.read(8))
            n = struct.unpack("<Q", f.read(8))[0]

            def rs():
                return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")

            def rv(t):
                if t == 8:
                    return rs()
                if t == 9:
                    et = struct.unpack("<I", f.read(4))[0]
                    cnt = struct.unpack("<Q", f.read(8))[0]
                    for _ in range(cnt):
                        if et == 8:
                            f.read(struct.unpack("<Q", f.read(8))[0])
                        else:
                            rv(et)
                    return ""
                return struct.unpack("<" + _GGUF_F[t], f.read(_GGUF_SZ[t]))[0]

            for _ in range(min(n, limit)):
                k = rs()
                t = struct.unpack("<I", f.read(4))[0]
                v = rv(t)
                if k == want:
                    val = v
                    break
    except Exception:
        val = None
    _GGUF_CACHE[ck] = val
    return val


def gguf_arch(path):
    """模型架構（general.architecture），例如 qwen35 / gemma4 / spark2_5 / k2-horizon。"""
    a = gguf_meta(path, "general.architecture")
    return a if isinstance(a, str) else ""


def gguf_mtp_layers(path):
    """模型 header 的 <arch>.nextn_predict_layers（內建 MTP 層數）；沒有回 0。

    有這個欄位 = 內建 MTP，可以用 --spec-type draft-mtp「不指定 -md」來啟用
    （實測：`creating MTP draft context against the target model`，不需要外掛檔案）。
    沒有這個欄位 = 內建 MTP 不存在；此時送 --spec-type draft-mtp 會直接載入失敗
    （實測錯誤：context type MTP requested but model doesn't contain MTP layers）。
    """
    a = gguf_arch(path)
    if not a:
        return 0
    v = gguf_meta(path, a + ".nextn_predict_layers", limit=120)
    try:
        return int(v)
    except Exception:
        return 0


# 實測結果（不是猜的）→ 主模型架構：能載它的 runtime 目錄名（空 tuple＝現有 runtime 都載不動）
#   spark2_5  : 官方 runtime/ 可正常載入並用 GPU
#   k2-horizon: 實測連官方 runtime/ 也是 unknown model architecture
RUNTIME_ARCH = {"spark2_5": ("runtime",), "k2-horizon": ()}


def model_alias(path):
    """--alias（網頁標題／對外模型名）：用模型檔名（跟 UI 下拉看到的一致）。

    檔名清不出安全字元時（例如全中文）才退回去讀 GGUF 的 general.name。
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    al = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.")
    if al:
        return al
    key = os.path.normcase(os.path.abspath(path))
    if key in _ALIAS_CACHE:
        return _ALIAS_CACHE[key]
    name = gguf_meta(path, "general.name") or ""
    al = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    _ALIAS_CACHE[key] = al or "local-model"
    return _ALIAS_CACHE[key]


# 檔名關鍵字（僅供「建議」用，真正的分類以 models.json 記住的 kind 為準）
#   DRAFT_STRONG：明確的外掛草稿/投機模型字樣
#   mtp         ：模糊——很多「主模型」本身就內建 MTP，所以不可以用它來排除主模型
DRAFT_HINT = ("dspark", "dflash", "draft", "mtp")
DRAFT_STRONG = ("dspark", "dflash", "draft")


# ---------- 手動添加清單（models.json 長久儲存）----------
def load_registry():
    """讀取手動添加的模型清單；檔壞掉/不存在一律回傳空清單。"""
    try:
        with open(REGISTRY, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return []
    lst = d.get("manual") if isinstance(d, dict) else None
    if not isinstance(lst, list):
        return []
    out = []
    for e in lst:
        if isinstance(e, dict) and isinstance(e.get("path"), str) and e["path"]:
            e = dict(e)
            e["path"] = fix_path(e["path"])
            if isinstance(e.get("drafter"), str) and e["drafter"]:
                e["drafter"] = fix_path(e["drafter"])
            out.append(e)
    return out


def save_registry(lst):
    """寫回 models.json（長久儲存）；成功回 True。"""
    try:
        with open(REGISTRY, "w", encoding="utf-8") as f:
            json.dump({"manual": lst}, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def registry_kind(path):
    """這個路徑在 models.json 裡記住的 kind（"main"/"drafter"）；沒記過回 ""。"""
    for e in load_registry():
        if os.path.normcase(e["path"]) == os.path.normcase(path):
            k = e.get("kind")
            if k in ("main", "drafter"):
                return k
    return ""


def looks_like_sidecar(path):
    """粗略判斷某個 .gguf 是否像「外掛小檔」（草稿/投機模型）。

    真正的草稿模型通常遠小於完整主模型；這裡用檔案大小當粗略訊號：
    小於 1 GiB 視為外掛小檔（自帶 mtp 的主模型通常 5 GiB 以上，不會被誤判）。
    """
    try:
        return os.path.getsize(path) < 1 * 2 ** 30
    except Exception:
        return False


def list_models():
    """回傳 [(檔名, 顯示字串, 位元組大小)]：主模型候選。

    規則（記憶優先，檔名只當建議）：
      1. models.json 記成 drafter → 排除
      2. 未記過的檔名含 DRAFT_STRONG（draft/dspark/dflash）→ 排除（明顯是外掛草稿）
      3. 其餘全部收錄，包含檔名含 mtp 的（可能只是主模型內建 MTP）
    """
    out = []
    try:
        names = sorted(os.listdir(BASE))
    except Exception:
        return out
    for fn in names:
        low = fn.lower()
        if not low.endswith(".gguf"):
            continue
        p = os.path.join(BASE, fn)
        k = registry_kind(p)
        if k == "drafter":
            continue                                  # 記過是草稿 → 不列主模型
        if not k:
            if any(h in low for h in DRAFT_STRONG):
                continue                              # 沒記過 + 明確草稿字樣 → 排除
            if "mtp" in low and looks_like_sidecar(p):
                continue                              # 沒記過 + mtp 小檔 → 像外掛草稿
        try:
            size = os.path.getsize(p)
        except Exception:
            continue
        hint = "   ← 主模型（內建 MTP）" if ("mtp" in low and size >= 1 * 2 ** 30) else ""
        out.append((fn, f"{fn}   {size / 2 ** 30:.2f} GiB{hint}", size))
    return out


def first_model_path():
    """本資料夾第一個可用主模型（取代寫死的檔名；沒有就回空字串）。"""
    ms = list_models()
    return os.path.join(BASE, ms[0][0]) if ms else ""


DEFAULT_MODEL = first_model_path()
PORT = 18435


# ---------- 模型清單：自動掃描 + 手動添加（models.json 長久儲存）----------
def guess_drafter(model_path, extra=None):
    """依主模型配對草稿模型，回傳 (草稿路徑 or None, --spec-type 值 or None)。
    手動登錄若有指定則優先；否則依檔名自動配對（bonsai→dspark / gemma→mtp）。"""
    if isinstance(extra, dict) and extra.get("drafter") and os.path.exists(extra["drafter"]):
        return extra["drafter"], (extra.get("spec") or "draft-mtp")
    low = os.path.basename(model_path).lower()
    try:
        files = sorted(os.listdir(BASE))
    except Exception:
        files = []

    def pick(*keys):
        for fn in files:
            fl = fn.lower()
            if fl.endswith(".gguf") and all(k in fl for k in keys):
                return os.path.join(BASE, fn)
        return None

    if "bonsai" in low:
        d = pick("bonsai", "dspark", "dflash") or pick("bonsai", "dspark")
        return (d, "draft-dspark") if d else (None, None)
    if "gemma" in low:
        d = pick("mtp", "gemma")
        return (d, "draft-mtp") if d else (None, None)
    return (None, None)


def build_model_list():
    """回傳 [(顯示字串, 完整路徑, 來源, 附加資料)]；來源 = auto（掃描本程式資料夾）/ manual（手動添加）。"""
    out, seen = [], set()
    for fn, disp, _size in list_models():
        p = os.path.join(BASE, fn)
        out.append((disp, p, "auto", None))
        seen.add(os.path.normcase(p))
    for e in load_registry():
        if e.get("kind") == "drafter":
            continue                     # 手動添加的草稿模型（DF）不列在主模型清單
        p = e["path"]
        if os.path.normcase(p) in seen:
            continue
        nm = e.get("label") or os.path.basename(p)
        try:
            sz = f"{os.path.getsize(p) / 2 ** 30:.2f} GiB"
        except Exception:
            sz = "檔案不存在"
        out.append((f"[手動] {nm}   {sz}", p, "manual", e))
        seen.add(os.path.normcase(p))
    return out


def list_drafters():
    """回傳 [(檔名, 完整路徑, 位元組大小)]：本資料夾內看起來是草稿/投機模型的 .gguf。

    記憶優先：models.json 記成 main → 不收；記成 drafter → 一定收；
    沒記過的才依檔名判斷（draft/dspark/dflash，或 mtp 但檔案明顯偏小＝外掛草稿）。
    """
    out = []
    try:
        names = sorted(os.listdir(BASE))
    except Exception:
        return out
    for fn in names:
        low = fn.lower()
        if not low.endswith(".gguf"):
            continue
        p = os.path.join(BASE, fn)
        k = registry_kind(p)
        if k == "main":
            continue                                  # 記過是主模型 → 不列草稿
        if not k:
            if any(h in low for h in DRAFT_STRONG):
                pass                                  # 明確草稿字樣 → 收
            elif "mtp" in low and looks_like_sidecar(p):
                pass                                  # 含 mtp 但檔案偏小 → 像外掛草稿
            else:
                continue
        try:
            out.append((fn, p, os.path.getsize(p)))
        except Exception:
            continue
    return out


def spec_for_drafter(path, extra=None):
    """依草稿模型檔名推斷 --spec-type（手動登錄若有指定 spec 則優先）。
    本 build 可用：none / draft-simple / draft-eagle3 / draft-mtp / draft-dflash / draft-dspark。"""
    if isinstance(extra, dict) and extra.get("spec"):
        return extra["spec"]
    low = os.path.basename(path).lower()
    if "mtp" in low or "assistant" in low:
        return "draft-mtp"
    if "dflash" in low:
        return "draft-dflash"
    if "dspark" in low:
        return "draft-dspark"
    if "eagle" in low:
        return "draft-eagle3"
    return "draft-simple"


FAMILY_TOKENS = ("gemma", "bonsai", "qwen", "minicpm", "spark", "phi", "mistral", "glm", "llama")


def drafter_mismatch(main_path, drafter_path):
    """草稿模型必須與主模型同家族。回傳警告字串（'' ＝ 看起來相配）。
    實測：配錯的 DF 不會友善報錯，llama-server 會直接崩潰（0xC0000005）。"""
    ml = os.path.basename(main_path).lower()
    dl = os.path.basename(drafter_path).lower()
    mf = next((t for t in FAMILY_TOKENS if t in ml), "")
    df = next((t for t in FAMILY_TOKENS if t in dl), "")
    if not mf or not df:
        return (f"無法確認 {os.path.basename(drafter_path)} 是否與主模型同家族"
                f"（建議先用「自動（依主模型配對）」）")
    if mf != df:
        return (f"DF 家族不符（主模型像 {mf}、草稿模型像 {df}）"
                f"→ llama-server 可能無法建立上下文而直接崩潰")
    return ""


def build_drafter_list():
    """回傳 (下拉顯示清單, {顯示: 路徑})。特殊值："" = 自動配對、"-" = 不使用。"""
    auto, none = "自動（依主模型配對）", "不使用外掛 DF（內建 MTP 仍會啟用）"
    vals, mp = [auto, none], {auto: "", none: "-"}
    for fn, p, size in list_drafters():
        d = f"{fn}   {size / 2 ** 30:.2f} GiB"
        vals.append(d)
        mp[d] = p
    for e in load_registry():                 # 手動添加的草稿模型（DF）
        if e.get("kind") != "drafter":
            continue
        p = e["path"]
        if any(os.path.normcase(v) == os.path.normcase(p) for v in mp.values()):
            continue
        try:
            sz = f"{os.path.getsize(p) / 2 ** 30:.2f} GiB"
        except Exception:
            sz = "檔案不存在"
        d = f"[手動] {e.get('label') or os.path.basename(p)}   {sz}"
        vals.append(d)
        mp[d] = p
    return vals, mp

# 進階取樣參數定義：(旗標, 變數key, 標籤, 提示, 預設值)
# 預設值取自本 build `llama-server.exe --help` 實查結果
ADV_CORE = [
    ("--temp",  "temp", "Temperature", "越高越有創意、越低越穩定。llama.cpp 內建預設 0.80", "0.8"),
    ("--top-p", "topp", "Top-P",       "只從累積機率前 P 的字挑。llama.cpp 內建預設 0.95，1.0=停用", "0.95"),
    ("--top-k", "topk", "Top-K",       "只從機率最高的 K 個字挑。llama.cpp 內建預設 40，0=停用", "40"),
    ("--min-p", "minp", "Min-P",       "丟掉機率低於「最高機率×P」的字。llama.cpp 內建預設 0.05，0=停用", "0.05"),
]
ADV_OPT = [
    ("--repeat-penalty",      "rpen",   "重複懲罰",              "懲罰重複出現的字詞，1.0=不罰。預設 1.00，1.0=停用"),
    ("--repeat-last-n",       "rlastn", "重複懲罰範圍",          "往前看幾個 token 做重複懲罰。預設 64，0=停用"),
    ("--presence-penalty",    "ppen",   "Presence penalty",     "鼓勵講新內容，正值=少重複。預設 0.00，0.0=停用"),
    ("--frequency-penalty",   "fpen",   "Frequency penalty",    "按出現次數加重懲罰，負值=鼓勵重複。預設 0.00，0.0=停用"),
    ("--dry-multiplier",      "drym",   "DRY 強度",             "DRY：抑制整段鬼打牆式重複。預設 0.00，0.0=停用"),
    ("--dry-base",            "dryb",   "DRY 底數",             "DRY 指數底數，越大罰越重。預設 1.75"),
    ("--dry-allowed-length",  "dryal",  "DRY 容忍長度",         "允許重複的最短長度。預設 2"),
    ("--dry-penalty-last-n",  "dryln",  "DRY 回溯範圍",         "DRY 只看最後 n 個 token。預設 64，0=停用"),
    ("--dry-sequence-breaker", "drybr", "DRY 斷句符號",         "決定重複片段怎麼切。預設無，常見值 \\n；需與強度併用"),
    ("--xtc-probability",     "xtcp",   "XTC 機率",             "隨機剔除最可能的字以增加多樣性。預設 0.00，0.0=停用"),
    ("--xtc-threshold",       "xtct",   "XTC 門檻",             "XTC 剔除的機率門檻。預設 0.10，1.0=停用"),
    ("--typical",             "typ",    "Typical 取樣 p",       "挑「最典型」的字，平衡穩定與變化。預設 1.00，1.0=停用"),
    ("--dynatemp-range",      "dtr",    "動態溫度範圍",         "依候選機率自動調溫，避免亂選。預設 0.00，0.0=停用"),
    ("--dynatemp-exp",        "dte",    "動態溫度指數",         "動態溫度的曲線形狀。預設 1.00"),
    ("--mirostat",            "mir",    "Mirostat 模式",        "自適應取樣，自動穩定困惑度。預設 0=停用；可填 1 或 2"),
    ("--mirostat-lr",         "mirlr",  "Mirostat 學習率 eta",  "Mirostat 收斂速度。預設 0.10"),
    ("--mirostat-ent",        "mirent", "Mirostat 目標熵 tau",  "Mirostat 想維持的隨機度。預設 5.00"),
]

# 投機解碼（草稿模型 DF）專用參數。留空 = 不送出，改用 llama.cpp 內建預設。
# 預設值取自本 build `llama-server.exe --help` 實查結果。
#   n-max 官方預設是 3（不是 4！）——實測在 VRAM 小的卡上 2~3 往往比 4 快，
#   因為草稿太長的額外成本會吃掉收益，硬體越受限越明顯。
ADV_SPEC = [
    ("--spec-draft-n-max",    "sdnmax",  "草稿長度上限 (n-max)",   "一次最多草稿幾個 token。llama.cpp 預設 3。太大在 VRAM 小的卡上反而變慢"),
    ("--spec-draft-n-min",    "sdnmin",  "草稿長度下限 (n-min)",   "最少用幾個草稿 token。llama.cpp 預設 0（= 不限制）"),
    ("--spec-draft-p-split",  "sdpsplit", "草稿切分機率 (p-split)", "草稿切分機率。llama.cpp 預設 0.10"),
    ("--spec-draft-p-min",    "sdpmin",  "草稿最低機率 (p-min)",   "貪婪解碼時的最低草稿機率。llama.cpp 預設 0.00"),
]

# MoE 權重放置（專家層）相關參數。本 build 實查旗標：
#   -ncmoe N           / --n-cpu-moe N             主模型前 N 層的專家權重放 CPU
#   --spec-draft-ncmoe / -ncmoed                   草稿模型（DF）版（本 UI 不做）
#   -ncffn N / --n-cpu-ffn N                       密集型 FFN（非 MoE 模型用）
# 只在「主模型是 MoE」時才有意義；上限 = 該模型自己的層數（header 的 block_count）。
# 留空 = 不送出該參數，完全沿用 llama.cpp 內建行為（全部照 -ngl 放 GPU）。
# 模型載入模式相關（本 build 實查旗標）：
#   -lm  / --load-mode   模型載入方式（預設 auto）
#   -lzm / --lazy-mode   特定 tensor 是否延後從磁碟讀取（需要 mmap）
ADV_LOAD = [
    ("--load-mode", "loadmode", "模型載入模式 (-lm)",
     "auto＝偵測不到就退回 mmap（預設）｜mmap＝記憶體映射｜mlock＝強制常駐 RAM｜mmap+mlock＝兩者"
     "｜dio＝DirectIO｜none＝不用特殊模式"),
    ("--lazy-mode", "lazymode", "延遲讀取模式 (-lzm)",
     "on＝大型 tensor（如 per-layer embedding）改成要用才從磁碟讀｜auto＝只對大於 4GiB 的 tensor 這樣做（預設）"
     "｜off＝一律常駐。需要 mmap"),
]

ADV_LORA = [
    ("--lora", "lora", "LoRA 適配檔路徑",
     "可搭配的 LoRA adapter（.gguf）路徑。留空＝不送這個參數。"),
]

ADV_EXTRA = [
    ("", "extra_args", "自訂指令",
     "直接附加到啟動指令最後。例：--lora-scaled x.gguf:1.5 -fa on"),
]

ADV_MOE = [
    ("--n-cpu-moe", "ncmoe", "MoE 放 CPU 層數 (-ncmoe)",
     "把前 N 層的專家（MoE）權重留在 CPU，其餘照 -ngl 上 GPU。\n"
     "VRAM 不夠時用來「少放幾層專家上卡」換取不爆顯存；調太大生成會明顯變慢。\n"
     "只在 MoE 模型有意義，N 的上限就是該模型自己的層數。"),
    ("--n-cpu-ffn", "ncffn", "密集 FFN 放 CPU 層數 (-ncffn)",
     "把前 N 層的「密集」FFN 權重留在 CPU。\n"
     "這是給非 MoE（dense）模型用的；MoE 模型的專家權重請用上面的 -ncmoe。"),
]

SETTINGS_FILE = os.path.join(BASE, "ui-settings.json")
PRESETS_FILE = os.path.join(BASE, "presets.json")   # 模板（具名參數組合）＋模型綁定


def load_presets():
    """讀回模板檔。回傳 {"templates": {名: {...}}, "bind": {模型路徑: 名}}"""
    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            d.setdefault("templates", {})
            d.setdefault("bind", {})
            d.setdefault("default", "")
            d.setdefault("runtime_of", {})
            if isinstance(d["templates"], dict) and isinstance(d["bind"], dict):
                if not isinstance(d["runtime_of"], dict):
                    d["runtime_of"] = {}
                return d
    except Exception:
        pass
    return {"templates": {}, "bind": {}, "default": "", "runtime_of": {}}


def save_presets(d):
    try:
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_settings():
    """讀回上次 UI 設定（檔案不存在或壞掉就回空 dict）"""
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_settings(d):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


proc = None  # type: subprocess.Popen | None

# 本 UI 啟動過的 server PID，程式結束時保底清理（只用 exact PID，絕不用 /IM 影像名）
_SPAWNED = set()


def _cleanup_spawned():
    for pid in list(_SPAWNED):
        # 走 _run（有 errors="replace"），避免中文 taskkill 輸出造成 UnicodeDecodeError
        _run(["taskkill", "/F", "/PID", str(pid)], timeout=15)
    _SPAWNED.clear()


atexit.register(_cleanup_spawned)


def _run(cmd, timeout=20):
    """執行外部指令並回傳合併輸出（失敗回空字串）。只用於唯讀查詢與 exact-PID 終止。"""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def pids_listening_on(port):
    """回傳正在 LISTEN 指定 TCP 埠的 PID 集合（netstat -ano，唯讀）"""
    pids = set()
    for line in _run(["netstat", "-ano", "-p", "TCP"]).splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING":
            if parts[1].rsplit(":", 1)[-1] == str(port) and parts[4].isdigit():
                pids.add(int(parts[4]))
    return pids


def find_llama_servers():
    """回傳 {pid: 命令列}，列出系統上所有 llama-server.exe（唯讀）"""
    out = {}
    ps = ("Get-CimInstance Win32_Process -Filter \"name='llama-server.exe'\" | "
          "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    raw = _run(["powershell", "-NoProfile", "-Command", ps], timeout=30).strip()
    if not raw:
        return out
    try:
        data = json.loads(raw)
    except Exception:
        return out
    if isinstance(data, dict):
        data = [data]
    for d in data:
        try:
            out[int(d.get("ProcessId"))] = d.get("CommandLine") or ""
        except Exception:
            pass
    return out


def kill_pid(pid, log=None):
    """只依 exact PID 終止（Process Safety Rule：絕不使用 taskkill /IM <名稱>）"""
    out = _run(["taskkill", "/F", "/PID", str(pid)])
    ok = ("成功" in out) or ("SUCCESS" in out.upper())
    if log:
        log(f"[清理] taskkill /F /PID {pid} -> {'OK' if ok else out.strip()[:120]}\n")
    return ok


def wait_port_free(port, timeout=25, log=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not port_open(port):
            return True
        time.sleep(0.8)
    if log:
        log(f"[清理] 警告：埠 {port} 在 {timeout}s 內仍未釋放\n")
    return False


# 全形 -> 半形對照（中文 IME 在中文模式下打 "." 會送全形 "。"）
_FW = {chr(0xFF10 + i): chr(48 + i) for i in range(10)}
_FW.update({"。": ".", "．": ".", "｡": ".", "，": ",", "－": "-", "—": "-", "–": "-",
            "＋": "+", "　": " ", "：": ":", "％": "%", "／": "/"})


def attach_numeric(var):
    """讓數字欄位吃全形輸入：全形數字/標點自動轉半形。
    解決中文 IME 下無法輸入小數點（會被送成 "。"）的問題。"""
    st = {"busy": False}

    def _cb(*_a):
        if st["busy"]:
            return
        s = var.get()
        t = "".join(_FW.get(c, c) for c in s)
        if t != s:
            st["busy"] = True
            var.set(t)
            st["busy"] = False

    var.trace_add("write", _cb)


# 中文輸入法常把 "." 這種標點整個吃掉：Tk 只收到「有 keysym、char 為空」的事件，
# 但貼上是走 <<Paste>>，繞過鍵盤 —— 這正是「能複製貼上、鍵盤打不出來」的真因。
# 這裡依 keysym 手動補回字元。（全形字元那條路仍由 attach_numeric 轉換）
_NUM_KEY = {"period": ".", "KP_Decimal": ".", "comma": ",", "minus": "-",
            "KP_Subtract": "-", "KP_Add": "+", "KP_Insert": "0"}
# Windows 虛擬鍵碼退路（Tk 在 Windows 的 event.keycode 就是 VK 碼）：
# 110=數字鍵盤 . ｜190=主鍵盤 . ｜188=, ｜189/- ｜109=數字鍵盤 - ｜107=數字鍵盤 + ｜96=數字鍵盤 0
_NUM_KEYCODE = {110: ".", 190: ".", 188: ",", 189: "-", 109: "-", 107: "+", 96: "0"}


def numeric_key_handler(e):
    """IME 吞掉按鍵（char 為空）時，依 keysym 手動插入對應字元。"""
    if e.char:
        # NumLock 關閉時，Windows 會把數字鍵盤的「.」送成 keysym=Delete + char='.'，
        # Tk 會把它當成 Delete 鍵、把游標後的字刪掉（＝小數點打不出來又吃掉字）。
        # 這裡改成直接插入該標點，並回傳 break 擋掉 Tk 的刪除行為。
        if e.keysym == "Delete" and e.char in (".", ",", "-", "+"):
            try:
                e.widget.insert("insert", e.char)
                return "break"
            except Exception:
                return None
        return None            # 正常字元：交給 Tk 預設處理 + attach_numeric 轉全形
    c = _NUM_KEY.get(e.keysym) or _NUM_KEYCODE.get(getattr(e, "keycode", None))
    if not c:
        return None
    try:
        e.widget.insert("insert", c)
    except Exception:
        return None
    return "break"


def bind_numeric_keys(w):
    w.bind("<KeyPress>", numeric_key_handler, add="+")
    return w


def bind_numeric_tree(widget):
    """遞迴把整個視窗內所有輸入框都接上 IME 補救。"""
    for ch in widget.winfo_children():
        if isinstance(ch, (ttk.Entry, ttk.Combobox)):
            bind_numeric_keys(ch)
        bind_numeric_tree(ch)


class Tip:
    """滑鼠移到控件上時顯示一行簡短說明"""

    def __init__(self, widget, text):
        self.w, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _=None):
        if self.tip:
            return
        x = self.w.winfo_rootx() + 16
        y = self.w.winfo_rooty() + self.w.winfo_height() + 4
        self.tip = tk.Toplevel(self.w)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, background="#ffffe0", relief="solid",
                 borderwidth=1, justify="left", padx=6, pady=3).pack()

    def _hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def port_open(port):
    s = socket.socket()
    s.settimeout(0.4)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def http_get(path, timeout=3, port=None):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port or PORT}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return {"_err": str(e)}


def http_post(path, obj, timeout=5, port=None):
    """POST JSON（給 /lora-adapters 動態調 LoRA 強度用，繞開 Windows 路徑冒號問題）。"""
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port or PORT}{path}",
            data=json.dumps(obj).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return {"_err": str(e)}


class App:
    def __init__(self, root):
        self.root = root
        root.title(f"llama-server 控制台 — {BASE}")
        # 視窗自適應螢幕：小螢幕（工作區小於約 1000x700）寫死 900x700 會把日誌擠出畫面
        _sw, _sh = root.winfo_screenwidth(), root.winfo_screenheight()
        _w = max(760, min(1000, _sw - 120))
        _h = max(460, min(800, _sh - 110))
        root.geometry(f"{_w}x{_h}")
        if root.winfo_x() <= 1 and root.winfo_y() <= 1:      # 沒被指定位置才置中
            root.geometry(f"+{max(0, (_sw - _w) // 2)}+{max(0, (_sh - _h) // 3)}")
        root.minsize(720, 440)

        # --- 版面骨架：上半「設定區」可上下滾動；下半「日誌」固定，不被設定長度擠壓 ---
        pane = ttk.PanedWindow(root, orient="vertical")
        pane.pack(fill="both", expand=True)
        _top = ttk.Frame(pane)
        pane.add(_top, weight=3)
        self._canvas = tk.Canvas(_top, highlightthickness=0, height=240)
        self._vsb = ttk.Scrollbar(_top, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vsb.set)
        self._vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(self._canvas)
        self._inner = inner
        self._win = self._canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: self._sync_scroll())
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        # Jinja 聊天模板開關（widget 建在下方「啟動設定」；先建變數讓模型狀態列能參考）
        self.jinja = tk.BooleanVar(value=True)

        # --- 模型選擇（主模型 / 草稿模型 DF）---
        mdf = ttk.LabelFrame(inner, text="模型（主模型 / 草稿模型）")
        mdf.pack(fill="x", padx=10, pady=(8, 0))

        row1 = ttk.Frame(mdf)
        row1.pack(fill="x")
        self._models = build_model_list()                # [(顯示, 路徑, 來源, 附加)]
        self._model_map = {d: p for d, p, _src, _e in self._models}
        self.model = tk.StringVar(value=self._models[0][0] if self._models else "")
        ttk.Label(row1, text="主模型").pack(side="left", padx=(6, 4))
        self.cb_model = ttk.Combobox(row1, textvariable=self.model, width=46, state="readonly",
                                     values=[d for d, _p, _s, _e in self._models])
        self.cb_model.pack(side="left", padx=4, pady=6)
        Tip(self.cb_model, "要載入的主模型檔。\n自動掃描本程式所在資料夾的 *.gguf，"
                           "再加上你手動添加的模型（記在 models.json）。\n"
                           "草稿模型（DF）請用下面那一排選。\n改完要按「■ 停止」再「▶ 啟動」才生效。")
        b_add = ttk.Button(row1, text="➕ 添加模型", command=lambda: self._add_model("main"))
        b_add.pack(side="left", padx=2, pady=6)
        Tip(b_add, "手動添加任意位置的 .gguf 主模型（存進 models.json，重開仍在）。\n"
                   "檔案不在本資料夾時會問你要不要複製一份進來\n（推薦：原檔之後搬動或刪除都不影響）")
        b_ref = ttk.Button(row1, text="⟳ 重掃", command=self._refresh_models)
        b_ref.pack(side="left", padx=2, pady=6)
        Tip(b_ref, "重新掃描模型清單（往資料夾丟了新 .gguf 之後可按）")

        row2 = ttk.Frame(mdf)
        row2.pack(fill="x")
        ttk.Label(row2, text="草稿模型（DF）").pack(side="left", padx=(6, 4))
        self._drafters, self._drafter_map = build_drafter_list()
        self.drafter = tk.StringVar(value=self._drafters[0])
        self.cb_drafter = ttk.Combobox(row2, textvariable=self.drafter, width=46, state="readonly",
                                       values=self._drafters)
        self.cb_drafter.pack(side="left", padx=4, pady=(0, 6))
        Tip(self.cb_drafter, "投機解碼用的草稿模型（DF / DSpark / MTP）。\n"
                             "「自動」＝依主模型家族配對（同家族才不會崩潰）。\n"
                             "「不使用外掛 DF」＝不用外掛檔；若主模型內建 MTP，會自動用內建 MTP。\n"
                             "--spec-type 依檔名自動判斷（mtp / dflash / dspark / eagle）。\n"
                             "只在模式選「投機解碼」時才生效；選擇會自動記住（下次開啟還在）。")
        b_dadd = ttk.Button(row2, text="➕ 添加 DF", command=lambda: self._add_model("drafter"))
        b_dadd.pack(side="left", padx=2, pady=(0, 6))
        Tip(b_dadd, "手動添加草稿模型（DF）：可以是任何位置的 .gguf，會存進 models.json。\n"
                    "加好後就會出現在左邊的「草稿模型（DF）」下拉選單。")
        b_del = ttk.Button(row2, text="🗑 刪除", command=self._delete_model)
        b_del.pack(side="left", padx=2, pady=(0, 6))
        Tip(b_del, "刪除「手動添加」的項目（主模型或草稿模型）：可只從清單移除，或連檔案一起刪。\n"
                   "自動掃描到的模型不能在這裡刪，避免誤刪檔案。")
        self.lbl_model = ttk.Label(mdf, text="", foreground="#666")
        self.lbl_model.pack(anchor="w", padx=8, pady=(2, 6))

        def _resolve_drafter(mp=None):
            """回傳 (草稿路徑 or None, spec or None, 來源說明)。"""
            sel = self._drafter_map.get(self.drafter.get(), "")
            if sel == "-":
                return None, None, "已指定不使用"
            if sel:
                return sel, spec_for_drafter(sel, self._model_extra(sel)), "手動指定"
            mp = mp or self.model_path()
            dr, spec = guess_drafter(mp, self._model_extra(mp))
            return (dr, spec, "自動配對") if dr else (None, None, "自動配對：找不到")

        self._resolve_drafter = _resolve_drafter
        # 供模板方法（_tpl_bind/_tpl_unbind）在綁定後刷新模型資訊列
        self._sync_model_fn = None

        def _sync_model(*_a):
            p = self.model_path()
            if not os.path.exists(p):
                self.lbl_model.config(text="!! 找不到這個模型檔", foreground="#b00")
                return
            dr, spec, how = _resolve_drafter(p)
            gib = os.path.getsize(p) / 2 ** 30
            warn = ""
            if "gemma" in os.path.basename(p).lower() and not self.jinja.get():
                warn = "｜⚠ Gemma 需要 Jinja 聊天模板，請勾選 --jinja"
            _arch = gguf_arch(p)
            _allow = RUNTIME_ARCH.get(_arch)
            if _allow is not None:
                _sd = getattr(self, "runtime", None)
                if _sd is not None and os.path.basename(self.server_dir()).lower() not in _allow:
                    if _allow:
                        _hint = ("只有官方 runtime/ 這套執行檔載得動，請把上面的「執行檔」切過去"
                                 if len(RUNTIMES) > 1 else
                                 "官方 runtime/ 才載得動（把 runtime/ 放回本資料夾即可）")
                    else:
                        _hint = (("目前兩套執行檔都不支援" if len(RUNTIMES) > 1
                                  else "目前這套執行檔不支援") + "，按啟動會直接失敗")
                    warn += "｜⚠ 這顆模型是 " + _arch + " 架構：" + _hint
            if dr and not os.path.exists(dr):
                warn += "｜⚠ 指定的草稿模型檔不存在"
            elif dr:
                _mm = drafter_mismatch(p, dr)
                if _mm:
                    warn += "｜⚠ " + _mm
            _mode = getattr(self, "mode", None)
            if _mode is not None and _mode.get() != "dspark":
                _sel = self._drafter_map.get(self.drafter.get(), "")
                if _sel and _sel != "-":
                    warn += "｜⚠ 目前是 Normal 模式，這個 DF 不會生效（要切「投機解碼」）"
            if dr:
                txt = f"✅ {gib:.2f} GiB｜DF：{os.path.basename(dr)}（{spec}）｜{how}"
            elif self.drafter.get() == self._drafters[1]:
                if _mode is not None and _mode.get() == "dspark":
                    _nl = gguf_mtp_layers(p)
                    txt = (f"✅ {gib:.2f} GiB｜不使用外掛 DF → 主模型內建 MTP"
                           f"（nextn_predict_layers={_nl}）") if _nl > 0 else \
                          (f"✅ {gib:.2f} GiB｜不使用外掛 DF，且主模型沒有內建 MTP"
                           f"→ 純主模型解碼")
                else:
                    txt = f"✅ {gib:.2f} GiB｜不使用草稿模型 → 純主模型解碼（Normal）"
            else:
                txt = f"✅ {gib:.2f} GiB｜找不到可搭配的草稿模型 → 投機解碼會自動改用 Normal"
            self.lbl_model.config(text=txt + warn, foreground="#b70" if warn else "#2a7")

        self.model.trace_add("write", _sync_model)
        self.drafter.trace_add("write", _sync_model)
        self._sync_model_fn = _sync_model
        _sync_model()

        # --- 設定區 ---
        # B：「更多設定」開關移到整區最上面，視窗再矮也一定看得到（原本在最下面，
        #    小視窗時會落在折線以下，使用者以為「沒有這個功能」）。
        _bar = ttk.Frame(inner)
        _bar.pack(fill="x", padx=10, pady=(8, 0))
        self.more = tk.BooleanVar(value=False)
        self.btn_more = ttk.Button(_bar, text="▸ 更多設定（KV 型別／思考模式／思考預算／MoE 放 CPU 層數）",
                                   command=self._toggle_more)
        self.btn_more.pack(side="left")

        cfg = ttk.LabelFrame(inner, text="啟動設定")
        cfg.pack(fill="x", padx=10, pady=(4, 8))
        cfg.columnconfigure(1, weight=1)

        # 不常改的項目收進可折疊區塊（預設收合，讓日誌拿得到高度）
        advf = ttk.Frame(cfg)
        advf.grid(row=4, column=0, columnspan=3, sticky="ew")
        # 空間不足時：讓「說明欄」(column 2) 承擔壓縮，輸入欄位 (column 1) 保留固定寬度。
        # （若把 weight 放在 column 1，Tk 會優先把它壓成 0 寬 → 輸入框整排消失）
        advf.columnconfigure(1, weight=0, minsize=110)
        advf.columnconfigure(2, weight=1)
        self.advf = advf

        ttk.Label(cfg, text="模式").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.mode = tk.StringVar(value="dspark")
        mf = ttk.Frame(cfg)
        mf.grid(row=0, column=1, sticky="w")
        for val, lab in [("normal", "Normal（只用主模型）"),
                         ("dspark", "投機解碼（草稿模型 DF）")]:
            rb = ttk.Radiobutton(mf, text=lab, value=val, variable=self.mode)
            rb.pack(side="left", padx=4)
            Tip(rb, "只用主模型解碼，最單純、最省資源" if val == "normal"
                    else "投機解碼：DF 選「自動」＝配外掛草稿模型（同家族才配，如 gemma→mtp）；\n"
                         "DF 選「不使用外掛 DF」＝改用主模型自己內建的 MTP（需模型有 nextn_predict_layers）。\n"
                         "若兩者都沒有，會以純主模型解碼啟動並在日誌提醒")
        # 模式切換時同步更新模型狀態列（DF 在 Normal 模式下不會生效的提醒）
        self.mode.trace_add("write", lambda *_a: _sync_model())

        ttk.Label(cfg, text="Context").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.ctx = tk.StringVar(value="65536")
        cb_ctx = ttk.Combobox(cfg, textvariable=self.ctx, width=10,
                              values=["4096", "8192", "16384", "32768", "65536", "131072"])
        cb_ctx.grid(row=1, column=1, sticky="w")
        Tip(cb_ctx, "模型一次能記住的 token 數。\n4096 = 模型原生訓練長度\n≥65536 必須勾「KV 放系統 RAM」")
        ttk.Label(cfg, text="一次能記住多少 token，越大越吃記憶體",
                  foreground="#666").grid(row=1, column=2, sticky="w", padx=6)

        # GPU 層數（-ngl / -ngld）
        ttk.Label(cfg, text="-ngl 主模型層數").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.ngl = tk.StringVar(value="all")
        ttk.Combobox(cfg, textvariable=self.ngl, width=10,
                     values=["all", "auto", "999", "64", "48", "32"]).grid(row=2, column=1, sticky="w")
        ttk.Label(cfg, text="主模型放 GPU 的層數，all = 全部（最快）",
                  foreground="#666").grid(row=2, column=2, sticky="w", padx=6)

        ttk.Label(cfg, text="-ngld draft 層數").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.ngld = tk.StringVar(value="all")
        ttk.Combobox(cfg, textvariable=self.ngld, width=10,
                     values=["all", "auto", "999", "32", "16", "0"]).grid(row=3, column=1, sticky="w")
        ttk.Label(cfg, text="草稿模型放 GPU 的層數（僅 DSpark 生效）",
                  foreground="#666").grid(row=3, column=2, sticky="w", padx=6)

        # KV cache 型別（-ctk / -ctv）
        ttk.Label(advf, text="KV 型別 K / V").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        kvf = ttk.Frame(advf)
        kvf.grid(row=0, column=1, sticky="w")
        self.ctk = tk.StringVar(value="q4_0")
        self.ctv = tk.StringVar(value="q4_0")
        ttk.Combobox(kvf, textvariable=self.ctk, width=6, state="readonly",
                     values=["f16", "q8_0", "q4_0"]).pack(side="left")
        ttk.Label(kvf, text="/").pack(side="left", padx=4)
        ttk.Combobox(kvf, textvariable=self.ctv, width=6, state="readonly",
                     values=["f16", "q8_0", "q4_0"]).pack(side="left")
        ttk.Label(cfg, text="KV 快取壓縮格式：q4_0 最省、f16 最精確",
                  foreground="#666").grid(row=0, column=2, sticky="w", padx=6)

        # --- 進階取樣參數變數（本區的 MoE 欄位與「⚙ 進階取樣設定」視窗共用同一個 dict）---
        self.adv = {}
        for _r in ADV_CORE:
            self.adv[_r[1]] = tk.StringVar(value=_r[4])
        for _r in ADV_OPT:
            self.adv[_r[1]] = tk.StringVar(value="")   # 空 = 不送出該參數
        for _r in ADV_SPEC:
            self.adv[_r[1]] = tk.StringVar(value="")   # 空 = 不送出（用 llama.cpp 預設）
        for _r in ADV_MOE:
            self.adv[_r[1]] = tk.StringVar(value="")   # 空 = 不送出（全部照 -ngl 放 GPU）
        for _r in ADV_LOAD:
            self.adv[_r[1]] = tk.StringVar(value="")   # 空 = 不送出（用 llama.cpp 預設）
        for _r in ADV_LORA:
            self.adv[_r[1]] = tk.StringVar(value="")   # 空 = 不送出
        for _r in ADV_EXTRA:
            self.adv[_r[1]] = tk.StringVar(value="")   # 空 = 不送出
        # 思考強度（--reasoning-effort）：有專屬 UI，但共用 adv 字典（模板/存檔才會一起記）
        self.adv["rea_effort"] = tk.StringVar(value="")
        # LoRA 強度（scale）：非命令列旗標，改用啟動後 POST /lora-adapters 套用
        # （--lora-scaled 的 FNAME:SCALE 格式在 Windows 會與磁碟機代號 "D:" 衝突）
        self.adv["lora_scale"] = tk.StringVar(value="")

        # 思考模式（-rea auto/on/off）：本 build 官方預設就是 auto
        ttk.Label(advf, text="思考模式（-rea）").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.think = tk.StringVar(value="auto")
        _cb_rea = ttk.Combobox(advf, textvariable=self.think, width=10, state="readonly",
                               values=["auto", "on", "off"])
        _cb_rea.grid(row=1, column=1, sticky="w")
        ttk.Label(advf, text="auto＝依 chat template 自動偵測（llama.cpp 預設）\n"
                             "on＝強制思考｜off＝跳過思考直接回答（通常快很多）",
                  foreground="#666", justify="left", wraplength=430).grid(
            row=1, column=2, sticky="w", padx=6)
        Tip(_cb_rea, "旗標：-rea / --reasoning [on|off|auto]\n"
                     "本 build 官方預設是 auto（依模型的 chat template 決定要不要思考）。\n"
                     "on＝一定先思考再回答；off＝跳過思考直接回答（通常快很多）。")

        # 思考強度（--reasoning-effort）：交給 chat template 的「用力程度」
        # 可自由輸入的下拉（有些模型自訂層級名稱，所以不鎖死）
        ttk.Label(advf, text="思考強度（--reasoning-effort）").grid(
            row=2, column=0, sticky="w", padx=6, pady=4)
        _cb_eff = ttk.Combobox(advf, textvariable=self.adv["rea_effort"], width=10,
                               values=["default", "minimal", "low", "medium", "high", "xhigh", "max"])
        _cb_eff.grid(row=2, column=1, sticky="w")
        ttk.Label(advf, text="default＝沿用 chat template 自己的預設（官方預設值）\n"
                             "★ 只有 chat template 認得 reasoning_effort 的模型才有效，其他會忽略",
                  foreground="#666", justify="left", wraplength=430).grid(
            row=2, column=2, sticky="w", padx=6)
        Tip(_cb_eff, "旗標：--reasoning-effort LEVEL（無簡寫）\n"
                     "官方預設值：default（＝不指定，由模型 chat template 決定）。\n"
                     "可填 default / minimal / low / medium / high / xhigh / max，\n"
                     "也可自由輸入其他值（少數模型模板有自訂層級）。\n"
                     "留空＝完全不送這個參數。\n"
                     "★ 它跟「思考預算」不同：這一項是語意層級，預算是硬性 token 上限。")

        # 思考預算（--reasoning-budget）
        ttk.Label(advf, text="思考預算（--reasoning-budget）").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.rbud = tk.StringVar(value="-1")
        ttk.Combobox(advf, textvariable=self.rbud, width=10,
                     values=["-1", "0", "64", "128", "256", "512", "1024", "2048", "4096"]).grid(
            row=3, column=1, sticky="w")
        ttk.Label(advf, text="限制思考長度，越小回覆越快｜0＝立即結束思考（官方語意）",
                  foreground="#666").grid(row=3, column=2, sticky="w", padx=6)

        # MoE 權重放置（-ncmoe / -ncffn）：只在主模型是 MoE 時有意義
        # 用自由輸入（不用下拉）：可填範圍隨模型而異，寫死預設值會誤導
        ttk.Label(advf, text="MoE 放 CPU 層數 (-ncmoe)").grid(
            row=4, column=0, sticky="w", padx=6, pady=4)
        self.ncmoe = ttk.Entry(advf, textvariable=self.adv["ncmoe"], width=12)
        self.ncmoe.grid(row=4, column=1, sticky="w")
        ttk.Label(advf, text="前 N 層的專家權重留在 CPU（留空＝不送，全部照 -ngl 上 GPU）\n"
                             "VRAM 不夠時用來少放幾層專家上卡；填 0＝全部照 -ngl",
                  foreground="#666", justify="left", wraplength=430).grid(
            row=4, column=2, sticky="w", padx=6)
        Tip(self.ncmoe, "旗標：--n-cpu-moe N（簡寫 -ncmoe）\n"
                        "把「前 N 層」的 MoE 專家權重留在系統 RAM，其餘層照 -ngl 放 GPU。\n"
                        "用途：顯存不夠時少放幾層專家上卡，避免爆 VRAM（生成會變慢）。\n"
                        "可填 0 到主模型的層數為止（層數看模型 header 的 block_count）。\n"
                        "留空＝完全不送這個參數（llama.cpp 預設行為）。")

        ttk.Label(advf, text="密集 FFN 放 CPU 層數 (-ncffn)").grid(
            row=5, column=0, sticky="w", padx=6, pady=4)
        self.ncffn = ttk.Entry(advf, textvariable=self.adv["ncffn"], width=12)
        self.ncffn.grid(row=5, column=1, sticky="w")
        ttk.Label(advf, text="前 N 層的密集 FFN 權重留在 CPU（留空＝不送）\n"
                             "這是給非 MoE（dense）模型用的；MoE 的專家權重請用上面的 -ncmoe",
                  foreground="#666", justify="left", wraplength=430).grid(
            row=5, column=2, sticky="w", padx=6)
        Tip(self.ncffn, "旗標：--n-cpu-ffn N（簡寫 -ncffn）\n"
                        "把「前 N 層」的密集 FFN 權重留在系統 RAM。\n"
                        "官方說明：dense 模型用這個；MoE 的專家權重要用 --n-cpu-moe。")

        opts = ttk.Frame(advf)
        opts.grid(row=6, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        self.fa = tk.BooleanVar(value=True)
        self.nkvo = tk.BooleanVar(value=True)
        for _i, (_txt, _var, _desc) in enumerate([
                ("Flash-Attn（-fa on）", self.fa, "加速注意力計算並省記憶體，建議常開"),
                ("KV 放系統 RAM（-nkvo）", self.nkvo, "KV 改放主記憶體，可跑大 Context 但生成變慢"),
                ("Jinja 聊天模板（--jinja）", self.jinja,
                 "用模型內建的 Jinja 聊天模板（Gemma 等新模型需要）。本 build 預設已開；取消勾選＝送 --no-jinja")]):
            _chk = ttk.Checkbutton(opts, text=_txt, variable=_var)
            _chk.grid(row=_i, column=0, sticky="w")
            ttk.Label(opts, text=_desc, foreground="#666").grid(row=_i, column=1, sticky="w", padx=12)
            Tip(_chk, _desc)

        # 模型載入模式（-lm / -lzm）只在「進階取樣設定」視窗的「載入模式」分頁，
        # 不在此處重複（避免同一組參數兩處輸入）。

        # Jinja 開關變動時，同步更新模型狀態列的提醒
        self.jinja.trace_add("write", lambda *_a: _sync_model())

        ttk.Label(cfg, text="埠").grid(row=7, column=0, sticky="w", padx=6, pady=4)
        self.port = tk.StringVar(value=str(PORT))
        e_port = ttk.Entry(cfg, textvariable=self.port, width=10)
        e_port.grid(row=7, column=1, sticky="w")
        ttk.Label(cfg, text="本地服務埠（網頁／API：http://127.0.0.1:埠/）",
                  foreground="#666").grid(row=7, column=2, sticky="w", padx=6)
        Tip(e_port, "llama-server 監聽的埠。\n網頁按鈕與 OpenAI 相容用戶端（含各種前端）都用這個埠。\n"
                    "伺服器回報的模型名（--alias）會自動跟著主模型變；\n"
                    "用戶端就算填別的名字也能連（實測 model 欄位不影響）。")

        ttk.Label(cfg, text="執行檔").grid(row=8, column=0, sticky="w", padx=6, pady=4)
        self.runtime = tk.StringVar(value=RUNTIMES[0][0] if RUNTIMES else "")
        self.cb_runtime = ttk.Combobox(cfg, textvariable=self.runtime, width=24, state="readonly",
                                       values=[n for n, _d in RUNTIMES])
        self.cb_runtime.grid(row=8, column=1, sticky="w")
        # 換執行檔也要重算狀態列（架構支援度會跟著變）
        self.runtime.trace_add("write", lambda *_a: _sync_model())
        # 手動換執行檔 → 記住「這顆模型用這套」（下次選到它自動還原）
        self.runtime.trace_add("write", self._rt_remember)
        ttk.Label(cfg, text=(f"偵測到 {len(RUNTIMES)} 套 llama-server，切換後按「▶ 啟動」生效"
                             if len(RUNTIMES) > 1 else
                             "本資料夾的 llama-server（丟新的一份到子資料夾會自動列出）"),
                  foreground="#666").grid(row=8, column=2, sticky="w", padx=6)
        Tip(self.cb_runtime, "要用哪一套 llama-server 執行檔。\n"
                             "自動掃描本資料夾下所有含 llama-server.exe 的子資料夾。\n"
                             "不同 build 支援的模型架構不同（例：spark2_5 只有官方 runtime 能載）。")

        # 「更多設定」開關已移到整區最上方（見上方 _bar）
        self._apply_more()

        # 所有數值欄位都吃全形輸入（中文 IME 打 "." 會送成 "。"）
        for _v in (self.ctx, self.ngl, self.ngld, self.rbud, self.port, *self.adv.values()):
            attach_numeric(_v)

        # --- 記憶上次設定：載入 + 任何欄位變動自動存檔（0.8 秒去抖）---
        self._loading = True
        self._save_job = None
        self._apply_settings(load_settings())
        self._loading = False
        for _v in self._persist_vars():
            _v.trace_add("write", self._schedule_save)

        # 模板（presets.json）：載入既有模板與模型綁定
        self._presets = load_presets()
        self._tpl_busy = False
        self._tpl_note = ""
        self._rt_busy = False
        # 切換主模型 → 自動套用綁定/預設模板
        self.model.trace_add("write", self._tpl_on_model_change)

        # --- 控制區 ---
        ctl = ttk.Frame(inner)
        ctl.pack(fill="x", padx=10)
        self.btn_start = ttk.Button(ctl, text="▶ 啟動", command=self.start)
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(ctl, text="■ 停止", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        self.btn_adv = ttk.Button(ctl, text="⚙ 進階取樣設定", command=self.open_advanced)
        self.btn_adv.pack(side="left", padx=4)
        self.btn_web = ttk.Button(ctl, text="🌐 開網頁（輕量）", command=self.open_web)
        self.btn_web.pack(side="left", padx=4)
        Tip(self.btn_web, "用「獨立小視窗」開啟 llama-server 內建的網頁介面。\n"
                          "做法＝Chromium 的 --app 模式＋獨立 profile＋關擴充功能，\n"
                          "不帶分頁、網址列、擴充功能（實測比日常 Chrome 省約 2 GB）。\n"
                          "用完直接把那個視窗關掉就結束，不會留在背景。\n"
                          "找不到 Chrome/Edge 時自動退回系統預設瀏覽器。")
        Tip(self.btn_start, "啟動 llama-server（會先自動檢查埠與殘留程序）")
        Tip(self.btn_stop, "終止本 UI 啟動的 server 並釋放埠")
        Tip(self.btn_adv, "開啟參數視窗（temp / top-p / DRY / XTC…與投機解碼共 25 項）。\n"
                          "MoE 的 -ncmoe / -ncffn 不在這個視窗，放在上方「更多設定」裡。")
        self.btn_dot = ttk.Button(ctl, text="．", width=3, command=self._insert_dot)
        self.btn_dot.pack(side="left", padx=4)
        Tip(self.btn_dot, "把「.」補進你上一個點過的欄位（含進階設定視窗裡的欄位）。\n"
                          "中文輸入法把鍵盤的「.」吃掉時用這顆 —— 不會跳任何視窗。")
        self.btn_keydbg = ttk.Button(ctl, text="🔍 按鍵偵錯", command=self._toggle_keydbg)
        self.btn_keydbg.pack(side="left", padx=4)
        Tip(self.btn_keydbg, "開啟後，你按的每個鍵都會把 keysym / keycode / char 寫進日誌。\n"
                             "用來查「鍵盤 . ／數字鍵盤 . 打不出來」是 IME 吃掉還是 NumLock 問題。\n"
                             "再按一次關閉。")
        self.lbl = ttk.Label(ctl, text="狀態：未執行", foreground="#b00")
        self.lbl.pack(side="left", padx=16)

        # --- 模板列（參數組合＋模型自動綁定）---
        tplf = ttk.LabelFrame(inner, text="參數模板（每個模型一套參數，選到模型自動套用）")
        tplf.pack(fill="x", padx=10, pady=(6, 0))
        _tr = ttk.Frame(tplf)
        _tr.pack(fill="x", padx=6, pady=4)
        ttk.Label(_tr, text="模板").pack(side="left", padx=(2, 4))
        self.tpl = tk.StringVar(value="（不使用模板）")
        self.cb_tpl = ttk.Combobox(_tr, textvariable=self.tpl, width=26, state="readonly",
                                   values=["（不使用模板）"])
        self.cb_tpl.pack(side="left", padx=4)
        self.cb_tpl.bind("<<ComboboxSelected>>", self._tpl_on_pick)
        Tip(self.cb_tpl, "選一個模板 → 立刻把它的參數套到下面的啟動設定與進階取樣設定。\n"
                         "模板只存「怎麼跑」（ctx / ngl / KV 型別 / 取樣 / 投機 / 載入模式…），\n"
                         "不含埠、主模型、草稿模型本身。")
        _b = ttk.Button(_tr, text="💾 另存模板", command=self._tpl_save_as)
        _b.pack(side="left", padx=3)
        Tip(_b, "把「目前畫面上的所有參數」存成一個新的具名模板。")
        _b2 = ttk.Button(_tr, text="🔗 綁定此模型", command=self._tpl_bind)
        _b2.pack(side="left", padx=3)
        Tip(_b2, "把「目前選的主模型」綁到「目前選的模板」。\n"
                 "以後只要選到這個模型，就會自動套用該模板（仍可手動換別的）。")
        _b3 = ttk.Button(_tr, text="✂ 解除綁定", command=self._tpl_unbind)
        _b3.pack(side="left", padx=3)
        Tip(_b3, "解除目前主模型的模板綁定（之後選到它不會再自動套用）。")
        _b4 = ttk.Button(_tr, text="🗑 刪除模板", command=self._tpl_delete)
        _b4.pack(side="left", padx=3)
        Tip(_b4, "刪除目前選的模板（有綁定它的模型會一併解除）。")
        _b5 = ttk.Button(_tr, text="⭐ 設為預設模板", command=self._tpl_set_default)
        _b5.pack(side="left", padx=3)
        Tip(_b5, "把目前選的模板設為「預設模板」。\n"
                 "之後選到『沒有個別綁定』的模型時，會自動套用這個模板。\n"
                 "（個別綁定優先於預設模板）")
        self.lbl_tpl = ttk.Label(_tr, text="", foreground="#666")
        self.lbl_tpl.pack(side="left", padx=10)
        self._tpl_refresh_ui()

        # 「每次開都走這模板」：開機載入的模型也要套用。
        # （trace 是在 _apply_settings 之後才註冊，所以開機不會自動觸發 → 這裡主動跑一次）
        # 用 _loading 保護，避免套用過程連帶觸發自動存回。
        _was_loading = self._loading
        self._loading = True
        try:
            self._tpl_on_model_change(force=True)
        finally:
            self._loading = _was_loading
        self._tpl_refresh_ui()

        # --- 日誌（固定在下半部；拉中間分隔線可調大小，不會被上面的設定擠掉）---
        _bot = ttk.Frame(pane)
        pane.add(_bot, weight=1)
        logf = ttk.LabelFrame(_bot, text="日誌")
        logf.pack(fill="both", expand=True, padx=10, pady=(4, 8))
        self.log = scrolledtext.ScrolledText(logf, height=10, wrap="word")
        self.log.pack(fill="both", expand=True)
        # 設定區滾輪：綁到設定區每個子控件（Tk 滾輪事件不會冒泡到 Canvas）
        self._bind_wheel(inner)
        # A：某些位置（子控件縫隙／Canvas 空白／捲軸）根本收不到事件 → 使用者會覺得
        # 「怎麼滾都沒反應」。這裡在視窗層級統一攔截，指標在日誌或別的子視窗時才放行。
        self._canvas.bind("<MouseWheel>", self._on_wheel)
        self.root.bind_all("<MouseWheel>", self._on_wheel_any, add="+")
        self._sync_scroll()

        self.external_pid = None  # 非本 UI 啟動、但被接管的 server PID
        # 所有輸入框都接上「IME 吞按鍵」的補救（char 為空時依 keysym 補回小數點等）
        bind_numeric_tree(root)
        self._last_num = None      # 最後一個用過的數字欄位（小數點急救鈕靠它，不再跳彈窗）
        self._keydbg = False       # 按鍵偵錯開關
        self._watch_focus(self.root)
        self.refresh_status()
        self.root.after(1500, self._poll)
        self.root.after(300, self._startup_scan)
        # 關閉視窗時自動收掉自己啟動的 server，避免留下孤兒
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- 版面（設定區滾動／折疊／日誌高度） ----------
    def _on_canvas_resize(self, e):
        """Canvas 變寬時同步內層寬度；模型狀態列改吃 wraplength 換行（不裁切）。"""
        try:
            self._canvas.itemconfigure(self._win, width=e.width)
            _lbl = getattr(self, "lbl_model", None)
            if _lbl is not None:
                _lbl.config(wraplength=max(240, e.width - 40))
        except Exception:
            pass
        self._sync_scroll()

    def _sync_scroll(self):
        """重算設定區可滾範圍（折疊或內容變動後呼叫）。"""
        _c = getattr(self, "_canvas", None)
        if _c is None:
            return
        try:
            _c.configure(scrollregion=_c.bbox("all"))
        except Exception:
            pass

    def _toggle_more(self):
        # 只有「使用者按下按鈕」才把該區捲進畫面；載入設定／程式化還原時不要捲，
        # 否則一開 UI 就被推下去，看不到最上面的「主模型」（實測 yview 會變 0.09）。
        self.more.set(not self.more.get())
        self._apply_more(scroll_to=True)

    def _apply_more(self, scroll_to=False):
        """依 self.more 展開／收起「更多設定」。scroll_to=True 才把該區捲進畫面。"""
        _open = bool(self.more.get())
        self.btn_more.config(text="▾ 更多設定（KV 型別／思考模式／思考預算／MoE 放 CPU 層數）" if _open
                             else "▸ 更多設定（KV 型別／思考預算／MoE 放 CPU 層數）")
        try:
            if _open:
                self.advf.grid()
            else:
                self.advf.grid_remove()
        except Exception:
            pass
        # C：使用者展開時把「更多設定」整區捲進畫面（視窗很矮時，這區原本會落在折線下）
        if _open and scroll_to:
            self.root.update_idletasks()
            self._scroll_to_advf()
        self._sync_scroll()

    def _scroll_to_advf(self):
        """把「更多設定」區塊頂端捲到設定區可視範圍內（置於最上方）。"""
        try:
            _f = getattr(self, "advf", None)
            if _f is None or not _f.winfo_ismapped():
                return
            _c = self._canvas
            _bbox = _c.bbox("all")
            if not _bbox or _bbox[3] <= 0:
                return
            _tot = float(_bbox[3])
            if _tot <= 0:
                return
            # advf 是 cfg 的子控件，winfo_y() 只給相對父層的座標；
            # 用 root 座標相減換算成「相對 inner frame」才對。
            _top = _f.winfo_rooty() - self._inner.winfo_rooty()
            _c.yview_moveto(max(0.0, min(1.0, (_top - 4) / _tot)))
        except Exception:
            pass

    def _bind_wheel(self, w):
        """把滾輪綁到設定區所有子控件（Tk 的滾輪不會冒泡，不綁就滾不動）。"""
        try:
            w.bind("<MouseWheel>", self._on_wheel)
            for _c in w.winfo_children():
                self._bind_wheel(_c)
        except Exception:
            pass

    def _on_wheel_any(self, e):
        """視窗層級攔截：只要滾輪不是落在日誌／彈出視窗上，就讓設定區捲動。
        子控件自己已綁 _on_wheel 的情況（回傳 break）不會走到這裡，因此不會滾兩倍。"""
        try:
            _w = e.widget
            _tl = _w.winfo_toplevel()
            if _tl is not self.root:
                return None                 # 在彈出的子視窗裡 → 不干預
            if self._is_in_log(_w):
                return None                 # 日誌自己捲，不去搶
            return self._on_wheel(e)
        except Exception:
            return None

    def _is_in_log(self, w):
        """判斷某控件是不是「日誌」區的子控件（沿父鏈往上找）。"""
        try:
            _log = getattr(self, "log", None)
            while w is not None:
                if w is _log:
                    return True
                w = getattr(w, "master", None)
        except Exception:
            pass
        return False

    def _on_wheel(self, e):
        try:
            _lo, _hi = self._canvas.yview()
            if _lo <= 0.0 and _hi >= 1.0:
                return None            # 內容沒超過視窗高度 → 不搶滾輪
            self._canvas.yview_scroll(-1 if getattr(e, "delta", 0) > 0 else 1, "units")
            return "break"
        except Exception:
            return None

    # ---------- 組參數 ----------
    def _model_extra(self, path):
        """回傳這個路徑在手動清單裡的附加資料（可指定 drafter / spec）。"""
        for e in load_registry():
            if os.path.normcase(e["path"]) == os.path.normcase(path):
                return e
        return None

    def _refresh_models(self, keep=None):
        """重新掃描並更新下拉清單；keep = 想保留選取的模型路徑。"""
        cur = keep or self._model_map.get(self.model.get()) or ""
        self._models = build_model_list()
        self._model_map = {d: p for d, p, _src, _e in self._models}
        vals = [d for d, _p, _s, _e in self._models]
        self.cb_model.config(values=vals)
        hit = next((d for d in vals
                    if os.path.normcase(self._model_map[d]) == os.path.normcase(cur)), "")
        self.model.set(hit or (vals[0] if vals else ""))
        self._log(f"[模型] 清單已更新，共 {len(vals)} 個\n")

    def _refresh_drafters(self, keep=None):
        """重新掃描草稿模型（DF）清單；keep = 想保留選取的草稿模型路徑。"""
        cur = keep or self._drafter_map.get(self.drafter.get()) or ""
        self._drafters, self._drafter_map = build_drafter_list()
        self.cb_drafter.config(values=self._drafters)
        if cur == "-":
            self.drafter.set(self._drafters[1])
        elif cur:
            hit = next((d for d, v in self._drafter_map.items()
                        if v and v != "-" and os.path.normcase(v) == os.path.normcase(cur)), "")
            self.drafter.set(hit or self._drafters[0])
        else:
            self.drafter.set(self._drafters[0])
        self._log(f"[模型] 草稿模型清單已更新，共 {len(self._drafters)} 項\n")

    def _add_model(self, kind="main"):
        """手動添加主模型或草稿模型（DF）：可複製到本資料夾，或只記錄原路徑到 models.json。"""
        is_df = (kind == "drafter")
        what = "草稿模型（DF）" if is_df else "主模型"
        p = filedialog.askopenfilename(
            title=f"選擇要加入的 GGUF {what}", initialdir=BASE,
            filetypes=[("GGUF 模型", "*.gguf"), ("所有檔案", "*.*")])
        if not p:
            return
        p = os.path.normpath(p)
        if not os.path.exists(p):
            messagebox.showerror("找不到檔案", p)
            return
        low = os.path.basename(p).lower()
        # 已記過分類 → 直接用記憶，不再問（「不要每次重複手動」）
        memo = self._model_extra(p)
        if memo is not None and memo.get("kind") in ("main", "drafter"):
            kind = "drafter" if memo["kind"] == "drafter" else "main"
            is_df = (kind == "drafter")
            what = "草稿模型（DF）" if is_df else "主模型"
            self._log(f"[模型] 依記憶的分類加入（{what}，不再詢問）：{p}\n")
        else:
            looks_strong = any(h in low for h in DRAFT_STRONG)
            looks_mtp = "mtp" in low
            if not is_df and (looks_strong or (looks_mtp and looks_like_sidecar(p))):
                # 「是」= 改用 DF 身分加入；「否」= 照樣當主模型加入。
                # 不論選哪個都會記住，下次直接用記憶、不再問。
                _why = (f"檔名含 {('/'.join(DRAFT_STRONG))}"
                        if looks_strong else "檔名含 mtp 且檔案偏小（像外掛草稿）")
                if messagebox.askyesno(
                        "這看起來是草稿模型",
                        f"{os.path.basename(p)}\n\n{_why}，"
                        "通常是草稿/投機模型而不是主模型。\n\n"
                        "「是」= 用「草稿模型（DF）」身分加入\n"
                        "「否」= 仍要當主模型加入\n\n"
                        "（選過一次就會記住，下次不再問）"):
                    is_df = True
                    what = "草稿模型（DF）"
            elif is_df and not looks_strong and not looks_mtp:
                if not messagebox.askyesno(
                        "檔名不像草稿模型",
                        f"{os.path.basename(p)}\n\n檔名沒有 {('/'.join(DRAFT_HINT))} 這些關鍵字，"
                        "不確定是不是草稿模型。\n\n仍要當草稿模型（DF）加入嗎？\n\n"
                        "（選過一次就會記住，下次不再問）"):
                    return
        if os.path.normcase(os.path.dirname(p)) != os.path.normcase(BASE):
            ans = messagebox.askyesnocancel(
                "要複製到本資料夾嗎？",
                f"這個檔案不在本資料夾：\n{p}\n\n"
                f"「是」= 複製一份到 {BASE}（推薦：長久穩定，\n"
                "          原檔之後被移動或刪除都不影響）\n"
                "「否」= 只記錄原路徑（models.json 會記住，但原檔被移動就失效）\n"
                "「取消」= 不加入")
            if ans is None:
                return
            if ans:
                dst = os.path.join(BASE, os.path.basename(p))
                if os.path.exists(dst):
                    if not messagebox.askyesno(
                            "本資料夾已有同名檔", f"本資料夾已經有：\n{dst}\n\n直接改用這一份？"):
                        return
                    p = dst
                else:
                    try:
                        self._log(f"[模型] 複製中（{os.path.getsize(p) / 2 ** 30:.2f} GiB）…\n")
                        self.log.update_idletasks()
                        shutil.copy2(p, dst)
                        self._log(f"[模型] 複製完成：{dst}\n")
                        p = dst
                    except Exception as e:
                        messagebox.showerror("複製失敗", str(e))
                        return
        inside = os.path.normcase(os.path.dirname(p)) == os.path.normcase(BASE)
        # 一律登記「這個檔的分類」＝把使用者的選擇記住（下次不再問、不再被檔名綁架）。
        # 主模型若在本資料夾內本來就會被自動掃描到，但仍登記一次，這樣「分類記憶」才不會丟。
        reg = load_registry()
        prev = next((e for e in reg if os.path.normcase(e["path"]) == os.path.normcase(p)), None)
        new_kind = "drafter" if is_df else "main"
        if prev is None:
            ent = {"label": os.path.basename(p), "path": p, "kind": new_kind}
            if is_df:
                ent["spec"] = spec_for_drafter(p)
            reg.append(ent)
        elif prev.get("kind") != new_kind:
            prev["kind"] = new_kind           # 使用者改過分類 → 更新記憶
            if is_df:
                prev["spec"] = spec_for_drafter(p, prev)
            else:
                prev.pop("spec", None)
        if not save_registry(reg):
            messagebox.showerror("儲存失敗", f"無法寫入\n{REGISTRY}")
            return
        self._refresh_models(keep=None if is_df else p)
        self._refresh_drafters(keep=p if is_df else None)
        self._log(f"[模型] 已加入{'草稿模型（DF）' if is_df else '主模型'}：{p}\n")

    def _delete_model(self):
        """刪除個別「手動添加」的項目（主模型或草稿模型 DF；可只移除清單或連檔案刪）。"""
        cand = []
        lab = self.model.get()
        p = self._model_map.get(lab)
        src = next((s for d, _p, s, _e in self._models if d == lab), "")
        if p and src == "manual":
            cand.append(("主模型", p))
        dp = self._drafter_map.get(self.drafter.get(), "")
        if dp and dp != "-" and self._model_extra(dp) is not None:
            cand.append(("草稿模型（DF）", dp))
        if not cand:
            if p and src == "auto":
                messagebox.showinfo(
                    "這是自動掃描到的項目",
                    f"{os.path.basename(p)}\n\n"
                    f"它在 {BASE} 裡面，是自動掃描來的，\n"
                    "不是「手動添加」的項目，所以不能用這裡刪除。\n\n"
                    "要讓它從清單消失，請自行把檔案移出資料夾\n（UI 不會替你刪掉自動掃描到的檔案）。")
            else:
                messagebox.showinfo("沒有可刪的項目",
                                    "請先選一個「手動添加」的主模型或草稿模型（DF）。")
            return
        if len(cand) > 1:
            ans = messagebox.askyesnocancel(
                "要刪除哪一個？",
                "主模型與草稿模型都是「手動添加」的項目：\n\n"
                f"「是」= 刪除主模型：{os.path.basename(cand[0][1])}\n"
                f"「否」= 刪除草稿模型：{os.path.basename(cand[1][1])}\n"
                "「取消」= 什麼都不做")
            if ans is None:
                return
            kind, target = cand[0] if ans else cand[1]
        else:
            kind, target = cand[0]
        inside = os.path.normcase(os.path.dirname(os.path.abspath(target))) == os.path.normcase(BASE)
        ans = messagebox.askyesnocancel(
            f"刪除手動添加的{kind}",
            f"{os.path.basename(target)}\n\n"
            + ("⚠ 這個檔案就在程式資料夾內。\n"
               "  只從清單移除沒有用，下次掃描會再自動撿回來（看起來就像刪不掉）。\n\n" if inside else "")
            + "「是」= 從清單移除，並把檔案丟進資源回收桶（可還原）\n"
            + ("「否」= 只從清單移除（此檔在資料夾內，之後會被自動掃描加回來）\n" if inside
               else "「否」= 只從清單移除，檔案保留\n")
            + "「取消」= 什麼都不做")
        if ans is None:
            return
        reg = [e for e in load_registry()
               if os.path.normcase(e["path"]) != os.path.normcase(target)]
        if not save_registry(reg):
            messagebox.showerror("儲存失敗", f"無法寫入\n{REGISTRY}")
            return
        self._log(f"[模型] 已從清單移除：{target}\n")
        if ans:
            if not os.path.exists(target):
                self._log(f"[模型] 檔案已不存在，略過刪除：{target}\n")
            elif _to_recycle_bin(target):
                self._log(f"[模型] 已丟進資源回收桶（可還原）：{target}\n")
            else:
                messagebox.showerror(
                    "無法丟進資源回收桶",
                    f"{os.path.basename(target)}\n\n"
                    "已從清單移除，但檔案還在。\n"
                    "請自行到檔案總管刪除（UI 不做永久刪除，以免無法救回）。")
        elif inside:
            messagebox.showinfo(
                "提醒：檔案還在資料夾內",
                f"{os.path.basename(target)}\n\n"
                "你選了「只從清單移除」，但這個檔案就在程式資料夾內，\n"
                "下次重新整理時會再被自動掃描撿回來。\n\n"
                "要讓它真的從清單消失，請改用「是」（丟資源回收桶）\n"
                "或自行把檔案移出這個資料夾。")
        self._refresh_models()
        self._refresh_drafters()

    def model_path(self):
        """目前選定的主模型完整路徑（自動掃描與手動添加的都能用；舊路徑會自動修復）。"""
        return fix_path(self._model_map.get(self.model.get()) or DEFAULT_MODEL)

    def server_dir(self):
        """目前選定的 llama-server 所在目錄（預設＝自動偵測到的第一套）。"""
        rv = getattr(self, "runtime", None)
        return dict(RUNTIMES).get(rv.get() if rv is not None else "", "") or os.path.dirname(SRV)

    def server_exe(self):
        return os.path.join(self.server_dir(), "llama-server.exe")

    def build_cmd(self):
        p = self.port.get()
        mp = self.model_path()
        if not os.path.exists(mp):
            raise FileNotFoundError(f"找不到模型檔：{mp}")
        cmd = [self.server_exe(), "-m", mp, "--alias", model_alias(mp),
               "--host", "127.0.0.1", "--port", p,
               "-c", self.ctx.get(), "-np", "1"]
        # 取樣參數（在「⚙ 進階取樣設定」視窗調整；留空 = 不送出該參數）
        for _r in ADV_CORE:
            _v = self.adv[_r[1]].get().strip()
            if _v:
                cmd += [_r[0], _v]
        for _r in ADV_OPT:
            _v = self.adv[_r[1]].get().strip()
            if _v:
                cmd += [_r[0], _v]
        # MoE 權重放置（-ncmoe / -ncffn）：留空 = 不送，全部照 -ngl 上 GPU
        for _r in ADV_MOE:
            _v = self.adv[_r[1]].get().strip()
            if _v:
                cmd += [_r[0], _v]
        # 模型載入模式（-lm / -lzm）：留空 = 不送，用 llama.cpp 內建預設
        for _r in ADV_LOAD:
            _v = self.adv[_r[1]].get().strip()
            if _v:
                cmd += [_r[0], _v]
        # LoRA 適配檔（--lora）：留空 = 不送
        for _r in ADV_LORA:
            _v = self.adv[_r[1]].get().strip()
            if _v:
                cmd += [_r[0], _v]
        # 思考模式：-rea auto/on/off（本 build 預設 auto＝依 chat template 偵測）
        cmd += ["-rea", self.think.get() or "auto"]
        # 思考強度：--reasoning-effort（留空 = 不送，用官方預設 default）
        _eff = self.adv["rea_effort"].get().strip()
        if _eff:
            cmd += ["--reasoning-effort", _eff]
        # 思考預算：-1 不限 / 0 立即結束 / N>0 上限（留空 = 用 llama.cpp 預設）
        rbud = self.rbud.get().strip()
        if rbud:
            cmd += ["--reasoning-budget", rbud]
        ngl = self.ngl.get().strip()
        if ngl:
            cmd += ["-ngl", ngl]
        # Jinja 聊天模板（本 build 預設已開；取消勾選＝明確關閉）
        cmd += ["--jinja"] if self.jinja.get() else ["--no-jinja"]
        # KV cache 型別（K / V 分開可選）
        cmd += ["-ctk", self.ctk.get(), "-ctv", self.ctv.get()]
        # V cache 量化必須搭配 flash attention，否則 llama.cpp 會拒絕啟動
        fa_on = self.fa.get()
        self._forced_fa = False
        if self.ctv.get() != "f16" and not fa_on:
            fa_on = True
            self._forced_fa = True
        if fa_on:
            cmd += ["-fa", "on"]
        # ctx > 32768 在小 VRAM 顯卡上放不下 KV，強制把 KV 放到系統 RAM
        try:
            big = int(self.ctx.get()) > 32768
        except Exception:
            big = False
        if self.nkvo.get() or big:
            cmd += ["-nkvo"]

        mode = self.mode.get()
        self._forced_normal = ""
        self._spec_note = ""
        if mode == "dspark":
            mp = self.model_path()
            sel = self._drafter_map.get(self.drafter.get(), "")
            if sel == "-":                      # 明確不使用外掛 DF
                # 模式才是主開關：投機模式 + 不用 DF ＝ 使用「主模型內建 MTP」
                # （實測：只送 --spec-type draft-mtp、不送 -md，llama-server 會自己
                #  對照主模型建立 MTP 上下文；沒有內建 MTP 的模型則會載入失敗。）
                if gguf_mtp_layers(mp) > 0:
                    cmd += ["--spec-type", "draft-mtp"]
                    for _r in ADV_SPEC:
                        _v = self.adv[_r[1]].get().strip()
                        if _v:
                            cmd += [_r[0], _v]
                    self._spec_note = (
                        f"[投機] 不使用外掛 DF：改用主模型內建 MTP"
                        f"（nextn_predict_layers={gguf_mtp_layers(mp)}）\n")
                else:
                    self._spec_note = (
                        "[注意] 投機模式但未選 DF，且主模型沒有內建 MTP（nextn_predict_layers）"
                        "→ 本次以純主模型解碼啟動\n")
            else:
                if sel:                          # 手動指定的 DF
                    dr, spec = sel, spec_for_drafter(sel, self._model_extra(sel))
                else:                            # 自動配對
                    dr, spec = guess_drafter(mp, self._model_extra(mp))
                if dr and spec and os.path.exists(dr):
                    cmd += ["-md", dr, "--spec-type", spec]
                    # 投機解碼參數：只有填了才送；沒填就讓 llama.cpp 用自己的預設
                    # （n-max 官方預設是 3，不是 4——寫死會讓小 VRAM 卡變慢）
                    for _r in ADV_SPEC:
                        _v = self.adv[_r[1]].get().strip()
                        if _v:
                            cmd += [_r[0], _v]
                    ngld = self.ngld.get().strip()
                    if ngld:
                        cmd += ["-ngld", ngld]
                    self._spec_note = f"[投機] DF：{os.path.basename(dr)}（--spec-type {spec}）\n"
                elif dr:
                    raise FileNotFoundError(f"找不到指定的草稿模型檔：{dr}")
                else:
                    self._spec_note = (
                        f"[注意] {os.path.basename(mp)} 找不到可搭配的草稿模型；"
                        f"若這顆模型內建 MTP，請把 DF 選「不使用外掛 DF」\n")
        # 自訂指令：原樣附加到最後（進階視窗「自訂指令」分頁）
        # 用 shlex 解析成 argv，支援引號與空白路徑；解析失敗則整串當一個參數送出
        _extra = self.adv.get("extra_args")
        if _extra is not None:
            _ev = _extra.get().strip()
            if _ev:
                try:
                    import shlex as _shlex
                    _parts = _shlex.split(_ev, posix=False)
                    # posix=False 會保留引號，去掉成對的引號
                    _parts = [q[1:-1] if len(q) >= 2 and q[0] == q[-1] and q[0] in ("\"", "'")
                              else q for q in _parts]
                    cmd += [x for x in _parts if x]
                except Exception:
                    cmd += [_ev]
        return cmd

    def _light_profile_dir(self):
        """輕量視窗專用的獨立 Chrome profile（不與日常分頁／擴充功能共用）。"""
        d = os.path.join(BASE, "tools", "_webview_profile")
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            return ""
        return d

    def _find_browser(self):
        """找可用的 Chromium 系執行檔（Chrome 優先，其次 Edge）。回傳 exe 路徑或 ''。"""
        cands = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        for c in cands:
            if os.path.exists(c):
                return c
        return ""

    def open_web(self):
        """開 llama-server 網頁介面。
        【輕量模式】用 Chromium 的 --app 開「單一視窗」＋獨立 profile＋關擴充功能。
        實測（同一頁、1100x780）：日常 Chrome 多分頁 2606 MB → 這樣開 585 MB。
        （WebView2 反而更重：660 MB，故不採用。）
        找不到 Chromium 時退回系統預設瀏覽器。"""
        raw = self.port.get().strip()
        try:
            port = int(raw)
        except Exception:
            messagebox.showerror("設定錯誤", f"埠必須是數字：{raw}")
            return
        url = f"http://127.0.0.1:{port}/"
        if not port_open(port):
            if not messagebox.askyesno(
                    "server 似乎沒在跑",
                    f"埠 {port} 沒有回應（server 可能還沒啟動）。\n\n仍要開啟 {url} 嗎？"):
                return

        exe = self._find_browser()
        if exe:
            prof = self._light_profile_dir()
            cmd = [exe, f"--app={url}"]
            if prof:
                cmd.append("--user-data-dir=" + prof)
            cmd += ["--no-first-run", "--no-default-browser-check",
                    "--disable-extensions", "--disable-background-networking",
                    "--window-size=1100,780"]
            self._log(f"[網頁] 輕量視窗開啟（獨立 profile、無擴充功能）：{url}\n"
                      f"        {os.path.basename(exe)}\n")
            try:
                subprocess.Popen(cmd)
                return
            except Exception as e:
                self._log(f"[網頁] 輕量視窗啟動失敗（{e}），改用預設瀏覽器\n")

        self._log(f"[網頁] 開啟 {url}（系統預設瀏覽器）\n")
        try:
            webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("開啟失敗", f"{e}\n\n請手動開啟：{url}")

    # ---------- 啟停 ----------
    def start(self):
        global proc
        if proc and proc.poll() is None:
            messagebox.showinfo("已執行", "server 已在執行中。")
            return
        try:
            port = int(self.port.get())
        except Exception:
            messagebox.showerror("設定錯誤", "埠必須是數字。")
            return
        # 每次啟動都強制執行重複檢查（埠佔用 / 殘留程序）
        if not self._preflight(port):
            return
        try:
            cmd = self.build_cmd()
        except Exception as e:
            messagebox.showerror("設定錯誤", str(e))
            return
        # 手動指定的 DF 若與主模型不同家族，先警告（實測會直接崩潰）
        if self.mode.get() == "dspark":
            _dr, _sp, _how = self._resolve_drafter()
            if _dr and _how == "手動指定":
                _mm = drafter_mismatch(self.model_path(), _dr)
                if _mm and not messagebox.askyesno(
                        "草稿模型可能不相容",
                        f"{os.path.basename(_dr)}\n\n{_mm}\n\n"
                        "實測：不相容的 DF 不會友善報錯，會讓 llama-server 直接崩潰。\n\n"
                        "仍要啟動嗎？（建議改用「自動（依主模型配對）」）"):
                    return
        self._log("$ " + " ".join(cmd) + "\n")
        if getattr(self, "_forced_fa", False):
            self._log("[注意] V cache 量化需要 Flash-Attn，已自動補上 -fa on\n")
        if getattr(self, "_spec_note", ""):
            self._log(self._spec_note)
        try:
            proc = subprocess.Popen(cmd, cwd=self.server_dir(),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
        except Exception as e:
            messagebox.showerror("啟動失敗", str(e))
            return
        threading.Thread(target=self._reader, args=(proc,), daemon=True).start()
        _SPAWNED.add(proc.pid)
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl.config(text="狀態：啟動中…", foreground="#b80")

    def stop(self):
        global proc
        if proc and proc.poll() is None:
            self._log("[停止] 終止 PID %d\n" % proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except Exception:
                kill_pid(proc.pid, self._log)
            _SPAWNED.discard(proc.pid)
        elif getattr(self, "external_pid", None):
            if messagebox.askyesno(
                    "外部 server",
                    f"目前接管的是外部啟動的 server（PID {self.external_pid}）。\n要終止它嗎？"):
                kill_pid(self.external_pid, self._log)
        proc = None
        self.external_pid = None
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.lbl.config(text="狀態：已停止", foreground="#b00")

    def _reader(self, p):
        for line in p.stdout:
            self._log(line.decode("utf-8", "replace"))
        self._log("[程序結束]\n")

    def _log(self, s):
        # 日誌讀取是在背景執行緒跑；視窗若已關閉（widget 沒了）就安靜結束，不要丟例外
        try:
            self.log.insert("end", s)
            self.log.see("end")
        except Exception:
            pass

    # ---------- 狀態輪詢 ----------
    def _poll(self):
        alive = proc and proc.poll() is None
        try:
            p = int(self.port.get())
        except Exception:
            p = PORT
        if port_open(p):
            info = http_get("/props", timeout=2, port=p)
            model = os.path.basename(info.get("model_path", "") or "")[:40]
            nctx = info.get("default_generation_settings", {}).get("n_ctx", "?")
            self.lbl.config(text=f"狀態：RUNNING  port={p}  ctx={nctx}  {model}", foreground="#080")
            self._apply_lora_scale(p)
        elif alive:
            self.lbl.config(text="狀態：啟動中（載入模型）…", foreground="#b80")
        else:
            self.lbl.config(text="狀態：未執行", foreground="#b00")
        self.root.after(2000, self._poll)

    def _apply_lora_scale(self, port):
        """server 就緒後，把 LoRA 強度以 API 套用（改值會即時重套，不必重啟）。
        用 POST /lora-adapters，繞開 --lora-scaled 在 Windows 的 FNAME:SCALE 冒號衝突。"""
        try:
            lora = self.adv["lora"].get().strip()
            if not lora:
                return
            _raw = self.adv["lora_scale"].get().strip()
            try:
                scale = float(_raw) if _raw else 1.0
            except ValueError:
                return                      # 填了非數字 → 不套用（避免洗版）
            want = (lora, scale)
            if getattr(self, "_lora_applied", None) == want:
                return                      # 已套用過這個組合
            now = time.time()
            if now - getattr(self, "_lora_try_ts", 0) < 10:
                return                      # 失敗後節流，10 秒才重試一次
            self._lora_try_ts = now
            r = http_post("/lora-adapters", [{"id": 0, "scale": scale}],
                          timeout=5, port=port)
            if isinstance(r, dict) and r.get("_err"):
                self._log(f"[LoRA] 套用強度 {scale} 失敗：{r['_err']}\n")
                return
            cur = http_get("/lora-adapters", timeout=3, port=port)
            got = cur[0].get("scale") if isinstance(cur, list) and cur else "?"
            self._lora_applied = want
            self._log(f"[LoRA] 已套用強度 scale={got}（{os.path.basename(lora)}）\n")
        except Exception as e:
            self._log(f"[LoRA] 套用強度例外：{e}\n")

    def _insert_dot(self):
        """小數點急救鈕：把「.」補進最後一個用過的數字欄位（含進階設定視窗裡的欄位）。
        舊版用 root.focus_get()，焦點在另一個 Toplevel（進階設定）時會拿到 None →
        跳模態視窗並搶走焦點，等於讓進階設定不能用。改成記憶最後欄位 + 不跳任何視窗。"""
        w = getattr(self, "_last_num", None)
        if w is not None:
            try:
                if w.winfo_exists() and str(w.winfo_class()) in ("TEntry", "Entry", "TCombobox", "Text"):
                    _tl = w.winfo_toplevel()
                    try:
                        _tl.lift()
                    except Exception:
                        pass
                    w.focus_set()
                    w.insert("insert", ".")
                    return
            except Exception:
                pass
        # 沒有可插入的欄位：只寫日誌（不跳視窗、不搶焦點、不擋住進階設定）
        self._log("[小數點鈕] 找不到可插入的欄位：請先用滑鼠點一下要輸入的欄位，再按「．」\n")
        try:
            self.root.bell()
        except Exception:
            pass

    # ---------- 記住最後用過的欄位 / 按鍵偵錯 ----------
    def _remember_focus(self, w):
        try:
            if isinstance(w, (ttk.Entry, ttk.Combobox, tk.Entry, tk.Text)):
                self._last_num = w
        except Exception:
            pass

    def _watch_focus(self, widget):
        """遞迴把輸入框接上 <FocusIn>，記住最後用過的是哪一個。"""
        for ch in widget.winfo_children():
            try:
                if isinstance(ch, (ttk.Entry, ttk.Combobox, tk.Entry, tk.Text)):
                    ch.bind("<FocusIn>", lambda e: self._remember_focus(e.widget), add="+")
            except Exception:
                pass
            self._watch_focus(ch)

    def _toggle_keydbg(self):
        self._keydbg = not getattr(self, "_keydbg", False)
        try:
            self.btn_keydbg.config(text="🔍 按鍵偵錯：開" if self._keydbg else "🔍 按鍵偵錯")
        except Exception:
            pass
        if self._keydbg:
            self._log("[按鍵偵錯] 已開啟：請用滑鼠點一下要測的欄位，再按那個按不出來的鍵"
                      "（例如數字鍵盤的 .）\n"
                      "            每顆鍵都會記一行 keysym / keycode / char，"
                      "按完把這幾行貼到 issue／討論串就能對症下藥\n")
            self._enable_keydbg(self.root)
        else:
            self._log("[按鍵偵錯] 已關閉\n")

    def _enable_keydbg(self, widget):
        """遞迴把 <KeyPress> 接上偵錯記錄（每顆控件只綁一次，避免重複疊加噴多行）。"""
        try:
            if getattr(widget, "_keydbg_bound", False):
                return
            widget._keydbg_bound = True
            widget.bind("<KeyPress>", self._keydbg_log, add="+")
            for ch in widget.winfo_children():
                self._enable_keydbg(ch)
        except Exception:
            pass

    def _keydbg_log(self, e):
        if not getattr(self, "_keydbg", False):
            return None
        try:
            _note = ""
            _kc = getattr(e, "keycode", None)
            if _kc == 46 and e.char:
                _note = "  ← NumLock 關著時數字鍵盤 . 會送 Delete+char（已自動改插該字元）"
            elif _kc == 46:
                _note = "  ← NumLock 關著時數字鍵盤 . 會送 Delete"
            elif _kc == 229:
                _note = "  ← 輸入法(IME)把整顆鍵吃掉，Tk 收不到字元"
            elif not e.char:
                _note = {110: "  ← 數字鍵盤 . (VK_DECIMAL)",
                         190: "  ← 主鍵盤 . (VK_OEM_PERIOD)",
                         96: "  ← 數字鍵盤 0"}.get(_kc, "")
            self._log(f"[按鍵] keysym={e.keysym!r} keycode={e.keycode!r} "
                      f"char={e.char!r} state={e.state!r}"
                      f" 控件={e.widget.winfo_class()}{_note}\n")
        except Exception:
            pass
        return None

    # ---------- 設定記憶 / 儲存 ----------
    def _persist_vars(self):
        return [self.mode, self.ctx, self.ngl, self.ngld, self.ctk, self.ctv, self.rbud,
                self.fa, self.nkvo, self.think, self.jinja, self.port, self.runtime,
                self.model, self.drafter, self.more] + list(self.adv.values())

    def _collect_settings(self):
        d = {"mode": self.mode.get(), "ctx": self.ctx.get(), "ngl": self.ngl.get(),
             "ngld": self.ngld.get(), "ctk": self.ctk.get(), "ctv": self.ctv.get(),
             "rbud": self.rbud.get(), "fa": self.fa.get(), "nkvo": self.nkvo.get(),
             "think": self.think.get(), "jinja": self.jinja.get(),
             "port": self.port.get(), "model": self.model.get(),
             "runtime": self.runtime.get(),
             "model_path": self.model_path(),
             "drafter_path": self._drafter_map.get(self.drafter.get(), ""),
             "more": bool(self.more.get()),
             "adv": {k: v.get() for k, v in self.adv.items()}}
        return d

    # ---------- 模板（presets.json）：參數組合 + 模型綁定 ----------
    # 模板「不含」埠、主模型、草稿模型——這三項是「選哪個模型」的一部分，
    # 不是「這模型要怎麼跑」的參數，所以不進模板。
    _TPL_KEYS = ("mode", "ctx", "ngl", "ngld", "ctk", "ctv", "rbud",
                 "fa", "nkvo", "think", "jinja", "runtime")

    def _tpl_snapshot(self):
        """把「目前 UI 上的啟動設定＋進階參數」抓成一份模板內容。"""
        snap = {k: self._collect_settings()[k] for k in self._TPL_KEYS}
        snap["adv"] = {k: v.get() for k, v in self.adv.items()}
        return snap

    def _tpl_apply(self, tpl):
        """把模板內容套到 UI（只套模板有的欄位，缺的不動）。"""
        for var, key in ((self.mode, "mode"), (self.ctx, "ctx"), (self.ngl, "ngl"),
                         (self.ngld, "ngld"), (self.ctk, "ctk"), (self.ctv, "ctv"),
                         (self.rbud, "rbud")):
            if isinstance(tpl.get(key), str):
                var.set(tpl[key])
        for var, key in ((self.fa, "fa"), (self.nkvo, "nkvo"), (self.jinja, "jinja")):
            if isinstance(tpl.get(key), bool):
                var.set(tpl[key])
        # think 是字串（auto/on/off）；舊模板可能是 bool → 轉換相容
        _th = tpl.get("think")
        if isinstance(_th, bool):
            self.think.set("on" if _th else "off")
        elif isinstance(_th, str) and _th in ("auto", "on", "off"):
            self.think.set(_th)
        rt = tpl.get("runtime")
        if isinstance(rt, str) and rt in [n for n, _d in RUNTIMES]:
            self.runtime.set(rt)
        adv = tpl.get("adv")
        if isinstance(adv, dict):
            for k, var in self.adv.items():
                if isinstance(adv.get(k), str):
                    var.set(adv[k])

    _TPL_NONE = "（不使用模板）"
    _TPL_DEFAULT = "（預設模板）"

    def _tpl_current_name(self):
        """目前下拉選的模板名；選到哨兵值（不使用/預設）時回傳空字串。"""
        if not hasattr(self, "tpl"):
            return ""
        v = self.tpl.get()
        return "" if v in ("", self._TPL_NONE, self._TPL_DEFAULT) else v

    def _tpl_default_name(self):
        """目前設定的『預設模板』（沒綁定的模型會自動用它）；沒有則回傳空字串。"""
        d = self._presets.get("default", "")
        return d if isinstance(d, str) and d in self._presets["templates"] else ""

    def _tpl_store_current(self, name=None, quiet=False):
        """把目前設定存回（指定或目前的）模板。"""
        name = name or self._tpl_current_name()
        if not name or name not in self._presets["templates"]:
            return False
        self._presets["templates"][name] = self._tpl_snapshot()
        ok = save_presets(self._presets)
        if not quiet:
            self._log(f"[模板] {'已存回' if ok else '⚠ 存檔失敗：'}「{name}」\n")
        return ok

    def _tpl_refresh_ui(self):
        """把模板清單同步到下拉選單（含預設模板標記）。"""
        if not hasattr(self, "cb_tpl"):
            return
        names = [self._TPL_NONE] + sorted(self._presets["templates"])
        self.cb_tpl.config(values=names)
        cur = self._tpl_current_name()
        if cur and cur in self._presets["templates"]:
            self.tpl.set(cur)
        else:
            self.tpl.set(self._TPL_NONE)
        dft = self._tpl_default_name()
        self._tpl_status()

    def _tpl_status(self, note=""):
        """模板區右側狀態列的唯一寫入點（避免多處覆蓋彼此的文字）。"""
        if not hasattr(self, "lbl_tpl"):
            return
        dft = self._tpl_default_name()
        base = (f"預設模板：{dft}（未綁定的模型會自動套用）" if dft else "尚未設定預設模板")
        self.lbl_tpl.config(text=base + (f"　｜　{note}" if note else ""))

    def _tpl_on_pick(self, *_a):
        """使用者從下拉手動選模板 → 立即套用。"""
        if getattr(self, "_loading", False) or getattr(self, "_tpl_busy", False):
            return
        name = self.tpl.get()
        if name in ("", self._TPL_NONE):
            self._tpl_note = ""
            return
        tpl = self._presets["templates"].get(name)
        if not isinstance(tpl, dict):
            return
        self._tpl_busy = True
        self._tpl_apply(tpl)
        self._tpl_busy = False
        self._tpl_note = f"[模板] 已套用模板「{name}」\n"
        self._log(f"[模板] 已套用「{name}」（{len([k for k in (tpl.get('adv') or {}) if (tpl.get('adv') or {}).get(k)])} 項進階參數）\n")

    # ---------- 模型 ↔ 執行檔 記憶（presets.json 的 "runtime_of"）----------
    # 每顆模型各自記住「上次用哪一套 llama-server」：切模型時自動還原。
    # 優先序：模板的 runtime（有綁模板）> 這張記憶表 > 不動。
    def _rt_key(self, path):
        return os.path.normcase(path) if path else ""

    def _rt_remember(self, *_a):
        """使用者手動換執行檔 → 記到目前模型名下（下次選到它自動還原）。"""
        if getattr(self, "_loading", False) or getattr(self, "_tpl_busy", False) \
                or getattr(self, "_rt_busy", False):
            return
        mp = self.model_path()
        if not mp:
            return
        rt = self.runtime.get()
        if not rt:
            return
        table = self._presets.setdefault("runtime_of", {})
        if table.get(self._rt_key(mp)) == rt:
            return                      # 沒變就不寫檔
        table[self._rt_key(mp)] = rt
        self._presets["runtime_of"] = table
        if save_presets(self._presets):
            self._log(f"[執行檔] 已記住：{os.path.basename(mp)} → {rt}（下次選到它自動還原）\n")

    def _rt_restore(self, path):
        """把該模型記住的執行檔套回來。回傳是否有套用。"""
        rt = self._presets.get("runtime_of", {}).get(self._rt_key(path))
        if rt and rt in [n for n, _d in RUNTIMES] and self.runtime.get() != rt:
            self._rt_busy = True
            try:
                self.runtime.set(rt)
            finally:
                self._rt_busy = False
            return True
        return False

    def _tpl_on_model_change(self, *_a, force=False):
        """切換主模型 → 套用綁定模板；沒綁定則套用「預設模板」；都沒有則不動。

        force=True 用於「開機載入」：此時 trace 尚未生效，但仍要套用模板。
        開機時（force）若沒綁模板不做任何事——保留使用者上次的執行檔選擇，
        不要被「回到預設」的保底邏輯覆蓋掉。
        """
        if getattr(self, "_loading", False) and not force:
            return
        mp = self.model_path()
        if not mp:
            return
        name = self._presets["bind"].get(os.path.normcase(mp))
        if not name:
            name = self._presets["bind"].get(os.path.basename(mp))
        src = "此模型綁定"
        if not (name and name in self._presets["templates"]):
            name = self._tpl_default_name()
            src = "預設模板"
        if name and name in self._presets["templates"]:
            self._tpl_busy = True
            self.tpl.set(name)
            self._tpl_apply(self._presets["templates"][name])
            self._tpl_busy = False
            self._tpl_note = f"[模板] 已自動套用「{name}」（{src}）\n"
            self._log(f"[模板] 切換模型 → 自動套用{src}「{name}」\n")
        else:
            # 沒綁模板 → 至少把「這顆模型上次用的執行檔」還原回來，
            # 否則會沿用上一顆模型的執行檔（例如切回 A 模型卻還在用 B 模型的 runtime）。
            self._tpl_busy = True
            self.tpl.set(self._TPL_NONE)
            self._tpl_busy = False
            self._tpl_note = ""
            if force and not self._tpl_default_name():
                return          # 開機且無綁模板/預設模板 → 保留上次的執行檔
            if not self._rt_restore(mp):
                # 這顆模型從沒被記住過 → 回到預設（清單第一套），
                # 不要沿用上一顆模型的執行檔。
                # 開機（force）時不套：保留使用者上次的選擇。
                _dflt = RUNTIMES[0][0] if RUNTIMES else ""
                if (not force) and _dflt and self.runtime.get() != _dflt:
                    self._rt_busy = True
                    try:
                        self.runtime.set(_dflt)
                    finally:
                        self._rt_busy = False
                    self._log(f"[執行檔] 切換模型 → 這顆沒記錄，回到預設執行檔：{_dflt}\n")

    def _tpl_save_as(self):
        """把目前設定另存成新模板。"""
        from tkinter import simpledialog
        name = simpledialog.askstring("另存模板", "模板名稱",
                                      parent=self.root)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in self._presets["templates"] and not messagebox.askyesno(
                "覆蓋模板", f"模板「{name}」已存在，要覆蓋嗎？"):
            return
        self._presets["templates"][name] = self._tpl_snapshot()
        if save_presets(self._presets):
            self.tpl.set(name)
            self._tpl_refresh_ui()
            self.tpl.set(name)
            self._log(f"[模板] 已另存模板「{name}」\n")
        else:
            messagebox.showerror("儲存失敗", "無法寫入 presets.json")

    def _tpl_bind(self):
        """把「目前選的主模型」綁到「目前選的模板」。"""
        mp = self.model_path()
        if not mp:
            messagebox.showwarning("沒有模型", "請先在主模型欄位選一個模型。")
            return
        name = self._tpl_current_name()
        if not name:
            messagebox.showwarning("沒有模板", "請先在上面的下拉選一個模板（或用「另存模板」建立）。")
            return
        self._presets["bind"][os.path.normcase(mp)] = name
        if save_presets(self._presets):
            self._log(f"[模板] 已綁定：{os.path.basename(mp)} → 「{name}」（以後選到它會自動套用）\n")
            if self._sync_model_fn:
                self._sync_model_fn()
        else:
            messagebox.showerror("儲存失敗", "無法寫入 presets.json")

    def _tpl_unbind(self):
        """解除目前主模型的模板綁定。"""
        mp = self.model_path()
        if not mp:
            return
        hit = [k for k in self._presets["bind"] if k == os.path.normcase(mp)
               or k == os.path.basename(mp)]
        if not hit:
            messagebox.showinfo("沒有綁定", f"{os.path.basename(mp)} 目前沒有綁定任何模板。")
            return
        for k in hit:
            self._presets["bind"].pop(k, None)
        if save_presets(self._presets):
            self._log(f"[模板] 已解除綁定：{os.path.basename(mp)}\n")
            self._tpl_note = ""
            if self._sync_model_fn:
                self._sync_model_fn()

    def _tpl_set_default(self):
        """把目前選的模板設為「預設模板」（未綁定的模型會自動套用）。"""
        name = self._tpl_current_name()
        if not name:
            messagebox.showwarning("沒有模板", "請先在上面的下拉選一個模板。")
            return
        self._presets["default"] = name
        if save_presets(self._presets):
            self._log(f"[模板] 已把「{name}」設為預設模板（未綁定的模型會自動套用）\n")
            self._tpl_refresh_ui()
        else:
            messagebox.showerror("儲存失敗", "無法寫入 presets.json")

    def _tpl_delete(self):
        """刪除目前選的模板。"""
        name = self._tpl_current_name()
        if not name:
            return
        if not messagebox.askyesno("刪除模板", f"要刪除模板「{name}」嗎？\n"
                                   "（有綁定它的模型會自動改回不套用）"):
            return
        self._presets["templates"].pop(name, None)
        for k in [k for k, v in self._presets["bind"].items() if v == name]:
            self._presets["bind"].pop(k, None)
        if save_presets(self._presets):
            self._log(f"[模板] 已刪除模板「{name}」\n")
            self._tpl_refresh_ui()

    def _apply_settings(self, d):
        for var, key in ((self.mode, "mode"), (self.ctx, "ctx"), (self.ngl, "ngl"),
                         (self.ngld, "ngld"), (self.ctk, "ctk"), (self.ctv, "ctv"),
                         (self.rbud, "rbud"), (self.port, "port")):
            if isinstance(d.get(key), str):
                var.set(d[key])
        for var, key in ((self.fa, "fa"), (self.nkvo, "nkvo"), (self.jinja, "jinja")):
            if isinstance(d.get(key), bool):
                var.set(d[key])
        # think：新格式是字串（auto/on/off）。
        # 舊設定檔存的是 bool（舊語意：只有 on/off，沒有 auto 概念）→ 一律忽略，回到新預設 auto
        _th = d.get("think")
        if isinstance(_th, str) and _th in ("auto", "on", "off"):
            self.think.set(_th)
        # 執行檔：只有「現在真的存在」的 runtime 才套用（換電腦／資料夾沒了也不會卡住）
        rt = d.get("runtime")
        if isinstance(rt, str) and rt in [n for n, _dir in RUNTIMES]:
            self.runtime.set(rt)
        # 草稿模型（DF）：存的是路徑（"" = 自動、"-" = 不使用）；顯示字串含檔案大小易變，故不比對字串
        dp = d.get("drafter_path")
        if isinstance(dp, str):
            if dp in ("", "-"):
                self.drafter.set(self._drafters[0] if dp == "" else self._drafters[1])
            else:
                dp = fix_path(dp)
                hit = next((k for k, v in self._drafter_map.items()
                            if v and v != "-" and os.path.normcase(v) == os.path.normcase(dp)), "")
                if not hit:      # 上次記住的 DF 不在清單（改檔名／搬位置）→ 補一項，別讓記憶消失
                    label = ("[記憶] " + os.path.basename(dp)
                             + ("" if os.path.exists(dp) else "（檔案不在）"))
                    if label not in self._drafters:
                        self._drafters.append(label)
                        self._drafter_map[label] = dp
                        self.cb_drafter.config(values=list(self._drafters))
                    hit = label
                self.drafter.set(hit)
        adv = d.get("adv")
        if isinstance(adv, dict):
            for k, var in self.adv.items():
                if isinstance(adv.get(k), str):
                    var.set(adv[k])
        # 模型：優先用完整路徑比對（顯示字串或檔案大小變了也不會跑掉），其次比對顯示字串
        mp = d.get("model_path")
        hit = ""
        if isinstance(mp, str) and mp:
            hit = next((lab for lab, p in self._model_map.items()
                        if os.path.normcase(p) == os.path.normcase(mp)), "")
        if not hit and isinstance(d.get("model"), str) and d["model"] in self._model_map:
            hit = d["model"]
        if hit:
            self.model.set(hit)
        # 折疊狀態（沒存過就維持預設收合）
        if isinstance(d.get("more"), bool):
            self.more.set(d["more"])
            self._apply_more()

    def _schedule_save(self, *_a):
        if getattr(self, "_loading", False):
            return
        if getattr(self, "_save_job", None):
            try:
                self.root.after_cancel(self._save_job)
            except Exception:
                pass
        self._save_job = self.root.after(800, self.save_now)

    def save_now(self):
        self._save_job = None
        try:
            cur = self._collect_settings()
            ok = save_settings(cur)
            # 有選模板時 → 把目前參數「自動存回」該模板（與現有的自動記憶行為一致）
            if self._tpl_current_name() and not getattr(self, "_tpl_busy", False):
                if self._tpl_store_current(quiet=True):
                    self._tpl_status(f"已自動存回「{self._tpl_current_name()}」")
            prev = getattr(self, "_last_saved", None)
            self._last_saved = cur
            names = {"model_path": "主模型", "drafter_path": "草稿模型(DF)", "runtime": "執行檔"}
            changed = [names[k] for k in names if prev is not None and prev.get(k) != cur.get(k)]
            if ok:
                self._log("[設定] 已自動記住：" + ("、".join(changed) if changed
                                                  else "目前選擇") + "（下次開啟自動還原）\n")
            else:
                self._log("[設定] ⚠ 存檔失敗（無法寫入 ui-settings.json）\n")
            return ok
        except Exception as e:
            self._log(f"[設定] ⚠ 存檔異常：{e}\n")
            return False

    def _manual_save(self):
        ok = self.save_now()
        if ok:
            messagebox.showinfo("設定已儲存", f"已寫入：\n{SETTINGS_FILE}\n\n下次開啟 UI 會自動載入。")
        else:
            messagebox.showerror("儲存失敗", f"無法寫入：\n{SETTINGS_FILE}")

    # ---------- 重複檢查 ----------
    def _startup_scan(self):
        """UI 一開就靜默掃一次，先把重複狀況寫進日誌（不打擾使用者）。"""
        try:
            p = int(self.port.get())
        except Exception:
            p = PORT
        busy = pids_listening_on(p)
        servers = find_llama_servers()
        ours = RUNTIME_DIRS
        if os.path.exists(SETTINGS_FILE):
            self._log(f"[設定] 已載入 {os.path.basename(SETTINGS_FILE)}（欄位異動會自動存檔）\n")
        else:
            self._log(f"[設定] 尚無 {os.path.basename(SETTINGS_FILE)}，異動時會自動建立\n")
        if busy:
            self._log(f"[開機掃描] 埠 {p} 已被佔用：PID {sorted(busy)}\n")
            self._log("            按「▶ 啟動」時會詢問【接管／終止並重啟／取消】\n")
        else:
            self._log(f"[開機掃描] 埠 {p} 目前可用\n")
        others = {pid: cl for pid, cl in servers.items()
                  if pid not in busy and any(o in (cl or "").lower() for o in ours)}
        if others:
            self._log(f"[開機掃描] 另有 {len(others)} 個殘留 llama-server：{sorted(others)}\n")

    def _preflight(self, port):
        """每次按「啟動」都強制執行。回傳 True=可啟動新的；False=中止或改為接管。"""
        busy = pids_listening_on(port)
        servers = find_llama_servers()
        ours = RUNTIME_DIRS

        if busy:
            detail = []
            for p in sorted(busy):
                cl = servers.get(p)
                if cl:
                    detail.append(f"  • PID {p}（llama-server）\n    {cl[:220]}")
                else:
                    detail.append(f"  • PID {p}（非 llama-server，可能是別的程式）")
            self._log(f"[預檢] 埠 {port} 已被佔用：PID {sorted(busy)}\n")
            ans = messagebox.askyesnocancel(
                "埠已被佔用",
                f"連接埠 {port} 已被佔用：\n\n" + "\n".join(detail) + "\n\n"
                "【是】接管它（不重新啟動，直接連線使用）\n"
                "【否】先終止它，再啟動新的\n"
                "【取消】中止本次啟動")
            if ans is None:
                self._log("[預檢] 使用者取消啟動\n")
                return False
            if ans:
                self.external_pid = sorted(busy)[0]
                self._log(f"[預檢] 接管既有 server（PID {self.external_pid}），不重新啟動\n")
                self.lbl.config(text=f"狀態：外部 server  PID {self.external_pid}", foreground="#080")
                return False
            self.external_pid = None
            for p in sorted(busy):
                kill_pid(p, self._log)
            wait_port_free(port, log=self._log)

        # 殘留：來自本專案目錄、但不在本埠的 llama-server
        leftovers = [(p, c) for p, c in servers.items()
                     if p not in busy and any(o in (c or "").lower() for o in ours)]
        if leftovers:
            lines = "\n".join(f"  • PID {p}" for p, _ in leftovers)
            self._log(f"[預檢] 發現 {len(leftovers)} 個殘留 llama-server：{sorted(p for p, _ in leftovers)}\n")
            ans = messagebox.askyesnocancel(
                "發現殘留程序",
                f"偵測到 {len(leftovers)} 個殘留的 llama-server（來自本專案目錄）：\n\n"
                f"{lines}\n\n"
                "【是】全部終止後再啟動\n"
                "【否】不管它，照樣啟動\n"
                "【取消】中止本次啟動")
            if ans is None:
                self._log("[預檢] 使用者取消啟動\n")
                return False
            if ans:
                for p, _ in leftovers:
                    kill_pid(p, self._log)
        return True

    def on_close(self):
        """關閉視窗＝自動關掉 server（防孤兒），並把最後的設定存檔。"""
        global proc
        try:
            self.save_now()
        except Exception:
            pass
        if proc and proc.poll() is None:
            self._log("[關閉] 自動終止本 UI 啟動的 server PID %d\n" % proc.pid)
            try:
                proc.terminate()
                proc.wait(timeout=6)
            except Exception:
                pass
            if proc.poll() is None:
                kill_pid(proc.pid, self._log)
            _SPAWNED.discard(proc.pid)
            proc = None
        elif getattr(self, "external_pid", None):
            if messagebox.askyesno(
                    "外部 server",
                    f"目前接管的是外部啟動的 server（PID {self.external_pid}）。\n要一起關閉嗎？"):
                kill_pid(self.external_pid, self._log)
            self.external_pid = None
        self.root.destroy()

    # ---------- 進階取樣設定視窗 ----------
    def open_advanced(self):
        if getattr(self, "_adv_win", None) is not None and self._adv_win.winfo_exists():
            self._adv_win.deiconify()
            self._adv_win.lift()
            self._adv_win.focus_force()
            return
        w = tk.Toplevel(self.root)
        self._adv_win = w
        w.title("進階取樣設定")
        # 實測內容只需約 897x302；原本寫死 1150x600，超過一半是空白。
        # 改成貼合需求，並置中於主視窗（小螢幕也不會開到畫面外）。
        _ww, _wh = 920, 380
        w.minsize(760, 300)
        try:
            self.root.update_idletasks()
            _x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - _ww) // 2)
            _y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - _wh) // 2)
            _sw, _sh = w.winfo_screenwidth(), w.winfo_screenheight()
            _x = min(max(0, _x), max(0, _sw - _ww))
            _y = min(max(0, _y), max(0, _sh - _wh))
            w.geometry(f"{_ww}x{_wh}+{_x}+{_y}")
        except Exception:
            w.geometry(f"{_ww}x{_wh}")
        w.bind("<Escape>", lambda e: w.destroy())   # ESC 直接關閉
        ttk.Label(w, text="留空 = 不送出該參數（改用 llama.cpp 內建預設）｜改完按主視窗「▶ 啟動」生效",
                  foreground="#666").pack(anchor="w", padx=10, pady=(8, 2))

        nb = ttk.Notebook(w)
        # 不用 expand：讓高度貼合內容，否則最矮的分頁（基本 131px）底下會出現大片空白
        nb.pack(fill="x", padx=10, pady=6)

        def fill(tab, rows, note=None):
            # 欄位總寬 = 標籤 + Entry + 說明；視窗 920 → 分頁可用約 880。
            # 舊的 wraplength 700/720 會讓「說明欄」被擠出右緣而截斷（實測 llama.cpp 被切成 llama.cp）。
            tab.columnconfigure(2, weight=1)
            r0 = 0
            if note:
                ttk.Label(tab, text=note, foreground="#666", wraplength=560, justify="left").grid(
                    row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(8, 4))
                r0 = 1
            for i, r in enumerate(rows):
                _flag, _key, _lab, _hint = r[0], r[1], r[2], r[3]
                ttk.Label(tab, text=_lab).grid(row=r0 + i, column=0, sticky="w", padx=6, pady=3)
                ttk.Entry(tab, textvariable=self.adv[_key], width=16).grid(
                    row=r0 + i, column=1, sticky="w", padx=6)
                ttk.Label(tab, text=f"{_flag}    {_hint}", foreground="#666",
                          wraplength=520, justify="left").grid(
                    row=r0 + i, column=2, sticky="w", padx=6)

        t1 = ttk.Frame(nb); nb.add(t1, text="基本")
        fill(t1, ADV_CORE, "這 4 項一定會送出（目前 UI 的生成風格就是靠它們）。")
        t2 = ttk.Frame(nb); nb.add(t2, text="重複懲罰")
        fill(t2, [r for r in ADV_OPT if r[1] in ("rpen", "rlastn", "ppen", "fpen")])
        t3 = ttk.Frame(nb); nb.add(t3, text="DRY")
        fill(t3, [r for r in ADV_OPT if r[1].startswith("dry")],
             "DRY 要真的生效，需同時設定 --dry-multiplier 與 --dry-sequence-breaker。")
        t4 = ttk.Frame(nb); nb.add(t4, text="其他")
        fill(t4, [r for r in ADV_OPT if r[1] in ("xtcp", "xtct", "typ", "dtr", "dte", "mir", "mirlr", "mirent")])
        t5 = ttk.Frame(nb); nb.add(t5, text="投機解碼")
        fill(t5, ADV_SPEC,
             "只在模式選「投機解碼（草稿模型 DF）」時生效。\n"
             "留空 = 不送出該參數，改用 llama.cpp 內建預設。\n"
             "★ n-max 官方預設是 3，不是 4。實測在 VRAM 較小的顯卡上 2~3 常常比 4 快——\n"
             "  草稿太長會讓額外成本吃掉收益，硬體越受限越明顯。建議自己從 1 掃到 6 找甜蜜點。")
        t7 = ttk.Frame(nb); nb.add(t7, text="LoRA")
        _lora_note = ("LoRA adapter（.gguf）。啟動時送 --lora <路徑>；留空 = 不送。\n"
                      "★ 需搭配支援 LoRA 的執行檔；且 LoRA 是綁在「哪一顆基底模型」上，\n"
                      "  基底不符會載入報錯或完全無效（llama.cpp 只對支援的路徑套用）。")
        ttk.Label(t7, text=_lora_note, foreground="#666", wraplength=560,
                  justify="left").grid(row=0, column=0, columnspan=3, sticky="w",
                                       padx=6, pady=(8, 4))
        ttk.Label(t7, text="LoRA 適配檔路徑").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        _e_lora = ttk.Entry(t7, textvariable=self.adv["lora"], width=46)
        _e_lora.grid(row=1, column=1, sticky="w", padx=6)
        _bf = ttk.Frame(t7)
        _bf.grid(row=1, column=2, sticky="w", padx=6)

        def _pick_lora():
            f = filedialog.askopenfilename(
                title="選擇 LoRA 適配檔",
                filetypes=[("GGUF 適配檔", "*.gguf"), ("所有檔案", "*.*")],
                parent=None)
            if f:
                self.adv["lora"].set(os.path.normpath(f))

        _b_pick = ttk.Button(_bf, text="📂 瀏覽…", command=_pick_lora)
        _b_pick.pack(side="left")
        _b_clear = ttk.Button(_bf, text="✕", width=3,
                              command=lambda: self.adv["lora"].set(""))
        _b_clear.pack(side="left", padx=4)

        # 強度（scale）：啟動後以 API 套用（繞開 --lora-scaled 在 Windows 的冒號衝突）
        ttk.Label(t7, text="強度 scale").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        _e_ls = ttk.Combobox(t7, textvariable=self.adv["lora_scale"], width=10,
                             values=["", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"],
                             state="normal")
        _e_ls.grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(t7, text="留空＝1.0（llama.cpp 預設）。1＝官方建議；頑固題材可試 2（更高會開始崩壞）。\n"
                          "啟動後自動以 API POST /lora-adapters 套用。",
                  foreground="#666", justify="left", wraplength=520).grid(
            row=2, column=2, sticky="w", padx=6)
        Tip(_e_ls, "LoRA 強度 scale（不需重啟即可調整）。\n"
                   "留空 = 用 llama.cpp 預設 1.0。\n"
                   "官方建議：0＝原始模型（對照組）、1＝精確投影、2＝連頑固題材也翻、3+＝開始崩壞。\n"
                   "★ 本 UI 用啟動後 POST /lora-adapters 套用，"
                   "因為 --lora-scaled 的 FNAME:SCALE 在 Windows 會與磁碟機代號（如 C:）衝突而報錯。")
        Tip(_e_lora, "旗標：--lora FNAME\n"
                     "LoRA 適配檔的完整路徑（.gguf）。\n"
                     "要用不同強度時 llama.cpp 另有 --lora-scaled FNAME:SCALE（本 UI 暫不支援）。\n"
                     "留空＝不送這個參數。")

        t8 = ttk.Frame(nb); nb.add(t8, text="自訂指令")
        ttk.Label(t8, text=("直接附加到 llama-server 啟動指令的「最後面」。\n"
                            "留空＝不附加。格式與你在命令列打的一樣，會用空白拆成多個參數；\n"
                            "路徑含空白時用雙引號包起來，例如：\n"
                            '    --lora-scaled "C:\\models\\lora.gguf:1.5" --no-mmap'),
                  foreground="#666", wraplength=620, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(8, 4))
        ttk.Label(t8, text="自訂參數").grid(row=1, column=0, sticky="w", padx=6, pady=(3, 3))
        _e_extra = ttk.Entry(t8, textvariable=self.adv["extra_args"], width=52)
        _e_extra.grid(row=1, column=1, sticky="w", padx=6)
        _bx = ttk.Frame(t8)
        _bx.grid(row=1, column=2, sticky="w", padx=6)
        ttk.Button(_bx, text="✕", width=3,
                   command=lambda: self.adv["extra_args"].set("")).pack(side="left")
        Tip(_e_extra, "這裡填的內容會『原樣』接在整條指令最後。\n"
                      "會自動用空白拆成多個參數（跟命令列一樣）；含空白的路徑請用雙引號包住。\n"
                      "★ 這是最後手段：UI 已有的欄位優先，這裡只補 UI 沒有的旗標。\n"
                      "★ 打錯的旗標會讓 llama-server 啟動失敗（錯誤會顯示在日誌）。\n"
                      "可用主視窗的「預覽完整指令」先確認送出內容。")

        t6 = ttk.Frame(nb); nb.add(t6, text="載入模式")
        fill(t6, ADV_LOAD,
             "控制模型檔怎麼載入記憶體。留空 = 不送出，用 llama.cpp 內建預設（-lm auto / -lzm auto）。\n"
             "★ -lm mmap 搭配 -lzm on：大型 tensor 不常駐，可明顯降低 RAM 佔用（適合大模型）。\n"
             "  -lzm on 需要 mmap，所以 -lm 要選 mmap 或 auto。")

        bf = ttk.Frame(w)
        bf.pack(fill="x", padx=10, pady=(4, 2))
        b_all = ttk.Button(bf, text=f"🧹 全部清空（{len(ADV_CORE+ADV_OPT+ADV_SPEC+ADV_MOE+ADV_LOAD+ADV_LORA+ADV_EXTRA)+1} 項）",
                           command=self._clear_all)
        b_all.pack(side="left", padx=4)
        Tip(b_all, "把全部參數（含「基本」4 項、「投機解碼」、MoE 層數、載入模式）全部清空，\n"
                   "啟動時改用 llama.cpp 內建預設值\n（temp 0.80 / top-p 0.95 / top-k 40 / min-p 0.05…）")
        b_opt = ttk.Button(bf, text="只清可選 17 項", command=self._clear_optional)
        b_opt.pack(side="left", padx=4)
        Tip(b_opt, "只清可選的 17 項，保留「基本」4 項與「投機解碼」4 項\n（temp / top-p / top-k / min-p 與 spec-draft 參數）")
        ttk.Button(bf, text="預覽完整指令", command=self._preview_cmd).pack(side="left", padx=4)
        ttk.Button(bf, text="💾 立即儲存設定", command=self._manual_save).pack(side="left", padx=4)
        ttk.Button(bf, text="關閉", command=w.destroy).pack(side="right", padx=4)
        # 第二列：冷門功能，避免窄視窗時把主要按鈕擠掉
        bf2 = ttk.Frame(w)
        bf2.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(bf2, text="（欄位一改就會自動存檔）", foreground="#666").pack(side="left", padx=4)
        ttk.Button(bf2, text="．插入小數點", command=self._insert_dot).pack(side="left", padx=4)
        _bd = ttk.Button(bf2, text="🔍 按鍵偵錯", command=self._toggle_keydbg)
        _bd.pack(side="left", padx=4)
        Tip(_bd, "開啟後，你按的每個鍵都會把 keysym / keycode / char 寫進主視窗日誌。")
        ttk.Label(bf2, text="ESC 可關閉本視窗", foreground="#666").pack(side="right", padx=6)
        bind_numeric_tree(w)
        self._watch_focus(w)      # 進階視窗裡的欄位也納入「最後用過的欄位」
        self._enable_keydbg(w)    # 偵錯開著時立刻生效

        # 依「目前分頁」的實際內容調整高度：ttk.Notebook 預設永遠取「最高分頁」的高度，
        # 所以矮分頁底下會固定空一大塊（實測最多 128px）。這裡直接改 Notebook 高度來貼合。
        # 寬度同理：取「最寬分頁」的需求寬 + 邊距，避免右側出現大片空白。
        w.update_idletasks()
        _need_w = max(w.nametowidget(t).winfo_reqwidth() for t in nb.tabs())
        _win_w = max(700, min(_need_w + 70, w.winfo_screenwidth() - 100))

        def _fit_tab(_e=None):
            try:
                w.update_idletasks()
                _tab = nb.nametowidget(nb.select())
                nb.configure(height=_tab.winfo_reqheight())
                w.update_idletasks()
                _h = max(260, min(w.winfo_reqheight() + 12, w.winfo_screenheight() - 120))
                _x, _y = w.winfo_x(), w.winfo_y()
                w.geometry(f"{_win_w}x{_h}+{_x}+{_y}")
            except Exception:
                pass
        nb.bind("<<NotebookTabChanged>>", _fit_tab)
        w.after(120, _fit_tab)

    def _clear_all(self):
        """清空全部參數（含「基本」「投機解碼」「MoE」「載入模式」）→ 啟動時用 llama.cpp 內建預設。"""
        _all = ADV_CORE + ADV_OPT + ADV_SPEC + ADV_MOE + ADV_LOAD + ADV_LORA + ADV_EXTRA
        _n = len(_all)
        if not messagebox.askyesno(
                f"清空全部 {_n} 項參數",
                f"會把全部 {_n} 項（含「基本」的 temp / top-p / top-k / min-p、\n"
                "「投機解碼」、MoE 放 CPU 層數、載入模式、LoRA、自訂指令）清成空白，\n"
                "啟動時改用 llama.cpp 內建預設：\n"
                "    temp 0.80 / top-p 0.95 / top-k 40 / min-p 0.05 / n-max 3 …\n\n"
                "（欄位一改就自動存檔，所以清空後會立刻覆蓋你目前的設定）\n\n要繼續嗎？"):
            return
        n = 0
        for r in _all:
            self.adv[r[1]].set("")
            n += 1
        # 思考強度 / LoRA 強度有專屬 UI（不在清單內）→ 一起清
        for _extra in ("rea_effort", "lora_scale"):
            if _extra in self.adv:
                self.adv[_extra].set("")
                n += 1
        self._lora_applied = None        # 清空後允許重新套用
        messagebox.showinfo("已清空", f"已清空 {n} 項參數。\n"
                                      "可按「預覽完整指令」確認實際送出的參數。")

    def _clear_optional(self):
        """只清可選的 17 項，保留「基本」4 項。"""
        k = 0
        for r in ADV_OPT:
            if self.adv[r[1]].get():
                self.adv[r[1]].set("")
                k += 1
        messagebox.showinfo("已清空可選項",
                            f"清掉 {k} 項可選參數。\n"
                            "「基本」4 項（temp / top-p / top-k / min-p）保留不動，\n"
                            "它們一定會送出。要一起清請用「🧹 全部清空」。")

    def _preview_cmd(self):
        try:
            cmd = self.build_cmd()
        except Exception as e:
            messagebox.showerror("設定錯誤", str(e))
            return
        messagebox.showinfo("完整啟動指令", " ".join(cmd))

    def refresh_status(self):
        pass


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
