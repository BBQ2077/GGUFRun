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
    """模型架構（general.architecture），例如 qwen3 / gemma4 / llama。"""
    a = gguf_meta(path, "general.architecture")
    return a if isinstance(a, str) else ""


# 模型架構名稱（general.architecture）→ 需要哪一套 runtime 才能載入。
# 空 tuple 表示目前已知沒有可用的 runtime。留空的架構預設為「不限制」。
RUNTIME_ARCH = {}


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
    小於 1 GiB 視為外掛小檔（含 mtp 的 9B 主模型約 5 GiB，不會被誤判）。
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
    手動登錄若有指定則優先；否則依檔名自動配對（例：dspark 家族配 dspark、gemma 配 mtp）。"""
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
    """回傳 [(顯示字串, 完整路徑, 來源, 附加資料)]；來源 = auto（D 槽掃描）/ manual（手動）。"""
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
    auto, none = "自動（依主模型配對）", "不使用（Normal）"
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

SETTINGS_FILE = os.path.join(BASE, "ui-settings.json")


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


class App:
    def __init__(self, root):
        self.root = root
        root.title(f"llama-server 控制台 — {BASE}")
        # 視窗自適應螢幕：你的工作區只有 1280x649，寫死 900x700 會把日誌擠出畫面
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
                             "也可以指定某個檔，或選「不使用」。\n"
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
                txt = f"✅ {gib:.2f} GiB｜不使用草稿模型 → 純主模型解碼（Normal）"
            else:
                txt = f"✅ {gib:.2f} GiB｜找不到可搭配的草稿模型 → 投機解碼會自動改用 Normal"
            self.lbl_model.config(text=txt + warn, foreground="#b70" if warn else "#2a7")

        self.model.trace_add("write", _sync_model)
        self.drafter.trace_add("write", _sync_model)
        _sync_model()

        # --- 設定區 ---
        cfg = ttk.LabelFrame(inner, text="啟動設定")
        cfg.pack(fill="x", padx=10, pady=8)
        cfg.columnconfigure(1, weight=1)

        # 不常改的項目收進可折疊區塊（預設收合，讓日誌拿得到高度）
        advf = ttk.Frame(cfg)
        advf.grid(row=4, column=0, columnspan=3, sticky="ew")
        advf.columnconfigure(1, weight=1)
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
                    else "自動配對草稿模型做投機解碼（同家族才會配對，如 dspark 家族→dspark、gemma→mtp）。\n"
                         "若這個模型沒有可搭配的草稿模型，會自動改用 Normal 並在日誌提醒")
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

        # 思考預算（--reasoning-budget）
        ttk.Label(advf, text="思考預算（--reasoning-budget）").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.rbud = tk.StringVar(value="-1")
        ttk.Combobox(advf, textvariable=self.rbud, width=10,
                     values=["-1", "64", "128", "256", "512", "1024", "2048", "4096"]).grid(
            row=1, column=1, sticky="w")
        ttk.Label(cfg, text="限制思考長度，越小回覆越快（實測 0 無效）",
                  foreground="#666").grid(row=1, column=2, sticky="w", padx=6)

        opts = ttk.Frame(advf)
        opts.grid(row=2, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        self.fa = tk.BooleanVar(value=True)
        self.nkvo = tk.BooleanVar(value=True)
        self.think = tk.BooleanVar(value=True)
        for _i, (_txt, _var, _desc) in enumerate([
                ("Flash-Attn（-fa on）", self.fa, "加速注意力計算並省記憶體，建議常開"),
                ("KV 放系統 RAM（-nkvo）", self.nkvo, "KV 改放主記憶體，可跑大 Context 但生成變慢"),
                ("思考模式（-rea on/off）", self.think, "開啟＝先思考再回答；關閉＝直接回答（快很多）"),
                ("Jinja 聊天模板（--jinja）", self.jinja,
                 "用模型內建的 Jinja 聊天模板（Gemma 等新模型需要）。本 build 預設已開；取消勾選＝送 --no-jinja")]):
            _chk = ttk.Checkbutton(opts, text=_txt, variable=_var)
            _chk.grid(row=_i, column=0, sticky="w")
            ttk.Label(opts, text=_desc, foreground="#666").grid(row=_i, column=1, sticky="w", padx=12)
            Tip(_chk, _desc)

        # Jinja 開關變動時，同步更新模型狀態列的提醒
        self.jinja.trace_add("write", lambda *_a: _sync_model())

        ttk.Label(cfg, text="埠").grid(row=7, column=0, sticky="w", padx=6, pady=4)
        self.port = tk.StringVar(value=str(PORT))
        e_port = ttk.Entry(cfg, textvariable=self.port, width=10)
        e_port.grid(row=7, column=1, sticky="w")
        ttk.Label(cfg, text="本地服務埠（網頁／API：http://127.0.0.1:埠/）",
                  foreground="#666").grid(row=7, column=2, sticky="w", padx=6)
        Tip(e_port, "llama-server 監聽的埠。\n網頁按鈕與自訂 API 用戶端都用這個埠。\n"
                    "伺服器回報的模型名（--alias）會自動跟著主模型變；\n"
                    "用戶端就算填別的名字也能連（實測 model 欄位不影響）。")

        ttk.Label(cfg, text="執行檔").grid(row=8, column=0, sticky="w", padx=6, pady=4)
        self.runtime = tk.StringVar(value=RUNTIMES[0][0] if RUNTIMES else "")
        self.cb_runtime = ttk.Combobox(cfg, textvariable=self.runtime, width=24, state="readonly",
                                       values=[n for n, _d in RUNTIMES])
        self.cb_runtime.grid(row=8, column=1, sticky="w")
        # 換執行檔也要重算狀態列（架構支援度會跟著變）
        self.runtime.trace_add("write", lambda *_a: _sync_model())
        ttk.Label(cfg, text=(f"偵測到 {len(RUNTIMES)} 套 llama-server，切換後按「▶ 啟動」生效"
                             if len(RUNTIMES) > 1 else
                             "本資料夾的 llama-server（丟新的一份到子資料夾會自動列出）"),
                  foreground="#666").grid(row=8, column=2, sticky="w", padx=6)
        Tip(self.cb_runtime, "要用哪一套 llama-server 執行檔。\n"
                             "自動掃描本資料夾下所有含 llama-server.exe 的子資料夾。\n"
                             "不同 build 支援的模型架構不同，舊版執行檔可能載不動較新的模型架構。")

        # 「更多設定」折疊開關（狀態寫進 ui-settings.json，下次開還原）
        self.more = tk.BooleanVar(value=False)
        self.btn_more = ttk.Button(cfg, text="▸ 更多設定（KV 型別／思考預算／開關）",
                                   command=self._toggle_more)
        self.btn_more.grid(row=9, column=0, columnspan=3, sticky="w", padx=6, pady=(2, 6))
        Tip(self.btn_more, "收起不常改的設定，下面的日誌就不會被擠小。\n展開／收合狀態會記住。")
        self._apply_more()

        # --- 進階取樣參數變數（視窗由「⚙ 進階取樣設定」按鈕開啟）---
        self.adv = {}
        for _r in ADV_CORE:
            self.adv[_r[1]] = tk.StringVar(value=_r[4])
        for _r in ADV_OPT:
            self.adv[_r[1]] = tk.StringVar(value="")   # 空 = 不送出該參數
        for _r in ADV_SPEC:
            self.adv[_r[1]] = tk.StringVar(value="")   # 空 = 不送出（用 llama.cpp 預設）

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

        # --- 控制區 ---
        ctl = ttk.Frame(inner)
        ctl.pack(fill="x", padx=10)
        self.btn_start = ttk.Button(ctl, text="▶ 啟動", command=self.start)
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(ctl, text="■ 停止", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        self.btn_adv = ttk.Button(ctl, text="⚙ 進階取樣設定", command=self.open_advanced)
        self.btn_adv.pack(side="left", padx=4)
        self.btn_web = ttk.Button(ctl, text="🌐 開網頁", command=self.open_web)
        self.btn_web.pack(side="left", padx=4)
        Tip(self.btn_web, "用瀏覽器開啟 llama-server 內建的網頁介面。\n"
                          "網址＝http://127.0.0.1:<上面「埠」欄位的值>/，改埠就跟著變。")
        Tip(self.btn_start, "啟動 llama-server（會先自動檢查埠與殘留程序）")
        Tip(self.btn_stop, "終止本 UI 啟動的 server 並釋放埠")
        Tip(self.btn_adv, "開啟參數視窗（temp / top-p / DRY / XTC…與投機解碼，共 25 項）")
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

        # --- 日誌（固定在下半部；拉中間分隔線可調大小，不會被上面的設定擠掉）---
        _bot = ttk.Frame(pane)
        pane.add(_bot, weight=1)
        logf = ttk.LabelFrame(_bot, text="日誌")
        logf.pack(fill="both", expand=True, padx=10, pady=(4, 8))
        self.log = scrolledtext.ScrolledText(logf, height=10, wrap="word")
        self.log.pack(fill="both", expand=True)
        # 設定區滾輪：綁到設定區每個子控件（Tk 滾輪事件不會冒泡到 Canvas）
        self._bind_wheel(inner)
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
        self.more.set(not self.more.get())
        self._apply_more()

    def _apply_more(self):
        """依 self.more 展開／收起「更多設定」。"""
        _open = bool(self.more.get())
        self.btn_more.config(text="▾ 更多設定（KV 型別／思考預算／開關）" if _open
                             else "▸ 更多設定（KV 型別／思考預算／開關）")
        try:
            if _open:
                self.advf.grid()
            else:
                self.advf.grid_remove()
        except Exception:
            pass
        self._sync_scroll()

    def _bind_wheel(self, w):
        """把滾輪綁到設定區所有子控件（Tk 的滾輪不會冒泡，不綁就滾不動）。"""
        try:
            w.bind("<MouseWheel>", self._on_wheel)
            for _c in w.winfo_children():
                self._bind_wheel(_c)
        except Exception:
            pass

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
        # 思考模式：-rea on/off（對應 chat template 的 enable_thinking；不預設 on 會走 auto）
        cmd += ["-rea", "on" if self.think.get() else "off"]
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
        # ctx > 32768 在 8GB VRAM 上放不下 KV，強制把 KV 放到系統 RAM
        try:
            big = int(self.ctx.get()) > 32768
        except Exception:
            big = False
        if self.nkvo.get() or big:
            cmd += ["-nkvo"]

        mode = self.mode.get()
        self._forced_normal = ""
        if mode == "dspark":
            mp = self.model_path()
            sel = self._drafter_map.get(self.drafter.get(), "")
            if sel == "-":                       # 使用者指定「不使用草稿模型」
                self._forced_normal = "[注意] 已指定「不使用草稿模型」，以 Normal 模式啟動\n"
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
                elif dr:
                    raise FileNotFoundError(f"找不到指定的草稿模型檔：{dr}")
                else:
                    self._forced_normal = (
                        f"[注意] {os.path.basename(mp)} 找不到可搭配的草稿模型，"
                        f"已自動改用 Normal 模式啟動\n")
        return cmd

    def open_web(self):
        """用預設瀏覽器開啟 llama-server 的網頁介面（網址跟隨「埠」欄位變化）。"""
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
        self._log(f"[網頁] 開啟 {url}\n")
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
        if getattr(self, "_forced_normal", ""):
            self._log(self._forced_normal)
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
        elif alive:
            self.lbl.config(text="狀態：啟動中（載入模型）…", foreground="#b80")
        else:
            self.lbl.config(text="狀態：未執行", foreground="#b00")
        self.root.after(2000, self._poll)

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
                      "按完把這幾行貼給我就能對症下藥\n")
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

    def _apply_settings(self, d):
        for var, key in ((self.mode, "mode"), (self.ctx, "ctx"), (self.ngl, "ngl"),
                         (self.ngld, "ngld"), (self.ctk, "ctk"), (self.ctv, "ctv"),
                         (self.rbud, "rbud"), (self.port, "port")):
            if isinstance(d.get(key), str):
                var.set(d[key])
        for var, key in ((self.fa, "fa"), (self.nkvo, "nkvo"), (self.think, "think"), (self.jinja, "jinja")):
            if isinstance(d.get(key), bool):
                var.set(d[key])
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
        w.geometry("1150x600")
        ttk.Label(w, text="留空 = 不送出該參數（改用 llama.cpp 內建預設）｜改完按主視窗「▶ 啟動」生效",
                  foreground="#666").pack(anchor="w", padx=10, pady=(8, 2))

        nb = ttk.Notebook(w)
        nb.pack(fill="both", expand=True, padx=10, pady=6)

        def fill(tab, rows, note=None):
            r0 = 0
            if note:
                ttk.Label(tab, text=note, foreground="#666", wraplength=720, justify="left").grid(
                    row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(8, 4))
                r0 = 1
            for i, r in enumerate(rows):
                _flag, _key, _lab, _hint = r[0], r[1], r[2], r[3]
                ttk.Label(tab, text=_lab).grid(row=r0 + i, column=0, sticky="w", padx=6, pady=3)
                ttk.Entry(tab, textvariable=self.adv[_key], width=16).grid(
                    row=r0 + i, column=1, sticky="w", padx=6)
                ttk.Label(tab, text=f"{_flag}    {_hint}", foreground="#666",
                          wraplength=700, justify="left").grid(
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

        bf = ttk.Frame(w)
        bf.pack(fill="x", padx=10, pady=(0, 10))
        b_all = ttk.Button(bf, text="🧹 全部清空（25 項）", command=self._clear_all)
        b_all.pack(side="left", padx=4)
        Tip(b_all, "把 25 項參數（含「基本」4 項與「投機解碼」4 項）全部清空，\n啟動時改用 llama.cpp 內建預設值\n"
                   "（temp 0.80 / top-p 0.95 / top-k 40 / min-p 0.05…）")
        b_opt = ttk.Button(bf, text="只清可選 17 項", command=self._clear_optional)
        b_opt.pack(side="left", padx=4)
        Tip(b_opt, "只清可選的 17 項，保留「基本」4 項與「投機解碼」4 項\n（temp / top-p / top-k / min-p 與 spec-draft 參數）")
        ttk.Button(bf, text="預覽完整指令", command=self._preview_cmd).pack(side="left", padx=4)
        ttk.Button(bf, text="💾 立即儲存設定", command=self._manual_save).pack(side="left", padx=4)
        ttk.Label(bf, text="（欄位一改就會自動存檔）", foreground="#666").pack(side="left", padx=6)
        ttk.Button(bf, text="．插入小數點", command=self._insert_dot).pack(side="left", padx=4)
        _bd = ttk.Button(bf, text="🔍 按鍵偵錯", command=self._toggle_keydbg)
        _bd.pack(side="left", padx=4)
        Tip(_bd, "開啟後，你按的每個鍵都會把 keysym / keycode / char 寫進主視窗日誌。")
        ttk.Button(bf, text="關閉", command=w.destroy).pack(side="right", padx=4)
        bind_numeric_tree(w)
        self._watch_focus(w)      # 進階視窗裡的欄位也納入「最後用過的欄位」
        self._enable_keydbg(w)    # 偵錯開著時立刻生效

    def _clear_all(self):
        """清空全部 25 項參數（含「基本」4 項與「投機解碼」4 項）→ 啟動時用 llama.cpp 內建預設。"""
        if not messagebox.askyesno(
                "清空全部 25 項參數",
                "會把全部 25 項（含「基本」的 temp / top-p / top-k / min-p\n"
                "與「投機解碼」的 spec-draft 參數）清成空白，\n"
                "啟動時改用 llama.cpp 內建預設：\n"
                "    temp 0.80 / top-p 0.95 / top-k 40 / min-p 0.05 / n-max 3 …\n\n"
                "（欄位一改就自動存檔，所以清空後會立刻覆蓋你目前的設定）\n\n要繼續嗎？"):
            return
        n = 0
        for r in ADV_CORE + ADV_OPT + ADV_SPEC:
            self.adv[r[1]].set("")
            n += 1
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
