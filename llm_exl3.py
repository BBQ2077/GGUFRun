"""EXL3 support for GGUFRun: model discovery, compatibility checks and launch commands.

Pure standard library and free of tkinter, so gguf-ui.py and the tests share it.
"""
import json
import os
import struct
import subprocess

MODEL_DIR_NAME = "LLM-MODELS-EXL3"
_NGRAM_PREFIX = "ngram_embedding"
_CFG_CACHE = {}
_VISION_CACHE = {}


# ---------- 模型資訊 ----------
def _config(path):
    cfg = os.path.join(path, "config.json")
    try:
        st = os.stat(cfg)
    except OSError:
        return {}
    key = (os.path.normcase(os.path.abspath(path)), st.st_mtime_ns, st.st_size)
    if key not in _CFG_CACHE:
        try:
            with open(cfg, encoding="utf-8") as f:
                d = json.load(f)
            _CFG_CACHE[key] = d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            _CFG_CACHE[key] = {}
    return _CFG_CACHE[key]


def quant_config(path):
    q = _config(path).get("quantization_config")
    if not isinstance(q, dict):
        try:
            with open(os.path.join(path, "quantization_config.json"), encoding="utf-8") as f:
                q = json.load(f)
        except (OSError, ValueError):
            q = {}
    return q if isinstance(q, dict) else {}


def is_exl3(path):
    return bool(path) and os.path.isdir(path) and quant_config(path).get("quant_method") == "exl3"


def architecture(path):
    a = _config(path).get("architectures")
    return a[0] if isinstance(a, list) and a and isinstance(a[0], str) else ""


def is_draft(path):
    return "draft" in architecture(path).lower()


def has_vision(path):
    """模型資料夾有沒有視覺塔（config 的 vision_config＋影像前處理設定）。"""
    return isinstance(_config(path).get("vision_config"), dict) and \
        os.path.isfile(os.path.join(path, "preprocessor_config.json"))


def vision_bytes(path):
    """視覺塔權重位元組（safetensors 標頭裡名稱含 "visual" 的張量）；沒有視覺塔為 0。"""
    if not has_vision(path):
        return 0
    total = 0
    try:
        for fn in sorted(os.listdir(path)):
            if not fn.endswith(".safetensors") or fn.startswith(_NGRAM_PREFIX):
                continue
            fp = os.path.join(path, fn)
            st = os.stat(fp)
            key = (os.path.normcase(os.path.abspath(fp)), st.st_mtime_ns, st.st_size)
            if key not in _VISION_CACHE:
                n = 0
                with open(fp, "rb") as f:
                    size = struct.unpack("<Q", f.read(8))[0]
                    head = json.loads(f.read(size)) if 0 < size < 64 * 2 ** 20 else {}
                for name, meta in head.items():
                    if "visual" in name and isinstance(meta, dict):
                        a, b = meta.get("data_offsets", (0, 0))
                        n += b - a
                _VISION_CACHE[key] = n
            total += _VISION_CACHE[key]
    except (OSError, ValueError, struct.error):
        return 0
    return total


def weight_bytes(path):
    """(GPU 權重位元組, n-gram 查表位元組)。n-gram 表可留在硬碟／RAM，不算進顯存。"""
    gpu = ngram = 0
    try:
        for fn in os.listdir(path):
            if fn.endswith(".safetensors"):
                n = os.path.getsize(os.path.join(path, fn))
                if fn.startswith(_NGRAM_PREFIX):
                    ngram += n
                else:
                    gpu += n
    except OSError:
        pass
    return gpu, ngram


def model_root(base):
    return os.path.join(base, MODEL_DIR_NAME)


def list_models(base, draft=False):
    """[(資料夾名, 完整路徑, GPU 權重位元組)]，只收完整 EXL3 資料夾。"""
    root = model_root(base)
    out = []
    try:
        names = sorted(os.listdir(root), key=str.lower)
    except OSError:
        return out
    for n in names:
        p = os.path.join(root, n)
        if is_exl3(p) and is_draft(p) == draft:
            out.append((n, p, weight_bytes(p)[0]))
    return out


def label(name, size, draft=False):
    return f"[EXL3{' 草稿' if draft else ''}] {name}   {size / 2 ** 30:.2f} GiB"


# ---------- Runtime ----------
def find_runtime(base):
    """RUNTIMES/ 下第一套 backend=exllamav3 的 runtime 目錄；沒有就回空字串。"""
    root = os.path.join(base, "RUNTIMES")
    try:
        names = sorted(os.listdir(root), key=str.lower)
    except OSError:
        return ""
    for n in names:
        p = os.path.join(root, n)
        try:
            with open(os.path.join(p, "runtime.json"), encoding="utf-8") as f:
                ok = json.load(f).get("backend") == "exllamav3"
        except (OSError, ValueError, AttributeError):
            continue
        if ok and os.path.isfile(os.path.join(p, "python", "python.exe")):
            return p
    return ""


def capabilities(runtime):
    try:
        with open(os.path.join(runtime, "supported-architectures.json"), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _version(text):
    out = []
    for part in str(text).split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out + [0] * (3 - len(out)))


# ---------- 相容性檢查 ----------
def assess(path, runtime):
    """回傳 (等級, 說明)。等級：ok / warn / error。"""
    if not runtime:
        return "error", "找不到 EXL3 Runtime（RUNTIMES/ 下需要有標示 exllamav3 的 runtime.json）"
    if not is_exl3(path):
        return "error", "不是完整的 EXL3 模型資料夾（缺 config.json，或量化格式不是 exl3）"
    # 架構清單、量化版本、顯存估算都只是提示：不擋啟動，能不能跑由 runtime 實際載入決定
    cap = capabilities(runtime)
    arch = architecture(path)
    supported = cap.get("architectures") or []
    hints = []
    if supported and arch not in supported:
        hints.append(f"{arch or '未知'} 不在 ExLlamaV3 {cap.get('exllamav3_version', '?')} 的架構清單")
    need = quant_config(path).get("version")
    have = cap.get("exllamav3_version")
    if need and have and _version(need) > _version(have):
        hints.append(f"模型以 ExLlamaV3 {need} 量化，Runtime 是 {have}")
    gpu, ngram = weight_bytes(path)
    if not gpu:
        return "error", "資料夾內沒有 .safetensors 權重"
    vram = cap.get("gpu_total_bytes") or 16 * 2 ** 30
    vis = vision_bytes(path)
    gpu -= vis                                   # 視覺塔可不載入或放 RAM，不算進文字模型
    note = f"權重 {gpu / 2 ** 30:.2f} GiB／顯存 {vram / 2 ** 30:.1f} GiB"
    if vis:
        note += f"｜視覺塔 {vis / 2 ** 30:.2f} GiB（選 gpu 才佔顯存）"
    if ngram:
        note += f"｜n-gram 查表 {ngram / 2 ** 30:.1f} GiB 留在硬碟"
    if gpu > vram * 0.97:
        hints.append("權重可能放不進顯存")
    elif gpu > vram - 2.5 * 2 ** 30:
        hints.append("剩餘顯存少，Context 請調小")
    if hints:
        return "warn", note + "：⚠ " + "；".join(hints) + "（照常啟動，以實際載入結果為準）"
    return "ok", note


def draft_matches(main, draft):
    if not is_exl3(main) or not is_exl3(draft) or not is_draft(draft):
        return False
    a, b = _config(main), _config(draft)
    a = a.get("text_config", a)
    return (a.get("hidden_size") == b.get("hidden_size") and a.get("vocab_size") == b.get("vocab_size")
            and a.get("num_hidden_layers") == b.get("num_target_layers"))


def guess_draft(main, base):
    """依名稱前綴與張量形狀配對 EXL3 草稿（DFlash2）。"""
    prefix = os.path.basename(main).lower().split("-exl3")[0]
    for name, p, _size in list_models(base, draft=True):
        if name.lower().startswith(prefix + "-") and draft_matches(main, p):
            return p
    return None


def draft_block(draft):
    block = _config(draft).get("dflash_config", {}).get("block_size")
    return block if isinstance(block, int) and block >= 2 else None


# ---------- 啟動指令 ----------
KV_BITS = {"f16": 16, "q8_0": 8, "q4_0": 4}
RECURRENT_CACHE_GB = 4.0   # 多輪對話接續用的遞迴狀態檢查點（系統 RAM）
CHUNK_SIZE = 2048          # 讀提示詞時每次送進 GPU 的 token 數（65536＋q4 實測：512→1030、2048→1186 tok/s，+0.6 GiB）
CHUNK_SIZE_DRAFT = 1024    # 開草稿時草稿模型另佔約 1 GiB，chunk 降一級留顯存餘裕
SAMPLING = (("temp", "-temp"), ("topp", "-topp"), ("topk", "-topk"), ("minp", "-minp"),
            ("rpen", "-repp"), ("ppen", "-presp"), ("fpen", "-freqp"), ("rlastn", "-penr"))
# 選到 EXL3 模型時變灰的欄位：{欄位 key: 原因}。key 是 App 屬性名或 adv 字典的 key。
_ON_GPU = "EXL3 的權重和 KV cache 一律放在顯卡，不分層，也不放系統 RAM"
_EXL3_DRY_XTC = "ExLlamaV3 有這個取樣器，但 EXL3 server 目前沒有接上"
_EXL3_NO_SAMPLER = "ExLlamaV3 沒有這種取樣器"
_DFLASH_ONLY_MAX = "EXL3 的 DFlash2 草稿只看「草稿長度上限」"
NA_REASONS = {
    "ngl": _ON_GPU, "ngld": _ON_GPU, "nkvo": _ON_GPU, "ncmoe": _ON_GPU, "ncffn": _ON_GPU,
    "fa": "EXL3 的注意力計算由 runtime 決定，沒有這個開關",
    "jinja": "EXL3 自動使用模型資料夾自帶的 chat template",
    "rbud": "EXL3 server 不支援思考預算（硬性 token 上限）",
    "runtime": "EXL3 固定使用 RUNTIMES/exl3-gfx1201，不用 llama-server 執行檔",
    "drym": _EXL3_DRY_XTC, "dryb": _EXL3_DRY_XTC, "dryal": _EXL3_DRY_XTC, "dryln": _EXL3_DRY_XTC,
    "drybr": _EXL3_DRY_XTC, "xtcp": _EXL3_DRY_XTC, "xtct": _EXL3_DRY_XTC,
    "typ": _EXL3_NO_SAMPLER, "dtr": _EXL3_NO_SAMPLER, "dte": _EXL3_NO_SAMPLER,
    "mir": _EXL3_NO_SAMPLER, "mirlr": _EXL3_NO_SAMPLER, "mirent": _EXL3_NO_SAMPLER,
    "sdnmin": _DFLASH_ONLY_MAX, "sdpsplit": _DFLASH_ONLY_MAX, "sdpmin": _DFLASH_ONLY_MAX,
    "loadmode": "EXL3 直接從 safetensors 載入顯卡，沒有 mmap／延遲讀取模式",
    "lazymode": "EXL3 直接從 safetensors 載入顯卡，沒有 mmap／延遲讀取模式",
    "lora": "EXL3 不能套用 GGUF LoRA", "lora_scale": "EXL3 不能套用 GGUF LoRA",
    "extra_args": "這一格的內容只會送給 llama-server",
}
# 只有 EXL3 用得到的欄位：選到 GGUF 模型時變灰
GGUF_NA = {"vision": "視覺選項只對 EXL3 模型有效"}
VISION_MODES = ("off", "gpu", "ram")
# 選到 EXL3 模型時，仍可用的欄位在提示框多加的說明
EXL3_NOTES = {
    "ctx": "EXL3：須為 256 的倍數；KV cache 全部放在顯卡，Context 越大越吃顯存",
    "sdnmax": "EXL3：上限＝DFlash2 草稿的 block_size − 1（目前的草稿是 7）；留空＝用到上限，填較小的數字可縮短",
}
# 日誌「已略過」清單用（adv 字典裡的 llama.cpp 專屬欄位）
LLAMA_ONLY = tuple(k for k in NA_REASONS if k not in ("ngl", "ngld", "nkvo", "fa", "jinja", "rbud", "runtime"))


def draft_count(value, block):
    """草稿長度上限 → (實際 -ndt, 說明)。留空＝block−1；超過上限時用上限。"""
    top = block - 1
    text = str(value or "").strip()
    if not text:
        return top, ""
    try:
        n = int(text)
    except ValueError:
        raise ValueError(f"草稿長度上限必須是整數：{text}") from None
    if n < 1:
        raise ValueError("EXL3 的草稿長度上限至少 1；不要投機解碼請切到 Normal")
    if n > top:
        return top, f"[EXL3] 草稿長度上限 {n} 超過這個草稿模型的最大值 {top}，改用 {top}\n"
    return n, ""


def recurrent_cache_gb():
    """遞迴狀態檢查點快取（系統 RAM）：ExLlamaV3 預設 4 GB；RAM 少於 32 GB 時按比例縮小。"""
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("pf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                        ("vt", ctypes.c_ulonglong), ("va", ctypes.c_ulonglong), ("ve", ctypes.c_ulonglong)]
        ms = _MS()
        ms.len = ctypes.sizeof(_MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return RECURRENT_CACHE_GB
        total = ms.total / 2 ** 30
    except Exception:
        return RECURRENT_CACHE_GB
    return max(1.0, min(RECURRENT_CACHE_GB, round(total / 8, 1)))


def build_cmd(main, runtime, base, *, port, ctx, ctk, ctv, think, effort, speculative,
              drafter, sampling, draft_max="", chunk=None, recurrent_gb=None, vision="off"):
    """回傳 (argv, 說明文字)。drafter：''＝自動、'-'＝不使用、其他＝路徑。
    vision：off＝不載入視覺塔；gpu＝權重放顯卡；ram＝權重放系統 RAM，看圖時才串流上顯卡。"""
    recurrent_gb = recurrent_cache_gb() if recurrent_gb is None else float(recurrent_gb)
    level, why = assess(main, runtime)
    if level == "error":
        raise ValueError(why)
    if is_draft(main):
        raise ValueError("這是草稿模型；請在「主模型」選一般 EXL3 模型")
    try:
        ctx = int(str(ctx).strip())
    except ValueError:
        raise ValueError("Context 必須是整數") from None
    if ctx < 256 or ctx % 256:
        raise ValueError("EXL3 的 Context 必須 ≥ 256 且為 256 的倍數（例：32768）")
    if ctk not in KV_BITS or ctv not in KV_BITS:
        raise ValueError("KV 型別只支援 f16 / q8_0 / q4_0")
    cmd = [os.path.join(runtime, "python", "python.exe"), "-I", "-X", "utf8", "-u",
           os.path.join(runtime, "runtime.py"), "script", "server.py",
           "-m", main, "--alias", os.path.basename(main), "--host", "127.0.0.1", "--port", str(port),
           "-cs", str(ctx), "-rcs", f"{recurrent_gb:g}", "-chunk_size", "{chunk}",
           "--thinking", think if think in ("auto", "on", "off") else "on"]
    k, v = KV_BITS[ctk], KV_BITS[ctv]
    if (k, v) != (16, 16):
        cmd += ["-cq", str(k) if k == v else f"{k},{v}"]
    effort = (effort or "").strip()
    if effort and effort.lower() != "default":
        cmd += ["--reasoning-effort", effort]
    note = f"[EXL3] Runtime：{runtime}\n[EXL3] 檢查：{why}\n"
    vision = (vision or "off").strip().lower()
    if vision not in VISION_MODES:
        raise ValueError("視覺只能選 off／gpu／ram")
    if vision != "off":
        if has_vision(main):
            cmd += ["--vision", vision]
            note += "[EXL3] 視覺：" + ("權重放顯卡" if vision == "gpu" else
                                     "權重放系統 RAM，看圖時才串流上顯卡（幾乎不佔顯存）") + "\n"
        else:
            note += "[EXL3] 視覺：這個模型沒有視覺塔 → 只聊文字\n"
    if speculative:
        draft = None if drafter == "-" else (drafter or guess_draft(main, base))
        if draft:
            if not draft_matches(main, draft):
                note += f"[EXL3] ⚠ {os.path.basename(draft)} 看起來跟主模型不搭（照常啟動，以實際載入結果為準）\n"
            block = draft_block(draft)
            cmd += ["-dm", draft]
            if block:
                n, clamp = draft_count(draft_max, block)
                cmd += ["-ndt", str(n)]
                note += clamp + f"[EXL3] 投機解碼：{os.path.basename(draft)}，一次草稿 {n} 個 token（上限 {block - 1}）\n"
            else:
                if str(draft_max or "").strip():
                    n, _ = draft_count(draft_max, 2 ** 31)
                    cmd += ["-ndt", str(n)]
                note += f"[EXL3] 投機解碼：{os.path.basename(draft)}（沒有 dflash_config.block_size，草稿長度由 runtime 決定）\n"
        else:
            note += "[EXL3] 沒有相容的 EXL3 草稿 → 只用主模型解碼\n"
    for key, flag in SAMPLING:
        val = str(sampling.get(key, "")).strip()
        if val:
            cmd += [flag, val]
    if not chunk:
        chunk = CHUNK_SIZE_DRAFT if "-dm" in cmd else CHUNK_SIZE
    cmd[cmd.index("{chunk}")] = str(int(chunk))
    note += f"[EXL3] 多輪快取：遞迴狀態檢查點 {recurrent_gb:g} GB（系統 RAM）；chunk {int(chunk)}\n"
    return cmd, note


# ---------- 程序 ----------
def find_servers(runtime):
    """{pid: 命令列}：由這套 runtime 啟動的 EXL3 server（唯讀查詢）。"""
    if not runtime:
        return {}
    ps = ("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
          "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    try:
        raw = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                             text=True, timeout=30, errors="replace").stdout.strip()
        data = json.loads(raw) if raw else []
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}
    if isinstance(data, dict):
        data = [data]
    want = os.path.normcase(os.path.abspath(runtime)).lower()
    out = {}
    for d in data:
        cl = d.get("CommandLine") or ""
        if want in os.path.normcase(cl).lower() and "server.py" in cl:
            try:
                out[int(d.get("ProcessId"))] = cl
            except (TypeError, ValueError):
                pass
    return out
