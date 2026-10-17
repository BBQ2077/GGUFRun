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


def is_lilicorr(path):
    return bool(path) and os.path.isdir(path) and architecture(path) == "LiLiCorrDraftModel"


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
        if (is_exl3(p) or (draft and is_lilicorr(p))) and is_draft(p) == draft:
            out.append((n, p, weight_bytes(p)[0]))
    return out


def label(name, size, draft=False):
    return f"[EXL3{' 草稿' if draft else ''}] {name}   {size / 2 ** 30:.2f} GiB"


# ---------- Runtime ----------
DEFAULT_RUNTIME = "exl3-gfx1201-b"   # 新版（8～64 列快路徑、自動草稿、前文快取）；不存在時退回第一套


def list_runtimes(base):
    """[(名稱, 目錄)]：RUNTIMES/ 下所有 backend=exllamav3 的 runtime。"""
    root = os.path.join(base, "RUNTIMES")
    try:
        names = sorted(os.listdir(root), key=str.lower)
    except OSError:
        return []
    out = []
    for n in names:
        p = os.path.join(root, n)
        try:
            with open(os.path.join(p, "runtime.json"), encoding="utf-8") as f:
                ok = json.load(f).get("backend") == "exllamav3"
        except (OSError, ValueError, AttributeError):
            continue
        if ok and os.path.isfile(os.path.join(p, "python", "python.exe")):
            out.append((n, p))
    return out


def find_runtime(base, prefer=""):
    """要用的 EXL3 runtime 目錄：prefer（名稱）→ DEFAULT_RUNTIME → 第一套；都沒有就回空字串。"""
    found = dict(list_runtimes(base))
    for name in (str(prefer or "").strip(), DEFAULT_RUNTIME):
        if name and name in found:
            return found[name]
    return next(iter(found.values()), "")


def runtime_features(runtime):
    """這套 runtime 的 server.py 支援哪些新開關（舊版 runtime 沒有的就略過，不讓 server 因未知參數啟動失敗）。"""
    try:
        with open(os.path.join(runtime, "server.py"), encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return set()
    feats = {flag for flag in ("--auto-draft", "--auto-draft-accept", "--gemv-wmma", "--gemv-max-m", "--gemv-wmma-tn", "--dflash2-host-cb",
                               "--qc-staging-sliced") if f"'{flag}'" in text}
    if "cpu_cache_size=int(" in text:
        feats.add("-ccs")
    if "hybrid_draft_tokens=self.args.hybrid_draft_tokens" in text:
        feats.add("-hdt")
    if "ngram_match_min=self.args.ngram_match_min" in text:
        feats.add("-ngram")
    return feats


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


# 顯存預估：權重＋草稿＋KV＋每路固定狀態（遞迴狀態、驗證暫存），對照「顯存 − 2.5 GiB」（Windows 桌面與其他程式
# 常駐約 2.5 GiB）。超過時 Windows 會把部分權重搬到系統 RAM，速度可能掉一半以上。
# 校準（RX 9070 16 GB、2.20bpw＋DFlash2、KV q4）：8 路×16384＋草稿 → 每輪 90 ms 變 160 ms；
# 8 路×8192＋草稿、8 路×16384 不開草稿、2 路×65536＋草稿 → 正常
VRAM_HEADROOM = 2.5 * 2 ** 30
VRAM_PER_SLOT = 0.25 * 2 ** 30


def kv_bytes_per_token(path, bits=4):
    """主模型每個 token 的 KV 位元組（只算 full attention 層；GDN／滑動視窗層是固定大小）"""
    c = _config(path) or {}
    t = c.get("text_config", c) if isinstance(c.get("text_config", c), dict) else c
    layers = t.get("num_hidden_layers") or 0
    types = t.get("layer_types")
    if isinstance(types, list) and types:
        full = sum(1 for x in types if x == "full_attention")
    elif t.get("full_attention_interval"):
        full = layers // int(t["full_attention_interval"])
    else:
        full = layers
    heads = t.get("num_key_value_heads") or t.get("num_attention_heads") or 0
    hd = t.get("head_dim") or ((t.get("hidden_size") or 0) // max(1, t.get("num_attention_heads") or 1))
    per_val = {16: 2.0, 8: 1.0625, 4: 0.53125}.get(bits, bits / 8 + 0.03125)
    return int(full * heads * hd * 2 * per_val)


def vram_check(main, runtime, ctx, parallel, ctk="q4_0", ctv="q4_0", draft=""):
    """(預估位元組, 上限位元組, 說明) — 權重＋草稿＋KV＋每路狀態，對照「顯存 − 2.5 GiB」"""
    try:
        ctx, parallel = int(str(ctx).strip()), max(1, int(str(parallel).strip()))
    except ValueError:
        return 0, 0, ""
    gpu = (weight_bytes(main)[0] or 0) - vision_bytes(main)
    dw = weight_bytes(draft)[0] if draft and draft != "-" and os.path.isdir(draft) else 0
    kb, vb = KV_BITS.get(ctk, 4), KV_BITS.get(ctv, 4)
    kv = (kv_bytes_per_token(main, kb) + kv_bytes_per_token(main, vb)) // 2 * ctx * parallel
    slots = VRAM_PER_SLOT * parallel
    total = gpu + dw + kv + slots
    vram = capabilities(runtime).get("gpu_total_bytes") or 16 * 2 ** 30 if runtime else 16 * 2 ** 30
    limit = vram - VRAM_HEADROOM
    text = (f"預估顯存 {total / 2 ** 30:.1f} GiB（權重 {gpu / 2 ** 30:.1f}"
            + (f"＋草稿 {dw / 2 ** 30:.1f}" if dw else "") + f"＋KV {kv / 2 ** 30:.1f}＋{parallel} 路狀態 {slots / 2 ** 30:.1f}）")
    return total, limit, text


def draft_matches(main, draft):
    if not is_exl3(main) or not (is_exl3(draft) or is_lilicorr(draft)) or not is_draft(draft):
        return False
    a, b = _config(main), _config(draft)
    a = a.get("llm_config", a.get("text_config", a))
    if is_lilicorr(draft):
        taps = b.get("dflash_config", {}).get("target_layer_ids", [])
        depth = a.get("num_hidden_layers") or len(a.get("layers_block_type", []))
        return (architecture(main).startswith("NemotronH") and a.get("hidden_size") == b.get("hidden_size")
                and a.get("vocab_size") == b.get("vocab_size") and bool(taps)
                and all(isinstance(t, int) and 0 < t < depth for t in taps))
    return (a.get("hidden_size") == b.get("hidden_size") and a.get("vocab_size") == b.get("vocab_size")
            and a.get("num_hidden_layers") == b.get("num_target_layers"))


def guess_draft(main, base):
    """依名稱前綴與張量形狀配對 EXL3 草稿（DFlash2）。"""
    prefix = os.path.basename(main).lower().split("-exl3")[0]
    for name, p, _size in list_models(base, draft=True):
        if not is_lilicorr(p) and name.lower().startswith(prefix + "-") and draft_matches(main, p):
            return p
    return None


def draft_block(draft):
    block = _config(draft).get("block_size") if is_lilicorr(draft) else _config(draft).get("dflash_config", {}).get("block_size")
    return block if isinstance(block, int) and block >= 2 else None


# ---------- 啟動指令 ----------
KV_BITS = {"f16": 16, "q8_0": 8, "q4_0": 4, "q3": 3, "q2": 2}   # q3／q2 只有 EXL3 能用（-cq 3、-cq 4,2）
RECURRENT_CACHE_GB = 4.0   # 多輪對話接續用的遞迴狀態檢查點（系統 RAM）
CHUNK_SIZE = 2048          # 讀提示詞時每次送進 GPU 的 token 數（65536＋q4 實測：512→1030、2048→1186 tok/s，+0.6 GiB）
CHUNK_SIZE_DRAFT = 1024    # 開草稿時草稿模型另佔約 1 GiB，chunk 降一級留顯存餘裕
SAMPLING = (("temp", "-temp"), ("topp", "-topp"), ("topk", "-topk"), ("minp", "-minp"),
            ("rpen", "-repp"), ("ppen", "-presp"), ("fpen", "-freqp"), ("rlastn", "-penr"))
# 選到 EXL3 模型時變灰的欄位：{欄位 key: 原因}。key 是 App 屬性名或 adv 字典的 key。
_ON_GPU = "EXL3 不使用 llama.cpp 的這個旗標；MoE 專家放系統 RAM 請調下方 EXL3 DRAM 放置，KV cache 仍在顯卡"
_EXL3_DRY_XTC = "ExLlamaV3 有這個取樣器，但 EXL3 server 目前沒有接上"
_EXL3_NO_SAMPLER = "ExLlamaV3 沒有這種取樣器"
_DFLASH_ONLY_MAX = "EXL3 的 DFlash2 草稿只看「草稿長度上限」"
NA_REASONS = {
    "ngl": _ON_GPU, "ngld": _ON_GPU, "nkvo": _ON_GPU, "ncmoe": _ON_GPU, "ncffn": _ON_GPU,
    "fa": "EXL3 的注意力計算由 runtime 決定，沒有這個開關",
    "jinja": "EXL3 自動使用模型資料夾自帶的 chat template",
    "rbud": "EXL3 server 不支援思考預算（硬性 token 上限）",
    "runtime": "EXL3 不用 llama-server 執行檔；要用哪套 EXL3 runtime 在「EXL3 進階」區選",
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
PLACEMENT_FIELDS = (
    ("exl3_mcl", "DRAM 專家層數 (-mcl)", "前 N 個 MoE 層的 routed experts 在 CPU 計算；只計 MoE 層，並非模型 block 編號。留空／0＝不指定。"),
    ("exl3_mcs", "每層 CPU 專家數 (-mcs)", "每個 MoE 層的尾端 N 個 routed experts 放 DRAM，由 CPU 計算；其餘專家在 GPU。與 -mcl 互斥；留空／0＝不指定。"),
    ("exl3_mct", "CPU 執行緒 (-mct)", "CPU MoE worker 的執行緒數。留空＝runtime 自動決定；Ryzen 7 5700 本機曾測得 6 線程較快。"),
    ("exl3_chunk", "Prefill chunk (-chunk_size)", "一次送入 GPU 的提示詞 token 數；較小可降低載入提示詞的顯存尖峰。留空＝自動，CPU offload 預設 512。"),
    ("exl3_rcs", "RAM 狀態快取 GB (-rcs)", "多輪對話的遞迴狀態檢查點快取，使用系統 RAM，不是 MoE 專家大小。可填 0／小數；留空＝自動，CPU offload 預設 0.25 GB。"),
)
# EXL3 進階：加速／快取／草稿（key, 標籤, 種類, 完整說明）。種類：check＝勾選框、entry＝輸入框。
# 勾選框都設計成「勾了＝改變預設」：模板缺值時是空字串，空字串必須等於預設行為。
RUNTIME_HELP = ("選要用哪一套 EXL3 執行環境（RUNTIMES/ 下有 runtime.json 標 exllamav3 的資料夾）。\n"
                "自動＝新版 exl3-gfx1201-b 優先：多了 8～64 列快路徑、自動草稿、前文快取等功能。\n"
                "新版有問題時選 exl3-gfx1201 退回舊版；舊版不支援的新開關會自動略過。改動後重新啟動 server。")
ACCEL_FIELDS = (
    ("exl3_no_wmma", "停用多列快路徑", "check",
     "新版預設「開」。多路同時生成或草稿較長時，顯卡一次要算 8～64 列；這條快路徑用顯卡的矩陣單元一次算完。\n"
     "實測（2.20bpw）：2 路＋草稿 7 從 13.8 → 63.8 tok/s；8 路不開草稿 190.7 tok/s。\n"
     "只有懷疑它算錯、要排查時才勾選關閉。舊版 runtime 沒有這條路徑，勾不勾都一樣。"),
    ("exl3_m32", "快路徑上限改回 32 列", "check",
     "新版預設一次最多 64 列（例：8 路 × 草稿 7 ＝ 64 列）。\n"
     "超過 32 列時用另一種加總順序，跟 32 列以內的結果只差在最後一位小數，精度一樣；\n"
     "但 greedy（溫度 0）輸出偶爾會差一個字。在意逐字一致才勾選；勾選後 33～64 列會走較慢的舊路徑。"),
    ("exl3_tn", "快路徑輸入共用 (--gemv-wmma-tn)", "entry",
     "快路徑裡每個 GPU 工作組一次算幾塊輸出。像廚房備料：原本每道菜各自把同一批食材洗切一次（填 1），\n"
     "改成一次備好再分給 2 或 4 道菜共用（填 2／4），省下重複搬資料的時間。只影響 1～4 bit 的權重層，其他照舊。\n"
     "留空／1＝原做法。2／4 的速度還沒有完整實測；覺得變慢或輸出怪異就清空。舊版 runtime 沒有這個開關，會自動略過。"),
    ("exl3_autodraft", "自動草稿長度 (--auto-draft)", "check",
     "只在 DSpark（投機解碼）模式有效。依「目前同時生成幾路」分別實測每種草稿長度的速度，自動挑最快的\n"
     "（範圍 0 ～ 草稿長度上限；0＝暫停打草稿，草稿模型照樣跟上進度，隨時可恢復）；要快 5% 以上才換，並定期重測。\n"
     "實測（2.20bpw、temp 0.7）：1～4 路開草稿明顯較快，8 路不開草稿較快；開這個就不用手動切換。"),
    ("exl3_autodraft_acc", "自動草稿（猜中率） (--auto-draft-accept)", "check",
     "只在 DSpark（投機解碼）模式有效。每一輪看「最近第 1、2、3…個草稿各猜中幾成」，再配上各路數實測的每輪耗時，\n"
     "挑每秒產出最多的草稿長度（0 ～ 草稿長度上限）。猜中率跟著內容走：寫程式／SVG 自動拉長，寫文章自動縮短，換題幾輪內就跟上。\n"
     "像排隊點餐：店員看最近客人點到第幾道就停，決定一次先備幾道。統計只在 CPU 做幾個加法，不佔 GPU。\n"
     "實測（2.20bpw、64K、KV 4,2）：1 路自動用 7、3 路自動用 4，速度跟手動挑最佳值相差 2% 內。勾了這個就不用再勾上一項。"),
    ("exl3_hdt", "重複內容長草稿 (-hdt)", "entry",
     "n-gram 補草稿的一種。當輸出跟前文「一字不差地重複」時（例：改程式碼時把整段原樣抄回、複述文件、\n"
     "照模板填表），直接把前文接下來那一段當草稿一次驗證，命中就一口氣前進很多字。\n"
     "填數字＝至少要連續相同幾個 token 才啟動，也是一次抄的長度（建議 16～32）。可和 DFlash2 草稿同時用；\n"
     "一般聊天幾乎不會觸發，不拖慢速度，但會多佔一點顯存。留空／0＝關。"),
    ("exl3_ngram", "純 n-gram 草稿 (-ngram)", "entry",
     "不用草稿模型，改從前文找相同片段來猜下一段（像輸入法的聯想字）。\n"
     "填數字＝至少連續相同幾個 token 才拿來當草稿（建議 2～4）。\n"
     "只在 Normal 模式（不載草稿模型）生效；DSpark 模式自動略過，因為 runtime 不允許兩種草稿同時開。留空／0＝關。"),
    ("exl3_ccs", "前文快取 RAM GB (-ccs)", "entry",
     "顯卡的 KV 快取滿了要騰位置時，把舊的前文搬到系統 RAM 保存；之後同一段前文（長系統提示、多輪對話、\n"
     "重送同一份文件）再出現就直接搬回，不用重新讀一次提示詞。填 GB；留空／0＝關。\n"
     "這台 32 GB RAM 建議 4～8。"),
    ("exl3_dcd", "前文快取 SSD 資料夾 (-dcd)", "entry",
     "RAM 那層也滿時，再往 SSD 這個資料夾存；server 重開後還在，同樣的長提示詞下次幾乎秒讀。\n"
     "要先填「前文快取 RAM GB」才會生效。請填專用資料夾，例：D:\\GGUFRun\\exl3-cache（不存在會自動建立）。"),
    ("exl3_dcs", "SSD 快取上限 GB (-dcs)", "entry",
     "SSD 快取資料夾最多佔多少 GB（所有模型共用），超過時刪最舊的。留空＝50 GB。"),
    ("exl3_hostcb", "實驗：草稿查表放 RAM (#490)", "check",
     "DFlash2 草稿模型裡用來選字的大查表改放系統 RAM，不佔顯存，約省 240 MB，可多擠一點 Context 或路數。\n"
     "在這張 AMD 卡上的速度影響還沒實測；覺得變慢就取消勾選。"),
    ("exl3_qcslice", "實驗：長 Context 分段搬運 (#517)", "check",
     "注意力計算時，量化 KV 分成固定大小的小段搬運，壓低超長 Context 時的顯存尖峰。\n"
     "只有 NVIDIA 上的實測數據，這張卡的效果未知；預設不勾。"),
)
ACCEL_KEYS = tuple(k for k, *_ in ACCEL_FIELDS) + ("exl3_runtime",)
# 欄位右邊的一行短說明（完整說明在「?」）
ACCEL_SHORT = {
    "exl3_no_wmma": "預設開：多路／長草稿時的加速主力，一般不要勾。",
    "exl3_m32": "預設 64 列；勾選＝逐字一致優先，速度較慢。",
    "exl3_tn": "留空＝原做法；填 2／4＝輸入共用（實驗）。",
    "exl3_autodraft": "依目前路數自動挑最快的草稿長度，路數多時自動暫停草稿（只在 DSpark）。",
    "exl3_autodraft_acc": "依最近猜中率每輪挑草稿長度，換內容幾輪內跟上（只在 DSpark）。",
    "exl3_hdt": "輸出照抄前文時一次猜一大段；建議 16～32，留空＝關。",
    "exl3_ngram": "不用草稿模型、從前文找相同片段猜；只在 Normal 模式。",
    "exl3_ccs": "前文存 RAM，重複的長提示不用重讀；建議 4～8，留空＝關。",
    "exl3_dcd": "RAM 滿了再存 SSD，重開還在；需先填 RAM GB。",
    "exl3_dcs": "SSD 快取最多佔幾 GB；留空＝50。",
    "exl3_hostcb": "實驗：草稿查表放 RAM，省約 240 MB 顯存。",
    "exl3_qcslice": "實驗：長 Context 降顯存尖峰；這張卡效果未知。",
}
GGUF_NA = {"vision": "視覺選項只對 EXL3 模型有效",
           **{key: "EXL3 專用；GGUF 請使用 -ncmoe／-ncffn 等欄位" for key, _label, _help in PLACEMENT_FIELDS},
           "exl3_embd": "EXL3 專用的 embedding 逐次讀取選項",
           **{key: "EXL3 專用（新版 runtime 的加速／快取／草稿選項）" for key in ACCEL_KEYS}}
VISION_MODES = ("off", "gpu", "ram")
# 選到 EXL3 模型時，仍可用的欄位在提示框多加的說明
EXL3_NOTES = {
    "ctx": "EXL3：每路容量，須為 256 的倍數；總 KV 容量＝Context × 同時生成數，KV cache 放在顯卡",
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


def parallel_count(value):
    """UI and launch commands share the supported concurrent generation range."""
    text = str(value).strip()
    try:
        count = int(text)
    except ValueError:
        raise ValueError("同時生成數必須是 1～8 的整數") from None
    if not 1 <= count <= 8:
        raise ValueError("同時生成數必須是 1～8 的整數")
    return count


def build_cmd(main, runtime, base, *, port, ctx, ctk, ctv, think, effort, speculative,
              drafter, sampling, draft_max="", chunk=None, recurrent_gb=None, vision="off", placement=None, parallel=2,
              accel=None):
    """回傳 (argv, 說明文字)。drafter：''＝自動、'-'＝不使用、其他＝路徑。
    vision：off＝不載入視覺塔；gpu＝權重放顯卡；ram＝權重放系統 RAM，看圖時才串流上顯卡。"""
    import math
    parallel = parallel_count(parallel)
    placement = placement or {}
    def integer(key, name, minimum=0):
        value = str(placement.get(key, "")).strip()
        if not value:
            return None
        try:
            number = int(value)
        except ValueError:
            raise ValueError(f"{name}必須是整數") from None
        if number < minimum:
            raise ValueError(f"{name}至少 {minimum}")
        return number
    mcl = integer("exl3_mcl", "DRAM 專家層數") or 0
    mcs = integer("exl3_mcs", "每層 CPU 專家數") or 0
    mct = integer("exl3_mct", "CPU 執行緒", 1)
    if mcl and mcs:
        raise ValueError("EXL3 的 -mcl（DRAM 專家層數）與 -mcs（每層 CPU 專家數）互斥，請只填一種")
    configured_chunk = integer("exl3_chunk", "Prefill chunk", 1)
    if configured_chunk is not None:
        chunk = configured_chunk
    elif chunk is None and (mcl or mcs):
        chunk = 512
    configured_rcs = str(placement.get("exl3_rcs", "")).strip()
    if configured_rcs:
        try:
            recurrent_gb = float(configured_rcs)
        except ValueError:
            raise ValueError("RAM 狀態快取必須是非負數（GB）") from None
    elif recurrent_gb is None:
        recurrent_gb = 0.25 if (mcl or mcs) else recurrent_cache_gb()
    recurrent_gb = float(recurrent_gb)
    if not math.isfinite(recurrent_gb) or recurrent_gb < 0:
        raise ValueError("RAM 狀態快取必須是有限的非負數（GB）")
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
        raise ValueError("KV 型別只支援 f16 / q8_0 / q4_0 / q3 / q2")
    cmd = [os.path.join(runtime, "python", "python.exe"), "-I", "-X", "utf8", "-u",
           os.path.join(runtime, "runtime.py"), "script", "server.py",
           "-m", main, "--alias", os.path.basename(main), "--host", "127.0.0.1", "--port", str(port),
           "-cs", str(ctx), "--parallel", str(parallel), "-rcs", f"{recurrent_gb:g}", "-chunk_size", "{chunk}",
           "--thinking", think if think in ("auto", "on", "off") else "on"]
    k, v = KV_BITS[ctk], KV_BITS[ctv]
    if (k, v) != (16, 16):
        cmd += ["-cq", str(k) if k == v else f"{k},{v}"]
    effort = (effort or "").strip()
    if effort and effort.lower() != "default":
        cmd += ["--reasoning-effort", effort]
    note = f"[EXL3] Runtime：{runtime}\n[EXL3] 檢查：{why}\n"
    note += f"[EXL3] 同時生成 {parallel} 路｜每路 Context {ctx}｜總 KV 容量 {ctx * parallel} token；主模型與草稿各載一份\n"
    _vt, _vl, _vtext = vram_check(main, runtime, ctx, parallel, ctk, ctv,
                                    ((drafter or guess_draft(main, base) or "") if drafter != "-" else "") if speculative else "")
    if _vt and _vt > _vl:
        note += (f"[EXL3] ⚠ {_vtext}，高於建議上限 {_vl / 2 ** 30:.1f} GiB：Windows 可能把部分權重搬到系統 RAM，"
                 f"速度可能掉一半以上。請調小每路 Context 或路數\n")
    if mcl:
        cmd += ["-mcl", str(mcl)]
    if mcs:
        cmd += ["-mcs", str(mcs)]
    if mct is not None:
        cmd += ["-mct", str(mct)]
    if str(placement.get("exl3_embd", "")).strip().lower() in ("1", "true", "on"):
        cmd += ["-embd"]
    if mcl or mcs:
        mode = f"前 {mcl} 個 MoE 層的專家" if mcl else f"每層尾端 {mcs} 個專家"
        note += f"[EXL3] DRAM 放置：{mode}由 CPU 計算；執行緒 {mct if mct is not None else '自動'}\n"
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
    note += _accel_args(cmd, runtime, accel or {})
    if not chunk:
        chunk = CHUNK_SIZE_DRAFT if "-dm" in cmd else CHUNK_SIZE
    cmd[cmd.index("{chunk}")] = str(int(chunk))
    note += f"[EXL3] 多輪快取：遞迴狀態檢查點 {recurrent_gb:g} GB（系統 RAM）；chunk {int(chunk)}\n"
    return cmd, note


def _accel_args(cmd, runtime, accel):
    """EXL3 進階（加速／快取／草稿）→ 附加到 cmd，回傳說明文字。runtime 不支援的開關略過並註明。"""
    import math
    feats = runtime_features(runtime)
    get = lambda k: str(accel.get(k, "") or "").strip()
    on = lambda k: get(k).lower() in ("1", "true", "on")
    note, skipped = "", []

    def number(key, name, kind=float, minimum=0):
        text = get(key)
        if not text:
            return None
        try:
            v = kind(text)
        except ValueError:
            raise ValueError(f"{name}必須是{'整數' if kind is int else '數字'}：{text}") from None
        if (kind is float and not math.isfinite(v)) or v < minimum:
            raise ValueError(f"{name}至少 {minimum}")
        return v

    def add(flag, *vals, feat=None):
        if (feat or flag) in feats:
            cmd.extend([flag, *vals])
            return True
        skipped.append(flag)
        return False

    if on("exl3_no_wmma"):
        add("--gemv-wmma", "0")
    if on("exl3_m32"):
        add("--gemv-max-m", "32")
    tn = get("exl3_tn")
    if tn and tn != "1":
        if tn not in ("2", "4"):
            raise ValueError(f"快路徑輸入共用只能填 1、2 或 4：{tn}")
        if add("--gemv-wmma-tn", tn):
            note += f"[EXL3] 快路徑輸入共用：每組一次算 {tn} 塊\n"
    if "--gemv-wmma" in feats:
        note += ("[EXL3] 多列快路徑：" + ("關" if on("exl3_no_wmma") else
                 f"開（一次最多 {32 if on('exl3_m32') else 64} 列）") + "\n")
    drafting = "-dm" in cmd
    if on("exl3_autodraft_acc"):
        if drafting:
            if add("--auto-draft-accept"):
                note += "[EXL3] 自動草稿（猜中率）：開（依最近各位置猜中率與每輪耗時，每輪挑草稿長度）\n"
        else:
            note += "[EXL3] 自動草稿（猜中率）：沒有載入草稿模型，略過\n"
    elif on("exl3_autodraft"):
        if drafting:
            if add("--auto-draft"):
                note += "[EXL3] 自動草稿長度：開（依同時生成路數分別實測，自動挑最快的長度）\n"
        else:
            note += "[EXL3] 自動草稿長度：沒有載入草稿模型，略過\n"
    hdt = number("exl3_hdt", "重複內容長草稿", int)
    if hdt:
        if add("-hdt", str(hdt)):
            note += f"[EXL3] 重複內容長草稿：連續相同 {hdt} token 起，一次抄 {hdt} token 驗證\n"
    ngram = number("exl3_ngram", "純 n-gram 草稿", int)
    if ngram:
        if drafting:
            note += "[EXL3] 純 n-gram 草稿：已載入草稿模型，兩者不能同時開，略過\n"
        elif add("-ngram", str(ngram)):
            note += f"[EXL3] 純 n-gram 草稿：連續相同 {ngram} token 起拿前文當草稿\n"
    ccs = number("exl3_ccs", "前文快取 RAM")
    dcd, dcs = get("exl3_dcd"), number("exl3_dcs", "SSD 快取上限")
    if ccs:
        if add("-ccs", f"{ccs:g}"):
            text = f"[EXL3] 前文快取：RAM {ccs:g} GB"
            if dcd:
                cmd.extend(["-dcd", dcd])
                text += f"＋SSD {dcd}"
                if dcs:
                    cmd.extend(["-dcs", f"{dcs:g}"])
                text += f"（上限 {dcs if dcs else 50:g} GB）"
            note += text + "\n"
    elif dcd:
        note += "[EXL3] 前文快取 SSD：要先填「前文快取 RAM GB」才會生效，略過\n"
    if on("exl3_hostcb"):
        if add("--dflash2-host-cb"):
            note += "[EXL3] 實驗：草稿查表放系統 RAM（#490）\n"
    if on("exl3_qcslice"):
        if add("--qc-staging-sliced"):
            note += "[EXL3] 實驗：量化 KV 分段搬運（#517）\n"
    if skipped:
        note += (f"[EXL3] ⚠ {os.path.basename(runtime)} 不支援這些開關，已略過：" + " ".join(skipped) +
                 f"（請在 EXL3 進階區選 {DEFAULT_RUNTIME}）\n")
    return note


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
