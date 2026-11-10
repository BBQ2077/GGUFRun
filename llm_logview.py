"""Server 輸出整理：把 llama-server／EXL3 server 的原始輸出整理成「有用的資訊」。

每一行原始輸出都會完整存檔（logs/），UI 只顯示整理後的行：
- 載入階段：裝置、自動分配結果、各模型上 GPU 層數、權重／KV／MoE 快取／計算緩衝放哪裡、視覺塔、投機解碼設定
- 請求階段：預填進度、生成中即時速度、完成時一行總結（預填、快取重用、生成速度、草稿接受率、Context 用量）
- 錯誤與值得注意的警告照實顯示；已知無害的雜訊隱藏
"""
import collections
import datetime
import glob
import json
import os
import re
import subprocess

_LINE = re.compile(r"^(?:\d+\.\d+\.\d+\.\d+ )?([IWED]) (.*)$")
_TASK = re.compile(r"\| task (\d+) \|")
_ERRORISH = re.compile(r"error|failed|abort|assert|exception|out of memory|denied|not found|cannot|unable", re.I)

# 已知無害、不顯示的警告／錯誤（原始檔仍保留）
_BENIGN = (
    "control-looking token", "is a normal token here", "sharing with layer",
    "security: no API key", "this warning is normal during memory fitting",
    "failed to measure the memory of the extra model", "the full capacity of the model will not be utilized",
)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _mib(x):
    return f"{x:,.0f} MiB" if x >= 100 else f"{x:.1f} MiB"


class _Model:
    def __init__(self, name):
        self.name = name
        self.layers = ""
        self.gpu_w = 0.0
        self.ram_w = 0.0
        self.kv_total = 0.0
        self.kv_ram = 0.0
        self.kv_types = ""
        self.compute_gpu = 0.0
        self.compute_ram = 0.0


class LlamaDigest:
    """llama-server 輸出 → 整理後的行（feed 回傳要顯示的字串清單，可能是空的）。"""

    def __init__(self):
        self.models = []          # [_Model]，第一個是主模型，之後是草稿
        self.cur = None
        self.loaded = False
        self.extra = []           # 就緒摘要裡的其他行
        self.tasks = {}           # task id → 統計
        self.last = None          # 最後一次完成的請求 {"tps", "acc", "pp_tps"}
        self.free_at_start = None
        self.total_vram = None
        self._tip_mmap = False

    # ---- 載入階段 ----
    def _model(self, name):
        m = _Model(name)
        self.models.append(m)
        self.cur = m
        return m

    def feed(self, raw):
        line = raw.rstrip("\r\n")
        if not line.strip():
            return []
        m = _LINE.match(line)
        if not m:
            # 非 llama.cpp 格式的行（續行、崩潰訊息）：只顯示像錯誤的
            s = line.strip()
            if s.startswith(("\t", "---")) or line.startswith("\t"):
                return []
            return [f"[輸出] {s}"] if _ERRORISH.search(s) else []
        level, msg = m.groups()
        msg = msg.strip()
        out = self._load_line(msg) if not self.loaded else None
        if out is None:
            out = self._request_line(msg)
        if out is not None:
            return out
        if level in "EW":
            if any(b in msg for b in _BENIGN):
                return []
            if "tensor overrides to CPU are used with mmap enabled" in msg:
                if self._tip_mmap:
                    return []
                self._tip_mmap = True
                return ["[提示] 專家放 RAM 時，llama.cpp 建議加 --load-mode none（不用 mmap）可能較快"]
            return [("[錯誤] " if level == "E" else "[警告] ") + re.sub(r"^\S+:\s+", "", msg)]
        return []

    def _load_line(self, msg):
        r = re.search(r"- (ROCm\d+|CUDA\d+|Vulkan\d+)\s*: (.+?) \((\d+) MiB, (\d+) MiB free\)", msg)
        if r and "common_param" in msg:
            total, free = int(r.group(3)), int(r.group(4))
            self.total_vram, self.free_at_start = total, free
            return [f"[GPU] {r.group(1)} {r.group(2)}｜顯存 {total:,} MiB，啟動時可用 {free:,} MiB"]
        r = re.search(r"system_info: n_threads = (\d+) \(n_threads_batch = (\d+)\) / (\d+)", msg)
        if r:
            return [f"[CPU] 生成執行緒 {r.group(1)}｜批次執行緒 {r.group(2)}｜邏輯核心 {r.group(3)}"]
        r = re.search(r"load_model: loading model '(.+)'", msg)
        if r:
            self._model("主模型")
            return [f"[載入] 主模型 {os.path.basename(r.group(1))}"]
        r = re.search(r"loading draft model '(.+)'", msg)
        if r:
            self._model("草稿")
            return [f"[載入] 草稿模型 {os.path.basename(r.group(1))}"]
        r = re.search(r"projected to use (\d+) MiB of device memory vs\. (\d+) MiB of free", msg)
        if r:
            return [f"[自動分配] 照目前參數預估用 {int(r.group(1)):,} MiB 顯存，可用 {int(r.group(2)):,} MiB"]
        r = re.search(r"will leave (\d+) >= (\d+) MiB of free device memory, no changes needed", msg)
        if r:
            return [f"[自動分配] 放得下，不用調整（預留 {int(r.group(1)):,} MiB）"]
        r = re.search(r"need to reduce device memory by (\d+) MiB", msg)
        if r:
            return [f"[自動分配] 放不下，要少放 {int(r.group(1)):,} MiB → llama.cpp 自動把部分權重移到 RAM"]
        r = re.search(r"- \S+ \((.+?)\): (\d+) layers(?: \(\s*(\d+) overflowing\))?,\s*(\d+) MiB used,\s*(\d+) MiB free", msg)
        if r and "fit" in msg:
            part = f"，其中 {r.group(3)} 層的專家留在 RAM" if r.group(3) else ""
            return [f"[自動分配] 結果：{r.group(2)} 層上 GPU{part}；顯存 {int(r.group(4)):,} MiB，剩 {int(r.group(5)):,} MiB"]
        if "failed to fit params" in msg:
            return ["[自動分配] ⚠ " + re.sub(r"^\S+:\s+", "", msg)]
        if self.cur is None:
            return None
        r = re.search(r"offloaded (\d+)/(\d+) layers to GPU", msg)
        if r:
            self.cur.layers = f"{r.group(1)}/{r.group(2)}"
            if r.group(1) == "0":
                return [f"[GPU] ⚠ {self.cur.name} 0 層上 GPU：整顆在 CPU 跑，會非常慢（沒抓到顯卡或 -ngl 0）"]
            return []
        r = re.search(r"(\S+) model buffer size =\s*([\d.]+) MiB", msg)
        if r:
            if r.group(1).startswith(("CPU", "Host")):
                self.cur.ram_w += _f(r.group(2))
            else:
                self.cur.gpu_w += _f(r.group(2))
            return []
        r = re.search(r"(\S+) KV buffer size =\s*([\d.]+) MiB", msg)
        if r:
            if r.group(1).startswith(("CPU", "Host")):
                self.cur.kv_ram += _f(r.group(2))
            return []
        r = re.search(r"llama_kv_cache: size =\s*([\d.]+) MiB.*K \((\w+)\):.*V \((\w+)\)", msg)
        if r:
            self.cur.kv_total += _f(r.group(1))
            self.cur.kv_types = f"{r.group(2)}/{r.group(3)}"
            return []
        r = re.search(r"MoE cache size =\s*([\d.]+) MiB for ([\d.]+) MiB of host experts, (\d+)% of the misses on the CPU", msg)
        if r:
            c, h = _f(r.group(1)), _f(r.group(2))
            self.extra.append(f"MoE 快取：顯卡 {_mib(c)}，對應 RAM 裡 {_mib(h)} 的專家（約 {c / h:.0%} 可同時在顯卡）"
                              + (f"；未命中時 {r.group(3)}% 交給 CPU 算" if r.group(3) != "0" else "") if h else "")
            return []
        r = re.search(r"(\S+) compute buffer size =\s*([\d.]+) MiB", msg)
        if r and "reserve_compute_meta" not in msg:
            if r.group(1).startswith(("CPU", "ROCm_Host", "Host")):
                self.cur.compute_ram = _f(r.group(2))
            else:
                self.cur.compute_gpu = _f(r.group(2))
            return []
        r = re.search(r"cap image_max_tokens \(original=(\d+)\) to (\d+)", msg)
        if r:
            self.extra.append(f"視覺：單張圖最多 {r.group(2)} token（模型原本 {r.group(1)}；加大 -ub 可提高）")
            return []
        if "mtmd_subcontext: using" in msg:
            self._vis_dev = "RAM（CPU 計算）" if "CPU" in msg else "顯卡"
            return []
        r = re.search(r"load_hparams: model size:\s*([\d.]+) MiB", msg)
        if r:
            self._vis_size = _f(r.group(1))
            return []
        r = re.search(r"loaded multimodal model, '(.+)'", msg)
        if r:
            where = getattr(self, "_vis_dev", "?")
            size = getattr(self, "_vis_size", 0)
            return [f"[視覺] 已載入 {os.path.basename(r.group(1))}｜放{where}｜{_mib(size)}"]
        r = re.search(r"adding speculative implementation '([\w-]+)'", msg)
        if r:
            self._spec = r.group(1)
            return []
        r = re.search(r"- n_max=(\d+), n_min=(\d+), p_min=([\d.]+)", msg)
        if r:
            self.extra.append(f"投機解碼：{getattr(self, '_spec', '')}｜一次草稿最多 {r.group(1)} 個（最少 {r.group(2)}）")
            return []
        r = re.search(r"initializing, n_slots = (\d+), n_ctx_slot = (\d+)", msg)
        if r:
            self._ctx = int(r.group(2))
            self.extra.insert(0, f"Context：每路 {self._ctx:,} token × {r.group(1)} 路")
            return []
        r = re.search(r"prompt cache is enabled, size limit: (\d+) MiB", msg)
        if r:
            self.extra.append(f"前文快取：RAM 最多 {int(r.group(1)):,} MiB（重複的前文不用重讀）")
            return []
        if "model loaded" in msg or "listening on" in msg:
            self.loaded = True
            out = self.summary()
            r = re.search(r"listening on (\S+)", msg)
            if r:
                out.append(f"[就緒] 監聽 {r.group(1)}")
            return out
        if "increase n_ubatch (-ub) to increase vision token budget" in msg:
            return []
        return None

    def summary(self):
        out = ["[就緒] ===== 載入摘要 ====="]
        gpu_total = 0.0
        for m in self.models:
            if not m.layers and not m.gpu_w and not m.ram_w:
                out.append(f"  {m.name}：已載入（細節要 -lv 4 才看得到）")
                continue
            kv_gpu = max(0.0, m.kv_total - m.kv_ram)
            row = [f"{m.name}：{m.layers or '?'} 層上 GPU", f"權重 顯卡 {_mib(m.gpu_w)}"]
            if m.ram_w:
                row.append(f"RAM {_mib(m.ram_w)}")
            if m.kv_total:
                row.append(f"KV {m.kv_types} 顯卡 {_mib(kv_gpu)}" + (f"、RAM {_mib(m.kv_ram)}" if m.kv_ram else ""))
            if m.compute_gpu:
                row.append(f"計算緩衝 顯卡 {_mib(m.compute_gpu)}")
            gpu_total += m.gpu_w + kv_gpu + m.compute_gpu
            out.append("  " + "｜".join(row))
        moe = 0.0
        for e in self.extra:
            r = re.match(r"MoE 快取：顯卡 ([\d,.]+)", e)
            if r:
                moe = _f(r.group(1).replace(",", ""))
            out.append("  " + e)
        gpu_total += moe
        if gpu_total:
            line = f"  顯卡合計（llama.cpp 自報）：約 {gpu_total:,.0f} MiB"
            if self.free_at_start:
                left = self.free_at_start - gpu_total
                line += f"｜啟動時可用 {self.free_at_start:,} MiB → 剩約 {left:,.0f} MiB"
                if left < 300:
                    line += "  ⚠ 幾乎沒餘裕，Windows 可能把部分資料搬到共用記憶體而變慢"
            out.append(line)
        return out

    # ---- 請求階段 ----
    def _request_line(self, msg):
        r = _TASK.search(msg)
        if not r:
            return None
        tid = r.group(1)
        t = self.tasks.setdefault(tid, {})
        r = re.search(r"new prompt, .*task\.n_tokens = (\d+)", msg)
        if r:
            t["todo"] = int(r.group(1))
            return []
        r = re.search(r"cached n_tokens = (\d+)", msg)
        if r and "cached" not in t:
            t["cached"] = int(r.group(1))
            return []
        r = re.search(r"prompt processing, n_tokens =\s*(\d+), progress = ([\d.]+), t =\s*([\d.]+) s / ([\d.]+) tokens per second", msg)
        if r:
            done, prog = int(r.group(1)), _f(r.group(2))
            todo = t.get("todo") or (round(done / prog) if prog else done)
            return [f"[GGUF] 預填 {done:,}/{todo:,} tok（{prog:.0%}）｜{_f(r.group(3)):.1f} s（{_f(r.group(4)):.1f} tok/s）"]
        r = re.search(r"n_gen =\s*(\d+), tg =\s*([\d.]+) t/s, tg_3s =\s*([\d.]+) t/s", msg)
        if r:
            return [f"[GGUF] 生成中 {int(r.group(1)):,} tok｜平均 {_f(r.group(2)):.1f} tok/s｜近 3 秒 {_f(r.group(3)):.1f} tok/s"]
        r = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens .*?([\d.]+) tokens per second", msg)
        if r:
            t["pp"] = (_f(r.group(1)) / 1000, int(r.group(2)), _f(r.group(3)))
            return []
        r = re.search(r"\s eval time =\s*([\d.]+) ms /\s*(\d+) tokens .*?([\d.]+) tokens per second", msg)
        if r:
            t["tg"] = (_f(r.group(1)) / 1000, int(r.group(2)), _f(r.group(3)))
            return []
        r = re.search(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)", msg)
        if r:
            t["draft"] = (int(r.group(2)), int(r.group(3)), _f(r.group(1)), _f(r.group(4)))
            return []
        r = re.search(r"acc per pos = \(([^)]*)\)", msg)
        if r:
            t["pos"] = "/".join(f"{_f(x):.0%}" for x in r.group(1).split(","))
            return []
        r = re.search(r"stop processing: n_tokens = (\d+), truncated = (\d+)", msg)
        if r:
            self.tasks.pop(tid, None)
            if "tg" not in t:
                return []
            return [self._done(t, int(r.group(1)), r.group(2) != "0")]
        return [] if msg.startswith("slot") or " slot " in msg[:20] else None

    def _done(self, t, used, truncated):
        parts = []
        if "pp" in t:
            sec, n, rate = t["pp"]
            parts.append(f"預填 {n:,} tok／{sec:.2f} s" + (f"（{rate:.1f} tok/s）" if n else ""))
        parts.append(f"快取重用 {t.get('cached', 0):,} tok")
        sec, n, rate = t["tg"]
        parts.append(f"生成 {n:,} tok／{sec:.2f} s（{rate:.1f} tok/s）")
        acc = None
        if "draft" in t:
            a, g, ratio, mean = t["draft"]
            acc = ratio
            d = f"草稿接受 {a}/{g}（{ratio:.0%}，每輪平均前進 {mean:.2f} 個）"
            if t.get("pos"):
                d += f"｜各位置命中 {t['pos']}"
            parts.append(d)
        ctx = getattr(self, "_ctx", None)
        parts.append(f"Context 已用 {used:,}" + (f"/{ctx:,}" if ctx else "") + " tok" + ("  ⚠ 超出 Context，前文被截掉" if truncated else ""))
        self.last = {"tps": rate, "acc": acc, "pp_tps": t.get("pp", (0, 0, 0))[2]}
        return "[GGUF] 完成：" + "｜".join(parts)


class Exl3Digest:
    """EXL3 server 本身已輸出中文摘要；這裡只收斂機器用的行，並記下最後一次的速度。"""

    def __init__(self):
        self.last = None
        self.loaded = False

    def feed(self, raw):
        line = raw.rstrip("\r\n")
        if not line.strip():
            return []
        r = re.match(r'HTTP \S+ "(\w+) (\S+) [^"]*" (\d{3})', line)
        if r:   # 存取紀錄：成功的不顯示（原始檔有），失敗的照實顯示
            return [] if r.group(3).startswith(("2", "3")) else [f"[EXL3] ⚠ HTTP {r.group(3)}：{r.group(1)} {r.group(2)}"]
        if line.startswith("EXL3_SERVER_READY"):
            self.loaded = True
            try:
                d = json.loads(line.split(" ", 1)[1])
                return [f"[EXL3] 就緒｜{d.get('alias') or ''}"]
            except (ValueError, IndexError):
                return ["[EXL3] 就緒"]
        if line.startswith("EXL3_HTTP_LISTEN"):
            p = line.split()
            return [f"[EXL3] 監聽 http://{p[1]}:{p[2]}" if len(p) >= 3 else "[EXL3] 開始監聽"]
        if line.startswith("EXL3_SERVER_LOAD_ERROR"):
            return ["[EXL3] ⚠ 載入失敗，原因見下方（最後一行通常就是重點）", line[len("EXL3_SERVER_LOAD_ERROR"):].strip()]
        r = re.search(r"生成 [\d,]+ tok／[\d.]+ s（([\d.]+) tok/s）", line)
        if r and "[EXL3]" in line:
            a = re.search(r"草稿接受 (\d+)/(\d+)", line)
            p = re.search(r"預填 [\d,]+ tok／[\d.]+ s（([\d.]+) tok/s）", line)
            self.last = {"tps": _f(r.group(1)), "acc": (int(a.group(1)) / int(a.group(2))) if a and int(a.group(2)) else None,
                         "pp_tps": _f(p.group(1)) if p else 0}
        return [line]


# ---- 原始輸出存檔 ----
def open_raw_log(base, engine, keep=30):
    """logs/<engine>-<時間>.log；只保留最新 keep 個。回傳 (檔案物件, 路徑)。"""
    d = os.path.join(base, "logs")
    os.makedirs(d, exist_ok=True)
    old = sorted(glob.glob(os.path.join(d, f"{engine}-*.log")))
    for f in old[:max(0, len(old) - keep + 1)]:
        try:
            os.remove(f)
        except OSError:
            pass
    path = os.path.join(d, f"{engine}-{datetime.datetime.now():%Y%m%d-%H%M%S}.log")
    return open(path, "a", encoding="utf-8", errors="replace"), path


class Tail:
    """最後 N 行原始輸出（程序異常結束時顯示）。"""

    def __init__(self, n=15):
        self.q = collections.deque(maxlen=n)

    def add(self, line):
        if line.strip():
            self.q.append(line.rstrip("\r\n"))


# ---- 顯存用量（Windows 效能計數器）----
_PS_VRAM = ("$d=(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples|Sort-Object CookedValue -Desc|Select-Object -First 1;"
            "$s=(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples|Sort-Object CookedValue -Desc|Select-Object -First 1;"
            "'{0} {1}' -f [int]($d.CookedValue/1MB),[int]($s.CookedValue/1MB)")


def vram_usage():
    """(專用 MiB, 共用 MiB)；讀不到回 None。整張顯卡的用量（含桌面、瀏覽器）。"""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", _PS_VRAM], capture_output=True, text=True,
                           timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        a, b = r.stdout.split()[:2]
        return int(a), int(b)
    except Exception:
        return None


def vram_report(before, after):
    """啟動前後的顯存比較 → 一行說明（含溢出警告）。"""
    if not after:
        return ""
    ded, sh = after
    s = f"[顯存] 整張卡：專用 {ded:,} MiB｜共用 {sh:,} MiB"
    if before:
        s += f"（啟動前 專用 {before[0]:,}｜共用 {before[1]:,}）"
        grow = sh - before[1]
        if grow > 800:   # 正常載入共用也會多幾百 MiB（實測 +200～+500）；溢出時是上千
            s += (f"\n[顯存] ⚠ 共用記憶體多了 {grow:,} MiB：顯卡放不下、溢到系統 RAM，生成會明顯變慢。"
                  "請調小 MoE 顯卡快取、Context，或多放幾層專家到 RAM")
    return s
