r"""-ncmoe 實測：同一個模型、不同 -ncmoe 值，量測載入時間與生成速度。

用法：
    python tools/_bench_ncmoe.py [值1 值2 ...]     (值 0 = 送 -ncmoe 0；'off' = 完全不送)

模型與執行檔來源（環境變數，擇一設定；都不設就用預設推斷）：
    set MODEL=C:\models\your-model.gguf      # 要測的主模型
    set RT_DIR=C:\llama-bXXXXX                # 內含 llama-server.exe 的資料夾（llama.cpp build 編號）
    set PORT=18990                             # 測試用埠（預設 18990，不與 UI 的 18435 衝突）
  預設推斷：MODEL = 專案資料夾裡第一個 *.gguf；RT_DIR = 專案資料夾下的 runtime\
"""
import json, os, re, socket, subprocess, sys, time, urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.environ.get("RT_DIR", "").strip() or os.path.join(BASE, "runtime")


def _pick_model():
    """MODEL 環境變數優先；否則取專案資料夾裡排序後第一個 .gguf。"""
    m = os.environ.get("MODEL", "").strip()
    if m:
        return m
    try:
        for n in sorted(os.listdir(BASE)):
            if n.lower().endswith(".gguf"):
                return os.path.join(BASE, n)
    except OSError:
        pass
    return ""


MODEL = _pick_model()
PORT = int(os.environ.get("PORT", "18990") or "18990")
PROMPT = "Explain speculative decoding in one sentence."


def port_open(p):
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", p))
        return True
    except Exception:
        return False
    finally:
        s.close()


def ask(prompt, mx=64):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mx}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                               data=body, headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=900).read())
    t = d.get("timings", {})
    return t.get("prompt_per_second", 0), t.get("predicted_per_second", 0)


def run(val):
    cmd = [os.path.join(RT, "llama-server.exe"), "-m", MODEL, "-c", "4096",
           "-ngl", "all", "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0",
           "--port", str(PORT), "-np", "1"]
    if val != "off":
        cmd += ["-ncmoe", str(val)]
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=RT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buf = []
    import threading
    def rd():
        for line in p.stdout:
            buf.append(line.decode("utf-8", "replace"))
    threading.Thread(target=rd, daemon=True).start()
    load = None
    while time.time() - t0 < 300:
        if p.poll() is not None:
            break
        if port_open(PORT):
            # 等到 /props 回應代表真的可服務
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=3).read()
                load = time.time() - t0
                break
            except Exception:
                pass
        time.sleep(0.5)
    res = {"ncmoe": val, "load_s": None if load is None else round(load, 1)}
    if load:
        tg, pp = [], []
        for i in range(3):
            try:
                a, b = ask(PROMPT)
                pp.append(a); tg.append(b)
            except Exception as e:
                res["err"] = str(e)[:120]
                break
            time.sleep(0.5)
        if tg:
            res["tg_runs"] = [round(x, 2) for x in tg]
            res["pp_runs"] = [round(x, 2) for x in pp]
            res["tg_med"] = round(sorted(tg)[len(tg) // 2], 2)
    txt = "".join(buf)
    m = re.findall(r"(CUDA0 model buffer size|CPU_Mapped model buffer size|CPU model buffer size)\s*=\s*([\d.]+) MiB", txt)
    res["buffers"] = m
    try:
        p.terminate(); p.wait(timeout=15)
    except Exception:
        subprocess.run(["taskkill", "/F", "/PID", str(p.pid)], capture_output=True)
    for _ in range(20):
        if not port_open(PORT):
            break
        time.sleep(0.5)
    return res


if __name__ == "__main__":
    if not MODEL or not os.path.isfile(MODEL):
        print("[ERROR] 找不到模型：請設 MODEL 環境變數，或把 .gguf 放到專案資料夾。")
        sys.exit(1)
    if not os.path.isfile(os.path.join(RT, "llama-server.exe")):
        print(f"[ERROR] 找不到執行檔：{os.path.join(RT, 'llama-server.exe')}（可用 RT_DIR 指定）")
        sys.exit(1)
    print(f"[INFO] model = {MODEL}\n[INFO] rt    = {RT}\n[INFO] port  = {PORT}", flush=True)
    vals = sys.argv[1:] or ["off", "16"]
    for v in vals:
        print(json.dumps(run(v), ensure_ascii=False), flush=True)
