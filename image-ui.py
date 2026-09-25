#!/usr/bin/env python3
"""Independent configurable stable-diffusion.cpp controller."""
import json
import os
import pathlib
import socket
import subprocess
import threading
import time
import urllib.request
import tkinter as tk
import webbrowser
from tkinter import filedialog, scrolledtext, ttk
from urllib.parse import urlencode

from image_save_service import ImageSaveService

BASE = pathlib.Path(__file__).resolve().parent
SETTINGS = BASE / 'image-settings.json'
DEFAULT_RUNTIME = 'stable-diffusion-cuda12-master-908-88411ef/'
FILES = {
    'diffusion': 'IMAGE-MODELS/qwen-image-2.1-UC-Q4_K_M.gguf',
    'llm': 'IMAGE-MODELS/Qwen3VL-8B-Instruct-Q4_K_M.gguf',
    'vae': 'IMAGE-MODELS/qwen_image_2.1_vae_bf16.safetensors',
    'llm_vision': 'IMAGE-MODELS/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf',
}
FIELDS = (
    ('model', '--model', '完整 Checkpoint', '整合式模型；與獨立 diffusion 二選一'),
    ('diffusion', '--diffusion-model', '生圖模型', '獨立 diffusion 權重'),
    ('llm', '--llm', 'LLM 文字編碼器', 'Qwen-Image 等模型使用'),
    ('llm_vision', '--llm_vision', '視覺 mmproj', '只在 Qwen 指令修圖模式載入；其他模式不佔記憶體'),
    ('clip_l', '--clip_l', 'CLIP-L', '部分模型需要'),
    ('clip_g', '--clip_g', 'CLIP-G', '部分模型需要'),
    ('t5xxl', '--t5xxl', 'T5XXL', '部分模型需要'),
    ('vae', '--vae', 'VAE', '將潛空間轉成圖片'),
)
MODEL_SUFFIXES = {'.gguf', '.safetensors', '.ckpt', '.pt', '.pth'}


def find_image_runtimes(base):
    root = pathlib.Path(base) / 'RUNTIMES'
    return [(p.name + '/', str(p)) for p in sorted(root.iterdir(), key=lambda p: p.name.casefold())
            if p.is_dir() and (p / 'sd-server.exe').is_file()] if root.is_dir() else []


def find_image_models(base):
    root = pathlib.Path(base) / 'IMAGE-MODELS'
    return sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES),
                  key=lambda p: str(p).casefold()) if root.is_dir() else []


def model_path(base, value):
    path = pathlib.Path(value)
    return str(path if path.is_absolute() else pathlib.Path(base) / path)


def valid_conditioning_cache_size(value):
    """Match the backend's non-negative int argument without accepting malformed input."""
    return (value == 'default' or
            (isinstance(value, str) and value.isascii() and value.isdecimal()
             and (value == '0' or not value.startswith('0'))
             and len(value) <= 10 and int(value) <= 2147483647))


def build_server_cmd(base, port, offload=True, runtime=DEFAULT_RUNTIME, models=None, vision=False,
                     cache_mode='off', diffusion_fa=False, conditioning_cache_size='default'):
    if vision and not str((FILES if models is None else models).get('llm_vision', '')).strip():
        raise ValueError('Qwen 指令修圖需要選擇視覺 mmproj 檔案')
    try:
        port = int(port)
    except (ValueError, TypeError) as exc:
        raise ValueError('埠必須是 1～65535 的整數') from exc
    if not 1 <= port <= 65535:
        raise ValueError('埠必須是 1～65535 的整數')
    base = pathlib.Path(base)
    runtime_path = pathlib.Path(runtime)
    if not runtime_path.is_absolute():
        runtime_path = base / 'RUNTIMES' / runtime_path
    chosen = FILES if models is None else models
    cmd = [str(runtime_path / 'sd-server.exe')]
    for key, flag, _, _ in FIELDS:
        if key == 'llm_vision' and not vision:
            continue
        value = str(chosen.get(key, '')).strip()
        if value:
            cmd.extend((flag, model_path(base, value)))
    cmd.extend(('--listen-ip', '127.0.0.1', '--listen-port', str(port),
                '--serve-html-path', str(base / 'assets' / 'image-web.html')))
    if cache_mode not in ('off', 'spectrum'):
        raise ValueError('快取模式僅支援關閉或 Spectrum')
    if cache_mode == 'spectrum':
        cmd.extend(('--cache-mode', 'spectrum'))
    if not isinstance(diffusion_fa, bool):
        raise ValueError('Flash Attention 必須為開啟或關閉')
    if diffusion_fa:
        cmd.append('--diffusion-fa')
    # Optional Viggle adapter is never applied at startup; the native API
    # supplies it per task; an optional adapter must be downloaded separately.
    viggle = base / 'IMAGE-MODELS' / 'loras' / 'Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128-fused-gguf.safetensors'
    if viggle.is_file():
        cmd.extend(['--lora-model-dir', str(viggle.parent)])
    if not valid_conditioning_cache_size(conditioning_cache_size):
        raise ValueError('Conditioning cache 請輸入 default 或 0～2147483647 的整數')
    if conditioning_cache_size != 'default':
        cmd.extend(('--conditioning-cache-size', conditioning_cache_size))
    if offload:
        cmd.append('--offload-to-cpu')
    return cmd


def supports_conditioning_cache(executable):
    """Avoid passing an unsupported flag to user-selected sd-server runtimes."""
    try:
        result = subprocess.run([executable, '--help'], capture_output=True, text=True,
                                errors='replace', timeout=5)
        return result.returncode == 0 and '--conditioning-cache-size' in (result.stdout + result.stderr)
    except (OSError, subprocess.TimeoutExpired):
        return False


def read_log_delta(path, position, limit=24000):
    """Read only recent/new log bytes; tolerate missing or truncated logs."""
    try:
        with open(path, 'rb') as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            if position is not None and position <= end and end - position > limit:
                stream.seek(position)
                head = stream.read(min(4096, limit // 4))
                stream.seek(end - (limit - len(head)))
                tail = stream.read()
                text = (head.decode('utf-8', errors='replace') +
                        '\n[略過部分過長日誌；完整內容請見 server.log]\n' +
                        tail.decode('utf-8', errors='replace'))
            else:
                start = max(0, end - limit) if position is None or position > end else position
                stream.seek(start)
                text = stream.read().decode('utf-8', errors='replace')
                if start > 0 and (position is None or position > end):
                    text = '[僅顯示最近的日誌內容]\n' + text
            return text, end
    except FileNotFoundError:
        return '', 0


def port_busy(port):
    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def load_settings(path=None):
    path = SETTINGS if path is None else path
    try:
        data = json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(data, path=None):
    path = pathlib.Path(SETTINGS if path is None else path)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.proc = None
        self.logfile = None
        self.save_service = None
        self.mode_lock = threading.Lock()
        self.vision_loaded = False
        self.switching = False
        self.mode_error = ''
        self.switch_started = 0
        saved = load_settings()
        self.port = tk.StringVar(value=str(saved.get('port', '18436')))
        self.offload = tk.BooleanVar(value=saved.get('offload', True))
        self.cache_mode = tk.StringVar(value=saved.get('cache_mode') if saved.get('cache_mode') in ('off', 'spectrum') else 'off')
        self.diffusion_fa = tk.BooleanVar(value=saved.get('diffusion_fa') is True)
        self.conditioning_cache_size = tk.StringVar(value=saved.get('conditioning_cache_size')
            if valid_conditioning_cache_size(saved.get('conditioning_cache_size')) else 'default')
        self.runtime = tk.StringVar(value=saved.get('runtime', DEFAULT_RUNTIME))
        previous = saved.get('models', {}) if isinstance(saved.get('models'), dict) else {}
        self.models = {key: tk.StringVar(value=previous.get(key, FILES.get(key, '')))
                       for key, _, _, _ in FIELDS}
        root.title('GGUFRun — Image 模式（獨立 sd-server）')
        root.geometry('940x820')
        root.minsize(840, 680)
        pane = ttk.Panedwindow(root, orient='vertical')
        pane.pack(fill='both', expand=True)
        frame = ttk.Frame(pane, padding=14)
        pane.add(frame, weight=3)
        ttk.Label(frame, text='Image 模式｜選擇 sd-server 與相容的模型組合；和 LLM 模式互不干擾',
                  font=('Microsoft JhengHei UI', 10)).pack(anchor='w', pady=(0, 9))
        row = ttk.Frame(frame)
        row.pack(fill='x', pady=3)
        ttk.Label(row, text='Image runtime：', width=18).pack(side='left')
        self.runtime_box = ttk.Combobox(row, textvariable=self.runtime, width=52)
        self.runtime_box.pack(side='left', fill='x', expand=True)
        ttk.Button(row, text='瀏覽 exe', command=self.browse_runtime).pack(side='left', padx=4)
        ttk.Button(row, text='重新掃描', command=self.refresh).pack(side='left')
        ttk.Label(frame, text='模型欄位可直接輸入路徑、從 IMAGE-MODELS 選取，或瀏覽外部檔案；不用的欄位留空。',
                  wraplength=850).pack(anchor='w', pady=(5, 5))
        self.entries = {}
        for key, flag, label, explanation in FIELDS:
            row = ttk.Frame(frame)
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=label + '：', width=18).pack(side='left')
            cb = ttk.Combobox(row, textvariable=self.models[key], width=52)
            cb.pack(side='left', fill='x', expand=True)
            self.entries[key] = cb
            ttk.Button(row, text='瀏覽', command=lambda field=key: self.browse_model(field)).pack(side='left', padx=4)
            ttk.Label(row, text=explanation, foreground='#555').pack(side='left', padx=3)
        self.refresh()
        ttk.Label(frame, text='所需編碼器以模型文件為準；選到檔案不代表相容。缺檔／不相容不預先攔截，由 sd-server 記錄錯誤。',
                  wraplength=850).pack(anchor='w', pady=(8, 3))
        options = ttk.Frame(frame)
        options.pack(fill='x', pady=5)
        ttk.Label(options, text='本機埠：').pack(side='left')
        self.port_entry = ttk.Entry(options, textvariable=self.port, width=9)
        self.port_entry.pack(side='left', padx=(0, 18))
        ttk.Checkbutton(options, text='Offload 到系統 RAM（8 GB 顯卡建議）', variable=self.offload).pack(side='left')
        cache_row = ttk.Frame(frame)
        cache_row.pack(fill='x', pady=(0, 5))
        ttk.Label(cache_row, text='取樣加速：').pack(side='left')
        self.cache_box = ttk.Combobox(cache_row, textvariable=self.cache_mode, values=('off', 'spectrum'),
                                      state='readonly', width=14)
        self.cache_box.pack(side='left', padx=(0, 8))
        ttk.Label(cache_row, text='off＝原版；spectrum＝預測並跳過部分計算，畫質可能改變；停止後再選、重啟生效。',
                  wraplength=650).pack(side='left')
        speed_row = ttk.Frame(frame)
        speed_row.pack(fill='x', pady=(0, 5))
        self.fa_box = ttk.Checkbutton(speed_row, text='Flash Attention（diffusion；可能省顯存）',
                                      variable=self.diffusion_fa)
        self.fa_box.pack(side='left', padx=(0, 12))
        ttk.Label(speed_row, text='Conditioning cache：').pack(side='left')
        self.conditioning_box = ttk.Combobox(speed_row, textvariable=self.conditioning_cache_size,
                                            values=('default', '0', '4', '8'), state='normal', width=12)
        self.conditioning_box.pack(side='left', padx=(0, 8))
        ttk.Label(frame, text='Conditioning cache：可手動輸入 default 或非負整數；default 沿用後端預設。新版 master-908 支援，舊版 runtime 選數字會阻止啟動；容量過大可能增加 RAM／顯存用量。',
                  wraplength=850, foreground='#555').pack(anchor='w', pady=(0, 5))
        buttons = ttk.Frame(frame)
        buttons.pack(fill='x', pady=5)
        self.start_btn = ttk.Button(buttons, text='▶ 啟動 Image Server', command=self.start)
        self.start_btn.pack(side='left', padx=3)
        self.stop_btn = ttk.Button(buttons, text='■ 停止', command=self.stop, state='disabled')
        self.stop_btn.pack(side='left', padx=3)
        ttk.Button(buttons, text='🌐 開繁中生圖網頁', command=self.open_web).pack(side='left', padx=3)
        ttk.Button(buttons, text='📄 開啟日誌', command=self.open_log).pack(side='left', padx=3)
        self.status = ttk.Label(frame, text='未執行')
        self.status.pack(anchor='w', pady=7)
        ttk.Label(frame, text='繁中網頁：自動儲存圖片到 image-output、即時耗時、參數說明與排程。下方為本輪 server 日誌。',
                  wraplength=850).pack(anchor='w')
        logframe = ttk.LabelFrame(pane, text='Image Server 日誌（可拖曳分隔線調整高度）')
        pane.add(logframe, weight=2)
        self.log = scrolledtext.ScrolledText(logframe, height=10, wrap='word', state='disabled')
        self.log.pack(fill='both', expand=True, padx=10, pady=(3, 8))
        self._log_offset = 0
        try:
            if port_busy(int(self.port.get())):
                path = BASE / 'image-output' / 'server.log'
                self._log_offset = path.stat().st_size if path.exists() else 0
            else:
                self.reset_log()
        except (OSError, ValueError) as exc:
            self.status.config(text=f'無法重設日誌：{exc}')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self._poll_after = root.after(500, self.poll)

    def browse_runtime(self):
        path = filedialog.askopenfilename(title='選擇 sd-server.exe', filetypes=[('執行檔', '*.exe')])
        if path:
            self.runtime.set(str(pathlib.Path(path).parent))

    def browse_model(self, key):
        path = filedialog.askopenfilename(title='選擇模型檔案', initialdir=str(BASE / 'IMAGE-MODELS'),
              filetypes=[('模型檔案', '*.gguf *.safetensors *.ckpt *.pt *.pth'), ('所有檔案', '*.*')])
        if path:
            self.models[key].set(path)

    def refresh(self):
        self.runtime_box['values'] = [name for name, _ in find_image_runtimes(BASE)]
        paths = [str(p) for p in find_image_models(BASE)]
        for cb in self.entries.values():
            cb['values'] = paths

    def settings(self):
        return {'port': self.port.get(), 'offload': self.offload.get(), 'cache_mode': self.cache_mode.get(),
                'diffusion_fa': self.diffusion_fa.get(),
                'conditioning_cache_size': self.conditioning_cache_size.get(),
                'runtime': self.runtime.get(), 'models': {key: var.get() for key, var in self.models.items()}}

    def reset_log(self):
        """Start a fresh session without preserving prior runs in the widget or file."""
        path = BASE / 'image-output' / 'server.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8'):
            pass
        self._log_offset = 0
        self.log.config(state='normal')
        self.log.delete('1.0', 'end')
        self.log.config(state='disabled')

    def refresh_log(self):
        """Append only fresh log content on Tk's UI thread; keep the view bounded."""
        text, self._log_offset = read_log_delta(BASE / 'image-output' / 'server.log', self._log_offset)
        if not text:
            return
        at_bottom = self.log.yview()[1] >= 0.98
        self.log.config(state='normal')
        self.log.insert('end', text)
        if int(self.log.index('end-1c').split('.')[0]) > 1200:
            self.log.delete('1.0', '401.0')
        self.log.config(state='disabled')
        if at_bottom:
            self.log.see('end')

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        try:
            config = self.settings()
            cmd = build_server_cmd(BASE, self.port.get(), self.offload.get(),
                                   self.runtime.get(), config['models'], cache_mode=config['cache_mode'],
                                   diffusion_fa=config['diffusion_fa'],
                                   conditioning_cache_size=config['conditioning_cache_size'])
            port = int(self.port.get())
            if port_busy(port):
                raise RuntimeError(f'埠 {port} 已被使用；請換埠，不接管別人的程序。')
            if config['conditioning_cache_size'] != 'default' and not supports_conditioning_cache(cmd[0]):
                raise ValueError('所選 Image runtime 不支援 --conditioning-cache-size；請選 default 或更換支援版本。')
            save_settings(config)
            self.launch_settings = config
            with self.mode_lock:
                self.vision_loaded = False
                self.switching = False
                self.mode_error = ''
            self.reset_log()
            logpath = BASE / 'image-output' / 'server.log'
            logpath.parent.mkdir(parents=True, exist_ok=True)
            self.logfile = open(logpath, 'a', encoding='utf-8')
            self.logfile.write('\n[GGUFRun] 啟動：' + subprocess.list2cmdline(cmd) + '\n')
            self.logfile.flush()
            self.refresh_log()
            self.save_service = ImageSaveService(BASE / 'image-output', f'http://127.0.0.1:{port}',
                                                  mode_status=self.mode_status, request_mode=self.request_mode,
                                                  log_path=logpath).start()
            self.proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=self.logfile,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.start_btn.config(state='disabled')
            self.port_entry.config(state='disabled')
            self.cache_box.config(state='disabled')
            self.fa_box.config(state='disabled')
            self.conditioning_box.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status.config(text=f'啟動中 PID {self.proc.pid}；請等待模型載入。')
        except Exception as exc:
            self.record_failure(exc)

    def mode_status(self):
        with self.mode_lock:
            vision, switching, error = self.vision_loaded, self.switching, self.mode_error
        proc = self.proc
        running = proc is not None and proc.poll() is None
        ready = False
        if running and not switching:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.launch_settings["port"]}/sdapi/v1/options', timeout=0.5) as response:
                    ready = response.status == 200
            except (OSError, ValueError):
                pass
        return {'vision': vision, 'offload': self.launch_settings['offload'] if running else self.offload.get(),
                'switching': switching, 'ready': ready, 'error': error}

    def request_mode(self, mode):
        vision = mode == 'qwen_edit'
        with self.mode_lock:
            if self.switching:
                return False, '伺服器正在切換，請稍候'
            if self.proc is None or self.proc.poll() is not None:
                return False, 'Image Server 未執行'
            desired_offload = self.launch_settings['offload']
            if vision == self.vision_loaded:
                return True, '模式已就緒'
            if vision:
                path = self.launch_settings['models'].get('llm_vision', '')
                if not path or not pathlib.Path(model_path(BASE, path)).is_file():
                    return False, '找不到所選視覺 mmproj；請在控制窗設定並重新啟動'
            self.switching = True
            self.mode_error = ''
        try:
            self.root.after(0, lambda: self.begin_switch(vision, desired_offload))
        except (RuntimeError, tk.TclError):
            with self.mode_lock:
                self.switching = False
            return False, '控制窗已關閉'
        return True, '切換中；正在安全重啟 Image Server'

    def begin_switch(self, vision, offload):
        if not self.switching or self.proc is None:
            return
        self.switch_started = time.monotonic()
        self.status.config(text='切換模式：等待舊 Image Server 退出；不關閉存圖服務與佇列。')
        self.proc.terminate()
        self.root.after(200, lambda: self.finish_switch(vision, offload))

    def finish_switch(self, vision, offload):
        if not self.switching or self.proc is None:
            return
        age = time.monotonic() - self.switch_started
        if self.proc.poll() is None:
            if age > 10:
                self.proc.kill()  # Only the process owned by this controller.
            if age > 15:
                self.switch_failed('舊 server 無法停止；請手動檢查程序')
                return
            self.root.after(200, lambda: self.finish_switch(vision, offload))
            return
        port = int(self.launch_settings['port'])
        if port_busy(port):
            if age > 15:
                self.switch_failed('埠仍被占用；沒有接管其他程序')
                return
            self.root.after(200, lambda: self.finish_switch(vision, offload))
            return
        try:
            config = self.launch_settings
            cmd = build_server_cmd(BASE, port, offload, config['runtime'], config['models'],
                                   vision=vision, cache_mode=config['cache_mode'],
                                   diffusion_fa=config.get('diffusion_fa', False),
                                   conditioning_cache_size=config.get('conditioning_cache_size', 'default'))
            self.logfile.write('\n[GGUFRun] 切換到' + ('Qwen 指令修圖（帶 mmproj）' if vision else '無視覺權重模式') +
                               ('；CPU offload 開啟：' if offload else '；自動適配：') + subprocess.list2cmdline(cmd) + '\n')
            self.logfile.flush()
            self.proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=self.logfile,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            config['offload'] = offload
            self.offload.set(offload)
            try:
                save_settings(config)
            except OSError as exc:
                self.logfile.write(f'[GGUFRun] 無法保存顯存策略設定（下次啟動需重選）：{exc}\n')
                self.logfile.flush()
            with self.mode_lock:
                self.vision_loaded = vision
                self.switching = False
                self.mode_error = ''
            self.status.config(text='Image Server 重新載入中；mmproj ' + ('按需載入' if vision else '已卸載'))
        except Exception as exc:
            self.switch_failed(str(exc))

    def switch_failed(self, message):
        with self.mode_lock:
            self.switching = False
            self.mode_error = message
        self.status.config(text='模式切換失敗：' + message)

    def record_failure(self, exc):
        logpath = BASE / 'image-output' / 'server.log'
        logpath.parent.mkdir(parents=True, exist_ok=True)
        with open(logpath, 'a', encoding='utf-8') as log:
            log.write(f'[GGUFRun] 啟動失敗：{type(exc).__name__}: {exc}\n')
        if self.save_service:
            self.save_service.close()
            self.save_service = None
        if self.logfile:
            self.logfile.close()
            self.logfile = None
        self.status.config(text=f'啟動失敗：{exc}；詳見下方日誌')
        self.refresh_log()

    def stop(self):
        with self.mode_lock:
            self.switching = False
            self.vision_loaded = False
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
            self.proc = None
        if self.save_service:
            self.save_service.close()
            self.save_service = None
        if self.logfile:
            self.logfile.close()
            self.logfile = None
        self.refresh_log()
        self.start_btn.config(state='normal')
        self.port_entry.config(state='normal')
        self.cache_box.config(state='readonly')
        self.fa_box.config(state='normal')
        self.conditioning_box.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status.config(text='已停止')

    def poll(self):
        if self.proc is not None:
            self.refresh_log()
        if self.proc is not None and self.proc.poll() is not None and not self.switching:
            code = self.proc.returncode
            self.stop()
            self.status.config(text=f'Server 啟動／執行失敗（代碼 {code}）；詳見下方日誌')
        self._poll_after = self.root.after(500, self.poll)

    def open_log(self):
        path = BASE / 'image-output' / 'server.log'
        if path.is_file():
            os.startfile(str(path))
        else:
            self.status.config(text='尚無日誌，啟動後會建立 image-output/server.log')

    def open_web(self):
        try:
            port = int(self.port.get())
            if not 1 <= port <= 65535:
                raise ValueError()
        except ValueError:
            self.status.config(text='埠必須是 1～65535 的整數')
            return
        if self.proc is None or self.proc.poll() is not None or not port_busy(port) or not self.save_service:
            self.status.config(text='請先啟動 Image Server 和存圖服務，並等模型載入完成；失敗詳見日誌。')
            return
        url = (f'http://127.0.0.1:{port}/?' + urlencode({
            'save_port': self.save_service.httpd.server_port, 'save_token': self.save_service.token}))
        browser = next((p for p in (r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                                    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
                                    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                                    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe') if os.path.isfile(p)), None)
        if browser:
            try:
                profile = ('_image_chrome_profile' if os.path.basename(browser).lower() == 'chrome.exe'
                           else '_image_webview_profile')
                subprocess.Popen([browser, f'--app={url}',
                                  f'--user-data-dir={BASE / "tools" / profile}',
                                  '--no-first-run', '--no-default-browser-check',
                                  '--disable-extensions', '--disable-background-networking',
                                  '--window-size=1100,780'])
                return
            except OSError as exc:
                self.status.config(text=f'輕量瀏覽器啟動失敗：{exc}；改用系統預設瀏覽器。')
        try:
            webbrowser.open(url)
        except Exception as exc:
            self.status.config(text=f'開啟瀏覽器失敗：{exc}；請從控制窗重新開啟。')

    def close(self):
        if self._poll_after is not None:
            try:
                self.root.after_cancel(self._poll_after)
            except tk.TclError:
                pass
            self._poll_after = None
        try:
            save_settings(self.settings())
        except OSError as exc:
            self.record_failure(exc)
        self.stop()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    ImageApp(root)
    root.mainloop()
