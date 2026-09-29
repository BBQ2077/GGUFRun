#!/usr/bin/env python3
"""Independent configurable stable-diffusion.cpp controller."""
import functools
import json
import os
import pathlib
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
import urllib.request
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, scrolledtext, ttk
from urllib.parse import urlencode

from image_save_service import ImageSaveService

BASE = pathlib.Path(__file__).resolve().parent
SETTINGS = BASE / 'image-settings.json'
DEFAULT_RUNTIME = 'stable-diffusion-cuda12-master-929-3f8527a/'
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


class Tip:
    """滑鼠移到控件上時顯示說明（照 gguf-ui.py 那套）。

    長段說明不再鋪在畫面上把表單撐高，改成掛在標籤或「?」鈕上。
    """

    def __init__(self, widget, text):
        self.w, self.text, self.tip = widget, text, None
        widget.bind('<Enter>', self._show, add='+')
        widget.bind('<Leave>', self._hide, add='+')

    def _show(self, _=None):
        if self.tip or not self.text:
            return
        try:
            x = self.w.winfo_rootx() + 16
            y = self.w.winfo_rooty() + self.w.winfo_height() + 4
            self.tip = tk.Toplevel(self.w)
            self.tip.wm_overrideredirect(True)
            self.tip.wm_geometry(f'+{x}+{y}')
            tk.Label(self.tip, text=self.text, background='#ffffe0', relief='solid',
                     borderwidth=1, justify='left', padx=6, pady=3,
                     wraplength=460).pack()
        except tk.TclError:
            self.tip = None

    def _hide(self, _=None):
        if self.tip:
            try:
                self.tip.destroy()
            except tk.TclError:
                pass
            self.tip = None


def hint(parent, text, width=2):
    """一個小小的「?」鈕，按下去或滑過去顯示說明。回傳該鈕。"""
    btn = ttk.Label(parent, text='?', foreground='#0b6bcb', cursor='question_arrow',
                    font=('Microsoft JhengHei UI', 9, 'bold'))
    Tip(btn, text)
    btn.bind('<Button-1>', lambda _e: _popup_hint(btn, text))
    return btn


def _popup_hint(widget, text):
    """「?」鈕的點擊版說明：可反白選取並複製，改錯設定時還能貼回欄位。"""
    try:
        win = tk.Toplevel(widget)
        win.title('說明（文字可選取複製）')
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill='both', expand=True)
        # ttk.Label 的文字無法反白選取；改用唯讀 Text，這樣說明、路徑與預設值都能用滑鼠
        # 選取後 Ctrl+C 複製，使用者不必憑記憶重打。
        box = tk.Text(frame, wrap='word', width=64, height=min(20, max(3, text.count(chr(10)) + 1)),
                      relief='flat', highlightthickness=0, background=frame.winfo_toplevel().cget('bg'))
        box.insert('1.0', text)
        box.configure(state='disabled')
        box.pack(fill='both', expand=True, anchor='w')
        buttons = ttk.Frame(frame)
        buttons.pack(fill='x', pady=(8, 0))

        def copy_all():
            try:
                widget.clipboard_clear()
                widget.clipboard_append(text)
            except tk.TclError:
                pass

        ttk.Button(buttons, text='複製全部', command=copy_all).pack(side='right')
        ttk.Button(buttons, text='關閉', command=win.destroy).pack(side='right', padx=(0, 6))
        win.transient(widget.winfo_toplevel())
    except tk.TclError:
        pass


class Section:
    """可折疊區塊：標題列一點就展開／收合，把不常改的選項收起來別擠壓日誌。"""

    def __init__(self, parent, title, expanded=False):
        self.expanded = bool(expanded)
        self.header = ttk.Frame(parent)
        self.header.pack(fill='x', pady=(6, 0))
        self.btn = ttk.Button(self.header, text=self._arrow(), width=3,
                              command=self.toggle)
        self.btn.pack(side='left')
        self.title_lbl = ttk.Label(self.header, text=title, font=('Microsoft JhengHei UI', 10, 'bold'))
        self.title_lbl.pack(side='left', padx=(4, 0))
        for w in (self.title_lbl, self.header):
            w.bind('<Button-1>', lambda _e: self.toggle())
        self.body = ttk.LabelFrame(parent, text='')
        if self.expanded:
            self.body.pack(fill='x', pady=(2, 4))

    def _arrow(self):
        return '▾' if self.expanded else '▸'

    def toggle(self):
        self.expanded = not self.expanded
        self.btn.config(text=self._arrow())
        if self.expanded:
            self.body.pack(fill='x', pady=(2, 4))
        else:
            self.body.pack_forget()
        # 內容高度變了，重算捲動範圍。
        cb = getattr(self, 'on_toggle', None)
        if cb:
            cb()


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


# Files that can never fill a given field, matched on the file's role in the name.
# Picking one of these is not a harmless mistake: sd-server dies at startup with
# 'model metadata validation failed' (a LoRA handed to --vae) and the field looks
# deceptively normal afterwards, so the dropdown says so up front.
_NOT_FOR_FIELD = {
    'clip_l': ('lora', 'upscal', 'esrgan', 'mmproj'),
    'clip_g': ('lora', 'upscal', 'esrgan', 'mmproj'),
    't5xxl': ('lora', 'upscal', 'esrgan', 'mmproj'),
    'llm': ('lora', 'upscal', 'esrgan'),
    'llm_vision': ('lora', 'upscal', 'esrgan'),
    'model': ('lora', 'upscal', 'esrgan'),
    'diffusion': ('upscal', 'esrgan'),
}

# Appended to candidates that cannot fill the field, so the dropdown stays honest
# instead of filtering them out (a filtered list would hide a value set elsewhere).
NOT_FOR_FIELD_MARK = '  ✗ 不適用'


@functools.lru_cache(maxsize=256)
def _safetensors_holds_vae(path_str):
    """Peek a .safetensors header for VAE tensors (cheap: only the JSON header is read)."""
    try:
        with open(path_str, 'rb') as fh:
            length = struct.unpack('<Q', fh.read(8))[0]
            if not 0 < length < 64 * 1024 * 1024:
                return False
            header = json.loads(fh.read(length))
    except (OSError, ValueError, struct.error):
        return False
    meta = header.get('__metadata__') or {}
    if 'vae' in str(meta.get('modelspec.architecture', '')).casefold():
        return True
    return any(str(k).startswith(('first_stage_model', 'conv1.', 'decoder.', 'encoder.'))
               for k in header if k != '__metadata__')


def model_fits_field(base, key, value):
    """False when the file plainly cannot serve this field (a LoRA in VAE, an upscaler...).

    Name-based (plus a VAE header peek) on purpose: the field still accepts any path the
    user types.  This drives the dropdown marks and the confirmation prompt, never a
    hard rejection, so an unusual but valid file is never locked out.
    """
    path = pathlib.Path(str(value))
    if not path.is_absolute():
        path = pathlib.Path(base) / path
    name = path.name.casefold()
    if key == 'vae':
        return 'vae' in name or _safetensors_holds_vae(str(path))
    return not any(token in name for token in _NOT_FOR_FIELD.get(key, ()))


def model_fieldui_labels(base, key, extra=()):
    """Dropdown candidates for one field, mismatches marked; outside files kept verbatim.

    The currently stored value is always included exactly as typed, because a value that
    is missing from the list would look empty in the field.
    """
    labels = []
    for model in find_image_models(base):
        shown = str(model)
        labels.append(shown if model_fits_field(base, key, shown) else shown + NOT_FOR_FIELD_MARK)
    for value in extra:
        value = str(value or '').strip()
        if value and value not in labels:
            labels.append(value if model_fits_field(base, key, value) else value + NOT_FOR_FIELD_MARK)
    return labels


CACHE_MODES = ('off', 'spectrum', 'dbcache', 'easycache', 'taylorseer', 'cache-dit')
CACHE_MODE_LABELS = {
    'off': '關閉（原版，最清晰）',
    'spectrum': 'Spectrum（可調 W，建議 0.10）',
    'dbcache': 'DBCache（區塊級，建議 threshold 0.25）',
    'easycache': 'EasyCache（最快，畫質降較多）',
    'taylorseer': 'TaylorSeer（平衡，約省 8%）',
    'cache-dit': 'Cache-DiT（平衡，約省 8%）',
}


SPECTRUM_W_DEFAULT = '0.10'
# DiT 區塊級快取（dbcache／taylorseer／cache-dit）以 threshold＋warmup 控制：
# threshold 越大越愛跳步（越快、越容易崩），warmup 是前 N 步不做快取。
# 8 GB 卡上的實測：threshold 0.25 + warmup 4 於 20 步省約 53 秒，肉眼幾乎無損；
# 0.35 起蕾絲／髮絲等細紋理明顯融掉，0.5 以上直接崩壞，所以預設保守。
CACHE_THRESHOLD_DEFAULT = '0.25'
CACHE_WARMUP_DEFAULT = '4'
CACHE_OPTION_MODES = ('dbcache', 'taylorseer', 'cache-dit')
# VAE 解碼分塊的相對大小（需搭配 --vae-tiling）。<=1 是「相對圖邊長的比例」，
# 越小塊數越多、越省顯存但越慢；留空＝後端預設（256x256，實測最慢）。
# 0.5 在 1024x1536 上把解碼從約 26 秒壓到約 9 秒且逐像素無損。
VAE_RELATIVE_TILE_DEFAULT = ''


def valid_spectrum_w(value):
    """Spectrum forecasting weight: only 0.05～1.0 is meaningful; larger is faster but softer."""
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return 0.05 <= number <= 1.0


def format_spectrum_w(value):
    """Normalise to a compact decimal string so the argv stays reproducible."""
    return f'{float(value):g}'


def valid_cache_threshold(value):
    """區塊級快取的 threshold：留空＝不送；否則 0～100 的數字。"""
    if isinstance(value, bool) or value is None:
        return False
    text = str(value).strip()
    if not text:
        return True
    try:
        number = float(text)
    except (TypeError, ValueError):
        return False
    return 0.0 <= number <= 100.0


def format_cache_threshold(value):
    """正規化為緊湊字串，讓 argv 可重現。"""
    return f'{float(value):g}'


def valid_cache_warmup(value):
    """快取暖機步數：留空＝不送；否則 0～50 的整數。"""
    if isinstance(value, bool) or value is None:
        return False
    text = str(value).strip()
    if not text:
        return True
    return text.isascii() and text.isdigit() and (text == '0' or not text.startswith('0')) \
        and int(text) <= 50


def valid_vae_relative_tile_size(value):
    """VAE 相對分塊：留空＝後端預設；'0.5'＝邊長比例；'512x768'＝明確尺寸。"""
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return True
    lowered = text.lower().replace(' ', '')
    if 'x' in lowered:
        parts = lowered.split('x')
        return (len(parts) == 2 and all(p.isascii() and p.isdigit() and int(p) > 0 for p in parts))
    try:
        number = float(lowered)
    except (TypeError, ValueError):
        return False
    return 0.01 <= number <= 1.0


def valid_conditioning_cache_size(value):
    """Match the backend's non-negative int argument without accepting malformed input."""
    return (value == 'default' or
            (isinstance(value, str) and value.isascii() and value.isdecimal()
             and (value == '0' or not value.startswith('0'))
             and len(value) <= 10 and int(value) <= 2147483647))


MAX_EXTRA_ARGS = 500
MAX_EXTRA_TOKENS = 40
# The window, the HTML page and the auto-save service all rely on these flags; re-typing
# them in «額外指令» would silently desynchronise the model fields, the page origin check
# or the log the page reads, so they are refused before launch instead of taking effect.
MANAGED_FLAGS = frozenset((
    '--model', '--diffusion-model', '--llm', '--llm_vision', '--clip_l', '--clip_g',
    '--t5xxl', '--vae', '--lora-model-dir', '--listen-ip', '--listen-port',
    '--serve-html-path', '--offload-to-cpu', '--cache-mode', '--cache-option',
    '--diffusion-fa', '--conditioning-cache-size',
))
# sd-server: «a negative value reserves that much free VRAM», so -3 keeps 3 GiB for the
# desktop while the automatic graph-cut execution budgets the rest.  The number itself is
# editable in the window and defaults to 0 (= the flag is not sent at all, upstream's own
# default).  Raise it when a run dies on «segment 1/1 (graph) failed during workspace capacity
# check»: the reserve must also cover what the desktop, the browser, the CUDA context and the
# CUDA VMM pool left behind by earlier phases hold, and only a negative --max-vram accounts for
# that invisible usage (upstream issue #2042).  1 GiB proved too small on this card.
MAX_VRAM_RESERVE_DEFAULT = 0
MAX_VRAM_RESERVE_LIMIT = 16
# Qwen-Image 2.1 only: the runtime first plans a graph that includes the prefix cache, that
# plan cannot be satisfied on an 8 GB card, and only then does it drop the cache and retry.
# Disabling the cache up front skips the doomed first attempt.  It is a model argument, so it
# has to be spelled as one, with the strict boolean the backend parses.
PREFIX_CACHE_FLAG = ('--model-args', 'qwen_image_2_1_prefix_cache=false')
PREFIX_CACHE_KEY = 'qwen_image_2_1_prefix_cache'
# The runtime's own wording (--vae-on-cpu is deprecated in favour of the assignment form):
#   --backend <string>  runtime backend assignment, e.g. cpu or clip=cpu,vae=cuda0,diffusion=vulkan0
# Running the VAE on the CPU frees the decode peak from the card; tiling keeps the decode on the
# GPU but in smaller pieces.  Both trade speed for VRAM, and both are checked at launch time.
VAE_CPU_FLAG = ('--backend', 'vae=cpu')
VAE_TILING_FLAG = '--vae-tiling'
# sd-server lists every upscaler model it can find in this directory in /sdapi/v1/upscalers,
# which is exactly what the page's «放大方法» dropdown reads: drop ESRGAN/Real-ESRGAN weights
# (e.g. RealESRGAN_x4plus_anime_6B.pth) in here and they become selectable for both 文字生圖 and
# 圖生圖 Hi-res.  Pointing the runtime at the directory is a startup flag, and the values are
# local paths, so the window owns the flag: a hand-written spelling is lifted into the field
# instead of being refused.
HIRES_UPSCALERS_DIR_FLAG = '--hires-upscalers-dir'
HIRES_UPSCALERS_DIR_DEFAULT = 'IMAGE-MODELS/upscalers'
# Flash attention ships two independent switches in sd-server:
#   --fa            use flash attention (the whole pipeline: text encoder, diffusion model, VAE)
#   --diffusion-fa  use flash attention in the diffusion model only
# The wider one is what relieves the text-encoder/VAE phases (the real 8 GB peak on this card),
# so it gets its own checkbox that defaults on; --diffusion-fa stays as the conservative
# narrow switch.  They never override each other — both are forwarded verbatim.
FA_FLAG = '--fa'
# Reference-image preprocessing.  img2img / qwen_edit re-encode the source image through the
# VAE, and an oversized input makes that encode allocate a huge activation and die on a small
# card.  sd-server takes a key-value list here, and this project's tested value caps the VAE
# input via the qwen preset at 800000 pixels.  Kept as a free-form string so any
# backend key can be appended, with the default shown and a one-click restore next to it.
REF_IMAGE_ARGS_FLAG = '--ref-image-args'
REF_IMAGE_ARGS_DEFAULT = 'preset=qwen,vae_input_max_pixels=800000'


def valid_vram_reserve(value):
    """GiB reserved for the desktop: 0＝不送旗標，0.1～16 送 --max-vram -N（可小數，如 1.5）."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return False
    text = str(value)
    whole, dot, fraction = text.partition('.')
    if not whole.isdigit() or len(whole) > 2:
        return False
    if dot and not (fraction.isdigit() and len(fraction) <= 2):
        return False
    return 0 <= float(text) <= MAX_VRAM_RESERVE_LIMIT


def normalise_vram_reserve(value):
    """Settings may still hold the old on/off flag; map it onto the editable number."""
    if isinstance(value, bool):
        return 1 if value else MAX_VRAM_RESERVE_DEFAULT
    if not valid_vram_reserve(value):
        return MAX_VRAM_RESERVE_DEFAULT
    number = float(value)
    return int(number) if number.is_integer() else number


def build_vram_reserve_arg(value):
    """('--max-vram', '-N'), or () when the user keeps the whole card for the runtime.

    The runtime parses the budget as a float, so a fractional reserve like 1.5 is sent as
    «-1.5» (it keeps 1.5 GiB free and budgets the rest).
    """
    number = normalise_vram_reserve(value)
    return ('--max-vram', f'-{number}') if number else ()


def migrate_vae_flags(text):
    """Lift hand-written VAE flags out of 「額外指令」 into the two checkboxes.

    Returns (remaining extra args, vae_on_cpu, vae_tiling).  Anything that is not one of the
    VAE spellings (--vae-tiling, --vae-on-cpu, --backend vae=cpu) is left exactly as typed, and
    the text is only rewritten when a flag was actually found.
    """
    source = str(text or '').split()
    on_cpu = tiling = False
    remaining = []
    index = 0
    while index < len(source):
        token = source[index]
        lowered = token.lower()
        if lowered == VAE_TILING_FLAG:
            tiling = True
        elif lowered == '--vae-on-cpu':
            on_cpu = True
        elif lowered == '--backend' and index + 1 < len(source) and source[index + 1].replace(' ', '') == 'vae=cpu':
            on_cpu = True
            index += 1
        elif lowered.startswith('--backend=') and lowered.split('=', 1)[1] == 'vae=cpu':
            on_cpu = True
        else:
            remaining.append(token)
        index += 1
    if not (on_cpu or tiling):
        return str(text or ''), False, False  # untouched: quoting of other values stays intact
    return ' '.join(remaining), on_cpu, tiling


def migrate_fa_flag(text):
    """Lift a hand-written --fa out of 「額外指令」 into the new 全流程 checkbox.

    Returns (remaining extra args, found).  --fa is a boolean switch with no value, so the
    whole token is removed; everything else is left exactly as typed, and the text is only
    rewritten when the flag was actually found.
    """
    source = str(text or '').split()
    remaining = [token for token in source if token.lower() != FA_FLAG]
    found = len(remaining) != len(source)
    return (' '.join(remaining) if found else str(text or '')), found


def migrate_hires_upscalers_dir(text):
    """Lift a hand-written --hires-upscalers-dir out of 「額外指令」 into its own field.

    Returns (remaining extra args, directory).  Both spellings are recognised
    (--hires-upscalers-dir DIR and --hires-upscalers-dir=DIR); everything else is left exactly
    as typed, and the text is only rewritten when the flag was actually found.
    """
    source = str(text or '').split()
    found = ''
    remaining = []
    index = 0
    while index < len(source):
        token = source[index]
        lowered = token.lower()
        if lowered == HIRES_UPSCALERS_DIR_FLAG:
            if index + 1 < len(source):
                found = source[index + 1]
                index += 1
        elif lowered.startswith(HIRES_UPSCALERS_DIR_FLAG + '='):
            found = token.split('=', 1)[1]
        else:
            remaining.append(token)
        index += 1
    if not found:
        return str(text or ''), ''  # untouched: quoting of other values stays intact
    return ' '.join(remaining), found


def valid_ref_image_args(value):
    """--ref-image-args takes a comma-separated key=value list; say what cannot be one.

    Empty is legal and means «do not send the flag».  A flag name (the field/flag mix-up)
    or a line break cannot be an argv value, so both are refused.  The keys belong to the
    backend, so they are deliberately not whitelisted: a typo must fail loudly at launch
    instead of being silently dropped here.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return True
    if looks_like_flag(text) or any(char in text for char in (chr(13), chr(10))):
        return False
    parts = [part.strip() for part in text.split(",")]
    return all(parts) and all("=" in part for part in parts)


def migrate_ref_image_args(text):
    """Lift a hand-written --ref-image-args out of 「額外指令」 into its own field.

    Both spellings are recognised (--ref-image-args VALUE and --ref-image-args=VALUE) and the
    value may be quoted, because it always carries '=' and often spaces as well.  A value that
    starts with '-' is treated as the next flag rather than a value, so a bare
    «--ref-image-args --fa» never swallows the following switch.  Returns (remaining extra
    args, value); the text is only rewritten when the flag was actually found.
    """
    source = str(text or "")
    flag = REF_IMAGE_ARGS_FLAG
    index = source.find(flag)
    while index != -1 and index > 0 and not source[index - 1].isspace():
        index = source.find(flag, index + 1)  # only a whole token counts
    if index == -1:
        return source, ""
    before = source[:index].rstrip()
    rest = source[index + len(flag):]
    if rest.startswith("="):
        rest = rest[1:]
    else:
        rest = rest.lstrip(" \t")
    if rest[:1] in ('"', "'"):
        quote = rest[0]
        close = rest.find(quote, 1)
        value, after = (rest[1:close], rest[close + 1:]) if close != -1 else (rest[1:], "")
    else:
        tokens = rest.split(None, 1)
        if tokens and not tokens[0].startswith("-"):
            value = tokens[0]
            after = tokens[1] if len(tokens) > 1 else ""
        else:
            value, after = "", rest  # bare flag: keep the next switch intact
    return " ".join((before + " " + after).split()), value


def looks_like_flag(value):
    """True for '--flag' / '-f' / '--flag=value' — a flag name, not a path or directory."""
    text = str(value or '').strip()
    return text.startswith('-') and len(text) > 1


def sanitise_hires_dir(saved_value, legacy_from_extra, base):
    """Decide what the «Hi-res 放大器目錄» field should hold when the window opens.

    A flag name pasted into the field (a very easy mistake, and one this project used to
    keep verbatim) is not a directory: it used to turn into '<base>/--hires-upscalers-dir'
    and blocked every launch, while the window saved it straight back so it never healed.
    Such a value is replaced by the conventional folder when that folder exists, otherwise
    by empty (no flag sent).  Real paths — including typos — are left exactly as typed, so
    a mistake still fails loudly at launch instead of being silently rewritten.
    """
    base = pathlib.Path(base)
    default = HIRES_UPSCALERS_DIR_DEFAULT if (base / HIRES_UPSCALERS_DIR_DEFAULT).is_dir() else ''
    if isinstance(saved_value, str):
        value = saved_value.strip()
        if not value:
            return legacy_from_extra or default
        return default if looks_like_flag(value) else saved_value
    return legacy_from_extra or default


def split_extra_args(text):
    """Split the optional «額外指令» field into sd-server argv tokens.

    The field is appended verbatim (it is the user's own command line), so only quotes
    are interpreted: they group one value that contains spaces. Flags this window owns
    are refused with a reason, because silently overriding them breaks the page.
    """
    if text is None:
        return []
    if not isinstance(text, str):
        raise ValueError('額外指令必須是文字')
    if len(text) > MAX_EXTRA_ARGS:
        raise ValueError(f'額外指令請少於 {MAX_EXTRA_ARGS} 個字元')
    tokens, current, quote, started = [], '', None, False
    for char in text:
        if quote:
            if char == quote:
                quote = None
            else:
                current += char
            continue
        if char in ('"', "'"):
            quote, started = char, True
        elif char.isspace():
            if current or started:
                tokens.append(current)
                current, started = '', False
        else:
            current += char
            started = True
    if quote:
        raise ValueError('額外指令的引號沒有成對，請補上收尾引號')
    if current or started:
        tokens.append(current)
    if len(tokens) > MAX_EXTRA_TOKENS:
        raise ValueError(f'額外指令請少於 {MAX_EXTRA_TOKENS} 個參數')
    for token in tokens:
        if token.lower() in MANAGED_FLAGS:
            raise ValueError(f'{token} 由控制窗的欄位管理，不能寫在額外指令')
    return tokens


def hires_dir_warning(hires_upscalers_dir, base):
    """Why the folder cannot be used, or '' when it is fine.

    Returned instead of raised: a wrong folder only costs the external upscaler list, so the
    caller logs this and starts anyway.  Blocking the whole launch over a typo meant the
    window was unusable until the user guessed which field was wrong.
    """
    hires_dir = str(hires_upscalers_dir or '').strip()
    if not hires_dir:
        return ''
    if looks_like_flag(hires_dir):
        return (f'Hi-res 放大器目錄填的是旗標名稱（{hires_dir}），不是資料夾路徑；'
                f'已忽略此參數，改用內建 Lanczos／Nearest（預設資料夾為 {HIRES_UPSCALERS_DIR_DEFAULT}）')
    resolved = pathlib.Path(hires_dir)
    if not resolved.is_absolute():
        resolved = pathlib.Path(base) / resolved
    if not resolved.is_dir():
        return (f'Hi-res 放大器目錄不存在：{resolved}；已忽略 --hires-upscalers-dir 並照常啟動，'
                '網頁「放大方法」只會有內建 Lanczos／Nearest。'
                '請確認此欄，或建立該資料夾（放入 ESRGAN／Real-ESRGAN 的 .pth 權重）')
    return ''


def build_server_cmd(base, port, offload=True, runtime=DEFAULT_RUNTIME, models=None, vision=False,
                     cache_mode='off', diffusion_fa=False, fa=True, conditioning_cache_size='default',
                     spectrum_w=SPECTRUM_W_DEFAULT, extra_args='', max_vram_reserve=0,
                     prefix_cache_disabled=False, vae_on_cpu=False, vae_tiling=False,
                     hires_upscalers_dir='', cache_threshold=CACHE_THRESHOLD_DEFAULT,
                     cache_warmup=CACHE_WARMUP_DEFAULT, vae_relative_tile_size=VAE_RELATIVE_TILE_DEFAULT,
                     ref_image_args=''):
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
        # Older settings files could store the dropdown's display marker as part of the path.
        if value.endswith(NOT_FOR_FIELD_MARK):
            value = value[:-len(NOT_FOR_FIELD_MARK)].rstrip()
        if value:
            cmd.extend((flag, model_path(base, value)))
    cmd.extend(('--listen-ip', '127.0.0.1', '--listen-port', str(port),
                '--serve-html-path', str(base / 'assets' / 'image-web.html')))
    if cache_mode not in CACHE_MODES:
        raise ValueError('快取模式僅支援：' + '、'.join(CACHE_MODES))
    if cache_mode != 'off':
        cmd.extend(('--cache-mode', cache_mode))
        if cache_mode == 'spectrum':
            if not valid_spectrum_w(spectrum_w):
                raise ValueError('Spectrum W 值必須介於 0.05～1.0')
            cmd.extend(('--cache-option', 'w=' + format_spectrum_w(spectrum_w)))
        elif cache_mode in CACHE_OPTION_MODES:
            # dbcache／taylorseer／cache-dit 共用 threshold＋warmup；兩個都留空＝不送，
            # 交給後端預設。只填一個也可以。
            if not valid_cache_threshold(cache_threshold):
                raise ValueError('區塊快取 threshold 請留空，或填 0～100 的數字（建議 0.25）')
            if not valid_cache_warmup(cache_warmup):
                raise ValueError('區塊快取 warmup 請留空，或填 0～50 的整數（建議 4）')
            options = []
            if str(cache_threshold).strip():
                options.append('threshold=' + format_cache_threshold(cache_threshold))
            if str(cache_warmup).strip():
                options.append('warmup=' + str(cache_warmup).strip())
            if options:
                cmd.extend(('--cache-option', ','.join(options)))
    if not isinstance(diffusion_fa, bool):
        raise ValueError('Flash Attention 必須為開啟或關閉')
    if diffusion_fa:
        cmd.append('--diffusion-fa')
    # --fa is the wider switch (whole pipeline) and defaults on; --diffusion-fa above is the
    # narrow one.  Both are independent flags, never alternatives.
    if not isinstance(fa, bool):
        raise ValueError('--fa（全流程 Flash Attention）必須為開啟或關閉')
    if fa:
        cmd.append(FA_FLAG)
    # Optional Viggle adapter is never applied at startup; the native API
    # supplies it per task.  Keep the original safetensors untouched.
    viggle = base / 'IMAGE-MODELS' / 'loras' / 'Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128-fused-gguf.safetensors'
    if viggle.is_file():
        cmd.extend(['--lora-model-dir', str(viggle.parent)])
    if not valid_conditioning_cache_size(conditioning_cache_size):
        raise ValueError('Conditioning cache 請輸入 default 或 0～2147483647 的整數')
    if conditioning_cache_size != 'default':
        cmd.extend(('--conditioning-cache-size', conditioning_cache_size))
    if offload:
        cmd.append('--offload-to-cpu')
    # VAE decode is the other peak behind a low-VRAM failure: it can run on the CPU
    # (the runtime's own wording for the deprecated --vae-on-cpu) or in tiles.
    for label, value in (('VAE 在 CPU 上執行', vae_on_cpu), ('VAE 分塊', vae_tiling)):
        if not isinstance(value, bool):
            raise ValueError(f'{label}必須為開啟或關閉')
    if vae_on_cpu:
        cmd.extend(VAE_CPU_FLAG)
    if vae_tiling:
        cmd.append(VAE_TILING_FLAG)
        # 分塊大小才是解碼速度的關鍵：後端預設 256x256 在 1024x1536 上要切約 30 塊，
        # 每塊的固定開銷＋邊緣重算把解碼拖到約 26 秒；0.5（每塊 512x768）只要 6 塊，
        # 解碼降到約 9 秒且逐像素相同。留空＝後端預設（不動）。
        tile_size = str(vae_relative_tile_size or '').strip()
        if tile_size:
            if not valid_vae_relative_tile_size(tile_size):
                raise ValueError('VAE 分塊相對大小請留空，或填 0.01～1.0（如 0.5），'
                                 '或明確尺寸（如 512x768）')
            cmd.extend(('--vae-relative-tile-size', tile_size))
    # Reference-image preprocessing (img2img / qwen_edit): a key-value list.  Empty＝不送，
    # 後端用預設；填了不是 key=value 的東西會擋下啟動，因為那一定不是可用的 argv 值。
    ref_image = str(ref_image_args or '').strip()
    if not valid_ref_image_args(ref_image):
        raise ValueError('--ref-image-args 請留空，或填 key=value 的逗號清單'
                         f'（預設 {REF_IMAGE_ARGS_DEFAULT}）')
    if ref_image:
        cmd.extend((REF_IMAGE_ARGS_FLAG, ref_image))
    # Hi-res upscalers are discovered by directory at launch time, and the page's 放大方法
    # dropdown is fed by /sdapi/v1/upscalers, so an unwired directory looks like "the model
    # does nothing".  Relative values are resolved against the project root (the same rule the
    # model fields follow).  A wrong value is dropped, not fatal: sd-server ignores a bad
    # directory anyway, so the launch proceeds and the caller logs the reason.
    hires_dir = str(hires_upscalers_dir or '').strip()
    if hires_dir and not hires_dir_warning(hires_dir, base):
        resolved_hires = pathlib.Path(hires_dir)
        if not resolved_hires.is_absolute():
            resolved_hires = base / resolved_hires
        cmd.extend((HIRES_UPSCALERS_DIR_FLAG, str(resolved_hires)))
    # Optional per-device VRAM budget.  The reserve is editable and 0 means "send nothing",
    # because the reserve must also cover the desktop/browser/CUDA context already on the
    # card: too small a reserve leaves the budget above one uncut segment's estimate,
    # so the runtime plans a monolithic segment and dies on the workspace check.
    if not valid_vram_reserve(max_vram_reserve):
        raise ValueError(f'保留顯存請填 0～{MAX_VRAM_RESERVE_LIMIT} 的數字，可用小數（例如 1.5）；'
                         '0＝不送 --max-vram')
    if not isinstance(prefix_cache_disabled, bool):
        raise ValueError('停用 prefix cache 必須為開啟或關閉')
    reserve_flag = build_vram_reserve_arg(max_vram_reserve)
    if reserve_flag:
        cmd.extend(reserve_flag)
    if prefix_cache_disabled:
        cmd.extend(PREFIX_CACHE_FLAG)
    if extra_args:
        tokens = split_extra_args(extra_args)
        if reserve_flag and any(token.lower() == '--max-vram' for token in tokens):
            raise ValueError('--max-vram 已在額外指令填寫；請把「保留顯存」改成 0，或清掉該參數')
        if prefix_cache_disabled and any(token.lower() == '--model-args' for token in tokens):
            raise ValueError('--model-args 已在額外指令填寫；請取消「停用 prefix cache」，或改用該欄自行指定')
        if vae_on_cpu and any(token.lower() in ('--backend', '--vae-on-cpu') or
                              token.lower().startswith('--backend=') for token in tokens):
            raise ValueError('--backend 已在額外指令填寫；請取消「VAE 在 CPU 上執行」，或改用該欄自行指定')
        if vae_tiling and any(token.lower() in (VAE_TILING_FLAG, '--vae-relative-tile-size', '--vae-tile-size')
                              for token in tokens):
            raise ValueError('--vae-tiling／--vae-relative-tile-size 已在額外指令填寫；'
                             '請取消「VAE 分塊」，或改用該欄自行指定')
        if hires_dir and any(token.lower() == HIRES_UPSCALERS_DIR_FLAG or
                             token.lower().startswith(HIRES_UPSCALERS_DIR_FLAG + '=') for token in tokens):
            raise ValueError(f'{HIRES_UPSCALERS_DIR_FLAG} 已在額外指令填寫；請清空「Hi-res 放大器目錄」欄位'
                             '改用它，或清掉額外指令中的該參數')
        if ref_image and any(token.lower() == REF_IMAGE_ARGS_FLAG or
                             token.lower().startswith(REF_IMAGE_ARGS_FLAG + '=') for token in tokens):
            raise ValueError(f'{REF_IMAGE_ARGS_FLAG} 已在額外指令填寫；請清空「參考圖參數」欄位'
                             '改用它，或清掉額外指令中的該參數')
        cmd.extend(tokens)
    return cmd


def build_cmd_from_settings(base, port, settings, vision=False, offload=None):
    """Launch command from a settings snapshot; start() and the mode switch share this.

    Both paths must produce the same command, otherwise a mode switch would silently drop
    the extra arguments or the VRAM reserve the user configured before starting.

    The editable reserve goes through untouched when it is present, so a malformed field is
    refused by build_server_cmd instead of silently falling back to the default; only legacy
    on/off settings (and snapshots that predate the number) are normalised.

    VAE tiling is the one switch that defaults to on: a snapshot without the key predates the
    checkbox, so it follows the window's default instead of silently disabling tiling.
    """
    return build_server_cmd(
        base, port,
        offload=settings.get('offload', True) if offload is None else offload,
        runtime=settings.get('runtime', DEFAULT_RUNTIME),
        models=settings.get('models'),
        vision=vision,
        cache_mode=settings.get('cache_mode', 'off'),
        diffusion_fa=settings.get('diffusion_fa', False),
        # --fa 預設開啟（全流程加速；只有存檔明確寫 False 才關掉），與 --diffusion-fa 並存。
        fa=settings.get('fa') is not False,
        conditioning_cache_size=settings.get('conditioning_cache_size', 'default'),
        spectrum_w=settings.get('spectrum_w', SPECTRUM_W_DEFAULT),
        extra_args=settings.get('extra_args', ''),
        max_vram_reserve=settings.get('max_vram_reserve_gib',
                                      normalise_vram_reserve(settings.get('max_vram_reserve'))),
        prefix_cache_disabled=settings.get('prefix_cache_disabled') is True,
        vae_on_cpu=settings.get('vae_on_cpu') is True,
        # 分塊解碼預設開啟（和視窗的勾選框預設一致）；只有存檔明確寫 False 才關掉，
        # 這樣「切換模式重啟」不會把使用者啟動時用的同一組參數悄悄改掉。
        vae_tiling=settings.get('vae_tiling') is not False,
        hires_upscalers_dir=settings.get('hires_upscalers_dir', ''),
        # 舊設定檔沒有這兩個欄位時沿用保守預設（dbcache 才會用到；off／spectrum 忽略）。
        cache_threshold=settings.get('cache_threshold', CACHE_THRESHOLD_DEFAULT),
        cache_warmup=settings.get('cache_warmup', CACHE_WARMUP_DEFAULT),
        vae_relative_tile_size=settings.get('vae_relative_tile_size', VAE_RELATIVE_TILE_DEFAULT),
        # 參考圖參數：舊存檔沒有這個欄位時沿用實測預設（preset=qwen,vae_input_max_pixels=800000），
        # 這樣「切換模式重啟」不會把圖生圖需要的防爆顯存設定悄悄拿掉。
        ref_image_args=settings.get('ref_image_args', REF_IMAGE_ARGS_DEFAULT))


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


# sd-server applies a LoRA into the running process (quantized weights force the
# at_runtime mode) and never gives those weights back, so a LoRA that was used once stays
# resident until the process is replaced. The log is truncated at every launch, so this
# marker in the current log means «this session already applied one».
LORA_RESIDENT_MARKER = 'apply_loras completed'


def lora_resident(log_path, since=0, limit=262144):
    """True when the running server session already applied a LoRA (memory still held).

    Only bytes written after ``since`` count: a restart appends to the same log file, so a
    marker left by the previous session must not keep the flag (and the page's button) set.
    """
    try:
        with open(log_path, 'rb') as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            start = max(since, size - limit)
            if start >= size:
                return False
            stream.seek(start)
            text = stream.read().decode('utf-8', 'replace')
    except OSError:
        return False
    return LORA_RESIDENT_MARKER in text


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


# Image 模式的參數模板（具名參數組合）與模型綁定，獨立一份檔案，
# 和 LLM 模式的 presets.json 分開，兩邊互不干擾。
PRESETS_FILE = BASE / 'image-presets.json'


def default_presets():
    """首次啟動的內建模板：本機實測過的兩條路線（見 qwen-speed-report.md）。"""
    return {
        'templates': {
            '20步 高品質（tile0.5+dbcache）': {
                'offload': True, 'cache_mode': 'dbcache',
                'cache_threshold': '0.25', 'cache_warmup': '4',
                'vae_tiling': True, 'vae_relative_tile_size': '0.5',
                'diffusion_fa': True, 'conditioning_cache_size': 'default',
            },
            '6步 LoRA 快速（tile0.5 無cache）': {
                'offload': True, 'cache_mode': 'off',
                'cache_threshold': '', 'cache_warmup': '',
                'vae_tiling': True, 'vae_relative_tile_size': '0.5',
                'diffusion_fa': True, 'conditioning_cache_size': 'default',
            },
            '最保守（僅 tile0.5 無損）': {
                'offload': True, 'cache_mode': 'off',
                'cache_threshold': '', 'cache_warmup': '',
                'vae_tiling': True, 'vae_relative_tile_size': '0.5',
                'diffusion_fa': False, 'conditioning_cache_size': 'default',
            },
        },
        'bind': {},
        'default': '',
    }


def load_presets(path=None):
    """讀模板檔；不存在或壞掉時回傳內建預設（不寫檔，第一次按儲存才落地）。"""
    path = pathlib.Path(PRESETS_FILE if path is None else path)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default_presets()
    if not isinstance(data, dict):
        return default_presets()
    for key, fallback in (('templates', {}), ('bind', {})):
        if not isinstance(data.get(key), dict):
            data[key] = fallback
    if not isinstance(data.get('default'), str):
        data['default'] = ''
    return data


def save_presets(data, path=None):
    path = pathlib.Path(PRESETS_FILE if path is None else path)
    try:
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(tmp, path)
        return True
    except OSError:
        return False


class ImageApp:
    def __init__(self, root):
        self.root = root
        # Guard: every widget is built and restored before any trace is allowed to act on
        # user data. Cleared at the end of __init__, which then applies the model's default
        # template — the same "force on boot" pass gguf-ui.py does.
        self._loading = True
        self.proc = None
        self.logfile = None
        self.save_service = None
        self.mode_lock = threading.Lock()
        self.vision_loaded = False
        self.switching = False
        self.mode_error = ''
        self.switch_started = 0
        self.restart_note = ''
        # Log offset of the newest launch: LoRA markers before it belong to an older session.
        self.launch_offset = 0
        saved = load_settings()
        self.port = tk.StringVar(value=str(saved.get('port', '18436')))
        self.offload = tk.BooleanVar(value=saved.get('offload', True))
        # mode_status() runs on the save service's HTTP thread, which must never touch Tk
        # variables (RuntimeError: main thread is not in main loop) — that crashed the /mode
        # handler, and an unanswered /mode left the page's 「♻ 釋放 LoRA 記憶體」 disabled.
        self.offload_snapshot = bool(self.offload.get())
        self.offload.trace_add('write', self.sync_offload_snapshot)
        self.cache_mode = tk.StringVar(value=saved.get('cache_mode') if saved.get('cache_mode') in CACHE_MODES else 'off')
        self.spectrum_w = tk.StringVar(value=saved.get('spectrum_w')
            if valid_spectrum_w(saved.get('spectrum_w')) else SPECTRUM_W_DEFAULT)
        self.diffusion_fa = tk.BooleanVar(value=saved.get('diffusion_fa') is True)
        # --fa 全流程 Flash Attention：預設開啟，只有存檔明確寫 False 才關掉。
        self.fa = tk.BooleanVar(value=saved.get('fa') is not False or _legacy_fa)
        self.conditioning_cache_size = tk.StringVar(value=saved.get('conditioning_cache_size')
            if valid_conditioning_cache_size(saved.get('conditioning_cache_size')) else 'default')
        _saved_extra = saved.get('extra_args')
        _saved_extra, _legacy_vae_cpu, _legacy_vae_tiling = migrate_vae_flags(
            _saved_extra if isinstance(_saved_extra, str) and len(_saved_extra) <= MAX_EXTRA_ARGS else '')
        _saved_extra, _legacy_fa = migrate_fa_flag(_saved_extra)
        _saved_extra, _legacy_hires_dir = migrate_hires_upscalers_dir(_saved_extra)
        _saved_extra, _legacy_ref_image = migrate_ref_image_args(_saved_extra)
        self.extra_args = tk.StringVar(value=_saved_extra)
        # 參考圖參數（--ref-image-args）：存檔有合法值（含使用者刻意留空）就照用，否則沿用實測預設，
        # 或吃下從「額外指令」搬過來的舊寫法。留空＝不送此參數。
        _saved_ref = saved.get('ref_image_args')
        if not (isinstance(_saved_ref, str) and valid_ref_image_args(_saved_ref)):
            _saved_ref = _legacy_ref_image or REF_IMAGE_ARGS_DEFAULT
        self.ref_image_args = tk.StringVar(value=_saved_ref)
        self.max_vram_reserve = tk.StringVar(value=str(normalise_vram_reserve(
            saved.get('max_vram_reserve_gib', saved.get('max_vram_reserve')))))
        self.prefix_cache_disabled = tk.BooleanVar(value=saved.get('prefix_cache_disabled') is True)
        # Hand-written VAE flags from the free-form field become the new checkboxes.
        self.vae_on_cpu = tk.BooleanVar(value=saved.get('vae_on_cpu') is True or _legacy_vae_cpu)
        # 分塊解碼是新安裝的預設：解碼階段是低顯存爆點的第二個高峰，而它對畫質幾乎無損。
        # 只有使用者明確取消（存檔裡是 False）才維持關閉；舊設定檔沒有這個欄位時視為預設開啟，
        # 不要因為升級而靜默變成關閉。
        self.vae_tiling = tk.BooleanVar(value=saved.get('vae_tiling') is not False or _legacy_vae_tiling)
        # 區塊級快取的 threshold／warmup：留空＝不送，交給後端預設。
        self.cache_threshold = tk.StringVar(value=saved.get('cache_threshold')
            if valid_cache_threshold(saved.get('cache_threshold')) else CACHE_THRESHOLD_DEFAULT)
        self.cache_warmup = tk.StringVar(value=saved.get('cache_warmup')
            if valid_cache_warmup(saved.get('cache_warmup')) else CACHE_WARMUP_DEFAULT)
        # VAE 分塊相對大小：留空＝後端預設 256x256（實測最慢）。
        self.vae_relative_tile_size = tk.StringVar(value=saved.get('vae_relative_tile_size')
            if valid_vae_relative_tile_size(saved.get('vae_relative_tile_size')) else VAE_RELATIVE_TILE_DEFAULT)
        # 模板（image-presets.json）：切模型自動套用；_tpl_busy 防止套用時又被 trace 存回去。
        self._presets = load_presets()
        self._tpl_busy = False
        self._tpl_note = ''
        self._tpl_model_key = ''
        # Auto-save-back state: hand-tuning a value while a template is selected writes the
        # change into that template, exactly like gguf-ui.py's save_now -> _tpl_store_current.
        self._tpl_dirty = False
        self._tpl_save_job = None
        # The upscaler directory is pre-filled with the conventional folder only when it exists,
        # so a fresh install never refuses to start over a directory that was never created;
        # a hand-written --hires-upscalers-dir in 「額外指令」 wins over that default, and a flag
        # name pasted into the field is repaired instead of joining onto the project root.
        self.hires_upscalers_dir = tk.StringVar(
            value=sanitise_hires_dir(saved.get('hires_upscalers_dir'), _legacy_hires_dir, BASE))
        self.runtime = tk.StringVar(value=saved.get('runtime', DEFAULT_RUNTIME))
        previous = saved.get('models', {}) if isinstance(saved.get('models'), dict) else {}
        self.models = {key: tk.StringVar(value=previous.get(key, FILES.get(key, '')))
                       for key, _, _, _ in FIELDS}
        root.title('GGUFRun — Image 模式（獨立 sd-server）')
        # 視窗不要開得比螢幕大：設定區在捲動畫布裡，本來就捲得到底，
        # 但視窗本身超出畫面高度時，底下的「啟動」列與日誌會被螢幕切掉、
        # 連拖曳都救不回來（高 DPI 螢幕上特別容易發生）。
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f'{min(960, max(640, screen_w - 40))}x{min(880, max(560, screen_h - 120))}')
        root.minsize(min(860, screen_w), min(600, screen_h))
        # 外觀維持 ttk 的細線（跟 gguf-ui.py 一致），但「可抓範圍」另外用程式放大：
        # vista 主題的 sash 只有 6px，而且可見線比可抓區高 3px，才會「要按上面空白」。
        # 下面用 widget 層的 <Button-1>（比 class binding 先跑）攔截 ±9px 的範圍，
        # 自己實作拖曳並 return 'break' 擋掉原生判定 → 細線外觀 + 18px 手感。
        pane = ttk.Panedwindow(root, orient='vertical')
        pane.pack(fill='both', expand=True)
        # The configuration block can be taller than its slot: it lives in a scrollable canvas
        # instead of being clipped (a Panedwindow silently clipped its last children, which hid
        # 「▶ 啟動 Image Server」), and the action bar is pinned underneath it — directly above the
        # log pane, so it is always reachable without scrolling.
        config_slot = ttk.Frame(pane)
        pane.add(config_slot, weight=3)
        actionbar = ttk.Frame(config_slot, padding=(14, 2, 14, 8))
        actionbar.pack(side='bottom', fill='x')
        canvas_host = ttk.Frame(config_slot)
        canvas_host.pack(side='left', fill='both', expand=True)
        canvas = tk.Canvas(canvas_host, highlightthickness=0, height=700)
        vsb = ttk.Scrollbar(canvas_host, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        frame = ttk.Frame(canvas, padding=14)
        canvas_window = canvas.create_window((0, 0), window=frame, anchor='nw')
        frame.bind('<Configure>',
                   lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda event: canvas.itemconfigure(canvas_window, width=event.width))

        def scroll_config(event):
            """Wheel anywhere inside the config form scrolls the form, and NEVER edits the
            widget under the pointer. Binding this on each child widget (not only on the
            root) matters: a ttk.Combobox has a class-level wheel binding that cycles its
            value, and a class binding runs before a bind_all one — so only a widget-level
            binding returning 'break' suppresses it. gguf-ui.py does the same in _bind_wheel.
            """
            try:
                canvas.yview_scroll(-1 if getattr(event, 'delta', 0) > 0 else 1, 'units')
            except tk.TclError:
                return None
            return 'break'

        def _resync_scroll():
            """表單高度變了（折疊開合）就重算捲動範圍。"""
            try:
                canvas.configure(scrollregion=canvas.bbox('all'))
            except tk.TclError:
                pass

        def _bind_wheel_tree(widget):
            """Recursively attach the wheel handler to the form and every descendant."""
            try:
                widget.bind('<MouseWheel>', scroll_config)
                for _child in widget.winfo_children():
                    _bind_wheel_tree(_child)
            except tk.TclError:      # widget destroyed while walking
                pass

        def _on_wheel_any(event):
            """Fallback for widgets created after the initial walk: only act when the pointer
            is inside the scrolling form, so the日志 below keeps its own wheel behaviour."""
            try:
                widget = root.winfo_containing(event.x_root, event.y_root)
                while widget is not None:
                    if widget is canvas:
                        return scroll_config(event)
                    widget = getattr(widget, 'master', None)
            except tk.TclError:
                return None
            return None

        root.bind_all('<MouseWheel>', _on_wheel_any)

        def fit_sash():
            """把設定區高度設成表單的自然高度，多出來的空間全給日誌。

            以前這裡只改 canvas 的 height 就放著，sash 停在 weight 分配的
            500px 讓設定區永遠多佔一格。改成量好表單後直接呼叫 sashpos 定位，
            日誌就拿到剩下的全部高度；之後使用者拖曳仍可自由調整。
            """
            try:
                # 先讓 Tk 把摺疊／展開造成的尺寸變化結算完，再讀高度。
                # 少了這行 update_idletasks()，pack_forget() 之後讀到的還是
                # 舊高度，分隔線就收不回來（使用者：「空位會一直存在」）。
                root.update_idletasks()
                canvas.configure(height=frame.winfo_reqheight())
                root.update_idletasks()
                want = canvas_host.winfo_reqheight() + actionbar.winfo_reqheight()
                top = pane.winfo_height()
                if top > 120:
                    # 設定區拿到表單需要的高度，剩下的都給日誌。
                    # 日誌的「預設大小」是靠視窗預設高度（700）控制的：700 高時
                    # 日誌只有 ~110px；以前視窗開 880，日誌一開就 418px，太佔位。
                    # 這裡不加硬上限——把多出來的高度硬塞回設定區，會在
                    # 動作列上方留下一大塊空白（見 fit_sash 上方的說明）。
                    pane.sashpos(0, max(140, min(want, top - 120)))
            except tk.TclError:  # window already closed
                pass

        ttk.Label(frame, text='Image 模式｜選擇 sd-server 與相容的模型組合；和 LLM 模式互不干擾',
                  font=('Microsoft JhengHei UI', 10)).pack(anchor='w', pady=(0, 9))
        row = ttk.Frame(frame)
        row.pack(fill='x', pady=3)
        ttk.Label(row, text='Image runtime：', width=18).pack(side='left')
        self.runtime_box = ttk.Combobox(row, textvariable=self.runtime, width=52)
        self.runtime_box.pack(side='left', fill='x', expand=True)
        ttk.Button(row, text='瀏覽 exe', command=self.browse_runtime).pack(side='left', padx=4)
        ttk.Button(row, text='重新掃描', command=self.refresh).pack(side='left')
        hint_line = ttk.Frame(frame)
        hint_line.pack(fill='x', pady=(5, 5))
        ttk.Label(hint_line, text='模型欄位可直接輸入路徑、從 IMAGE-MODELS 選取，或瀏覽外部檔案；不用的欄位留空。').pack(side='left')
        hint(hint_line, '下拉選單會列出 IMAGE-MODELS 內所有模型檔；標「✗ 不適用」的是該欄不能用的檔'
                        '（例如把 LoRA 選進 VAE 會讓 sd-server 啟動失敗）。欄位可自行輸入或清空，不必從選單挑。\n\n'
                        '所需編碼器以模型文件為準；選到檔案不代表相容。缺檔／不相容不預先攔截，由 sd-server 記錄錯誤；'
                        '遇到「model metadata validation failed」就是選錯檔案類型。').pack(side='left', padx=(4, 0))
        self.entries = {}
        # 常用的欄位直接攤開；CLIP-L／CLIP-G／T5XXL 很少動，
        # 收進折疊區塊，免得八列模型欄位把設定區撐高、日誌被擠掉。
        self._OPTIONAL_FIELDS = {'clip_l', 'clip_g', 't5xxl'}

        def _make_model_row(parent, key, label, explanation):
            row = ttk.Frame(parent)
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=label + '：', width=18).pack(side='left')
            cb = ttk.Combobox(row, textvariable=self.models[key], width=52)
            cb.pack(side='left', fill='x', expand=True)
            cb.bind('<<ComboboxSelected>>', lambda event, field=key: self.check_model_choice(field))
            self.entries[key] = cb
            ttk.Button(row, text='瀏覽', command=lambda field=key: self.browse_model(field)).pack(side='left', padx=4)
            ttk.Button(row, text='清空',
                       command=lambda field=key: self.models[field].set('')).pack(side='left')
            ttk.Label(row, text=explanation, foreground='#555').pack(side='left', padx=3)

        for key, flag, label, explanation in FIELDS:
            if key not in self._OPTIONAL_FIELDS:
                _make_model_row(frame, key, label, explanation)
        self.sec_opt = Section(frame, '其他編碼器（CLIP-L / CLIP-G / T5XXL）', expanded=False)
        for key, flag, label, explanation in FIELDS:
            if key in self._OPTIONAL_FIELDS:
                _make_model_row(self.sec_opt.body, key, label, explanation)
        self.refresh()
        # --- 參數模板（照 GGUF LLM 模式那套：一鍵套用＋另存／改名／刪除／設預設／綁定模型）---
        tplf = ttk.LabelFrame(frame, text='參數模板')
        tplf.pack(fill='x', pady=(6, 4))
        _tr = ttk.Frame(tplf)
        _tr.pack(fill='x', padx=6, pady=4)
        ttk.Label(_tr, text='模板：').pack(side='left')
        self.tpl = tk.StringVar(value=self._TPL_NONE)
        self.cb_tpl = ttk.Combobox(_tr, textvariable=self.tpl, width=26, state='readonly',
                                   values=[self._TPL_NONE])
        self.cb_tpl.pack(side='left', padx=(0, 6))
        self.cb_tpl.bind('<<ComboboxSelected>>', self._tpl_on_pick)
        ttk.Button(_tr, text='💾 另存', command=self._tpl_save_as).pack(side='left', padx=2)
        ttk.Button(_tr, text='✏ 改名', command=self._tpl_rename).pack(side='left', padx=2)
        ttk.Button(_tr, text='🗑 刪除', command=self._tpl_delete).pack(side='left', padx=2)
        ttk.Button(_tr, text='⭐ 設為預設', command=self._tpl_set_default).pack(side='left', padx=2)
        _tr2 = ttk.Frame(tplf)
        _tr2.pack(fill='x', padx=6, pady=(0, 4))
        ttk.Button(_tr2, text='🔗 綁定目前生圖模型', command=self._tpl_bind_model).pack(side='left', padx=2)
        ttk.Button(_tr2, text='✂ 解除綁定', command=self._tpl_unbind).pack(side='left', padx=2)
        self.lbl_tpl = ttk.Label(_tr2, text='', foreground='#666')
        self.lbl_tpl.pack(side='left', padx=10)
        hint(_tr2, '模板只存「怎麼跑」：offload／取樣加速／threshold／warmup／VAE 分塊／'
                   'Flash Attention／conditioning cache。不含埠、模型路徑本身。\n'
                   '切到「綁定」的模型會自動套用該模板；沒有綁定的模型套用「預設模板」。\n'
                   '手動改上面任何一個模板欄位，會自動存回目前選的模板（和 LLM 模式一致）。'
                   '\n\n另存＝把目前畫面參數存成新模板；設為預設＝沒綁定的模型都套用它。').pack(side='left', padx=(4, 0))

        options = ttk.Frame(frame)
        options.pack(fill='x', pady=5)
        ttk.Label(options, text='本機埠：').pack(side='left')
        self.port_entry = ttk.Entry(options, textvariable=self.port, width=9)
        self.port_entry.pack(side='left', padx=(0, 18))
        ttk.Checkbutton(options, text='Offload 到系統 RAM（8 GB 顯卡建議）', variable=self.offload).pack(side='left')
        # VAE 在 CPU 上執行：原本就攤在主表單（省顯存的主力開關之一），
        # 之前被我收進預設收合的「進階」區，使用者就找不到它了 —— 搬回來。
        self.vae_cpu_box = ttk.Checkbutton(options, text='VAE 在 CPU 上執行（--backend vae=cpu）',
                                           variable=self.vae_on_cpu)
        self.vae_cpu_box.pack(side='left', padx=(14, 0))
        hint(options, '把 VAE 解碼移出顯卡，用速度換顯存；解碼階段爆顯存、且分塊仍不夠時再開。\n'
                      '與「Offload 到系統 RAM」不同：Offload 是讓權重暫用系統 RAM，這顆是把 VAE 運算丟到 CPU。').pack(side='left', padx=(4, 0))
        # --- 取樣與加速 ---
        cache_row = ttk.Frame(frame)
        cache_row.pack(fill='x', pady=(0, 5))
        ttk.Label(cache_row, text='取樣加速：').pack(side='left')
        self.cache_box = ttk.Combobox(cache_row, textvariable=self.cache_mode, values=CACHE_MODES,
                                      state='readonly', width=14)
        self.cache_box.pack(side='left', padx=(0, 8))
        ttk.Label(cache_row, text='Spectrum W：').pack(side='left')
        self.spectrum_w_entry = ttk.Entry(cache_row, textvariable=self.spectrum_w, width=6)
        self.spectrum_w_entry.pack(side='left', padx=(0, 8))
        ttk.Label(cache_row, text='threshold：').pack(side='left')
        self.cache_threshold_entry = ttk.Entry(cache_row, textvariable=self.cache_threshold, width=6)
        self.cache_threshold_entry.pack(side='left', padx=(0, 8))
        ttk.Label(cache_row, text='warmup：').pack(side='left')
        self.cache_warmup_entry = ttk.Entry(cache_row, textvariable=self.cache_warmup, width=4)
        self.cache_warmup_entry.pack(side='left', padx=(0, 8))
        self.cache_hint = ttk.Label(cache_row, text='', wraplength=380)
        self.cache_hint.pack(side='left')
        self.cache_mode.trace_add('write', lambda *_: self.sync_cache_controls())
        hint(cache_row, 'off＝原版最清晰（不加速）。\n'
                        'spectrum＝外推跳步：W 越小越保畫質（建議 0.10；後端預設 0.40 較糊）。\n'
                        'dbcache＝DiT 區塊級快取：threshold 越大越愛跳步。8 GB 卡實測 0.25＋warmup 4'
                        '於 20 步省約 53 秒、肉眼幾乎無損；0.35 起細紋理明顯融掉，0.5 以上崩壞。\n'
                        'easycache＝最快但畫質降較多。taylorseer／cache-dit＝平衡（約省 8%）。\n'
                        'threshold／warmup 留空＝交給後端預設，只在 dbcache 類模式送出。\n'
                        '加速選項停止後再選、重啟生效。').pack(side='left', padx=(4, 0))
        speed_row = ttk.Frame(frame)
        speed_row.pack(fill='x', pady=(0, 5))
        self.fa_box = ttk.Checkbutton(speed_row, text='Flash Attention（僅擴散模型）',
                                      variable=self.diffusion_fa)
        self.fa_box.pack(side='left', padx=(0, 12))
        self.fa_all_box = ttk.Checkbutton(speed_row, text='Flash Attention（全流程，預設開啟）',
                                          variable=self.fa)
        self.fa_all_box.pack(side='left', padx=(0, 12))
        ttk.Label(speed_row, text='Conditioning cache：').pack(side='left')
        self.conditioning_box = ttk.Combobox(speed_row, textvariable=self.conditioning_cache_size,
                                            values=('default', '0', '4', '8'), state='normal', width=12)
        self.conditioning_box.pack(side='left', padx=(0, 8))
        hint(speed_row, '兩顆 Flash Attention 是 sd-server 的兩個獨立旗標，互不取代：\n'
                        '「僅擴散模型」送 --diffusion-fa：只在擴散模型用 Flash Attention（較保守）。\n'
                        '「全流程」送 --fa：文字編碼、擴散模型與 VAE 全流程都開（預設開啟）；\n'
                        '8 GB 卡上文字編碼／解碼階段正是顯存高峰，開這個通常最省顯存。兩者可同時勾選。\n\n'
                        'Conditioning cache：可手動輸入 default 或非負整數；default 沿用後端預設。\n'
                        '新版 runtime（master-929 起）支援，舊版 runtime 選數字會阻止啟動；'
                        '容量過大可能增加 RAM／顯存用量。').pack(side='left', padx=(4, 0))
        extra_row = ttk.Frame(frame)
        extra_row.pack(fill='x', pady=(0, 3))
        ttk.Label(extra_row, text='額外指令（選填）：').pack(side='left')
        self.extra_args_entry = ttk.Entry(extra_row, textvariable=self.extra_args, width=52)
        self.extra_args_entry.pack(side='left', padx=(0, 6))
        ttk.Button(extra_row, text='清空', command=lambda: self.extra_args.set('')).pack(side='left')
        hint(extra_row, '額外指令原樣接在啟動指令最後（留空＝不送；値含空格請用雙引號包住）。\n'
                        '控制窗已管理的參數（模型欄位、埠、--serve-html-path、--offload-to-cpu、'
                        '--cache-mode 等）會被擋下並說明原因。').pack(side='left', padx=(4, 0))
        ref_row = ttk.Frame(frame)
        ref_row.pack(fill='x', pady=(0, 3))
        ttk.Label(ref_row, text='參考圖參數（圖生圖用）：').pack(side='left')
        self.ref_image_entry = ttk.Entry(ref_row, textvariable=self.ref_image_args, width=42)
        self.ref_image_entry.pack(side='left', padx=(0, 6))
        self.ref_image_reset = ttk.Button(ref_row, text='還原預設',
                                          command=lambda: self.ref_image_args.set(REF_IMAGE_ARGS_DEFAULT))
        self.ref_image_reset.pack(side='left', padx=(0, 4))
        ttk.Button(ref_row, text='清空',
                   command=lambda: self.ref_image_args.set('')).pack(side='left')
        hint(ref_row, '送 --ref-image-args：設定圖生圖／Qwen 指令修圖時參考圖的處理方式。\n'
                      f'預設值 {REF_IMAGE_ARGS_DEFAULT}：preset=qwen 走 Qwen 專用前處理，\n'
                      'vae_input_max_pixels=800000 把 VAE 看到的輸入圖上限壓在約 80 萬像素，\n'
                      '避免原圖過大時在 VAE 編碼階段爆顯存（圖生圖最常見的爆點）。\n'
                      '可自由增減 key=value（逗號分隔），例如再調大上限或換 preset；\n'
                      '留空＝不送此參數（後端用預設）。改壞了按「還原預設」即可回復原值。\n'
                      '執行期間鎖定、重啟生效；若以前手寫在「額外指令」，載入時會自動搬進本欄。'
                      ).pack(side='left', padx=(4, 0))
        # 輸入框下方固定標注預設值（並在格式錯誤時當場轉紅），避免改壞了改不回來。
        self.ref_image_hint = ttk.Label(frame, text=f'預設值：{REF_IMAGE_ARGS_DEFAULT}', foreground='#555')
        self.ref_image_hint.pack(anchor='w', padx=6)
        self.ref_image_args.trace_add('write', lambda *_: self.sync_ref_image_hint())
        vae_row = ttk.Frame(frame)
        vae_row.pack(fill='x', padx=6, pady=(0, 3))
        ttk.Label(vae_row, text='VAE 解碼：').pack(side='left')
        # 「在 CPU 上執行」已搬回上方主表單（Offload 旁），這裡只留分塊相關。
        self.vae_tiling_box = ttk.Checkbutton(vae_row, text='分塊處理（預設開啟）',
                                              variable=self.vae_tiling)
        self.vae_tiling_box.pack(side='left', padx=(0, 10))
        ttk.Label(vae_row, text='分塊相對大小：').pack(side='left')
        self.vae_tile_size_entry = ttk.Entry(vae_row, textvariable=self.vae_relative_tile_size, width=7)
        self.vae_tile_size_entry.pack(side='left', padx=(0, 8))
        self.vae_tile_hint = ttk.Label(vae_row, text='', wraplength=430)
        self.vae_tile_hint.pack(side='left')
        hint(vae_row, '「分塊處理」送 --vae-tiling（VAE 仍在顯卡上，但切成小塊編碼，尖峰較低；預設已開啟）。\n'
                      '解碼階段爆顯存時先試分塊，仍不夠再開上方的「VAE 在 CPU 上執行」；也可搭配「保留顯存」。\n\n'
                      '「分塊相對大小」送 --vae-relative-tile-size：留空＝後端預設 256x256（實測最慢）；\n'
                      '填 1 以下的比例（0.5＝每塊約圖邊長一半，1024x1536 只需約 6 塊）可大幅加快解碼，且逐像素無損；\n'
                      '也可填明確尺寸如 512x768（比比例更好預測，因為比例會隨解析度變動）。需搭配「分塊處理」。\n'
                      '兩者執行期間鎖定，停止後才能改，重啟生效。').pack(side='left', padx=(4, 0))
        self.vae_relative_tile_size.trace_add('write', lambda *_: self.sync_vae_tile_controls())
        # --- 進階：顯存與 VAE 解碼（不常改，預設收合以免擠壓日誌）---
        self.sec_adv = Section(frame, '進階：顯存保留與 VAE 分塊', expanded=False)
        def _after_toggle():
            """折疊／展開後：重算捲動範圍，並把分隔線重新貼到表單底部。

            只做 _resync_scroll 的話，表單變矮了但分隔線還留在原地，
            設定區就會留一大塊空白（使用者：「空位會一直存在」）。
            """
            _resync_scroll()
            fit_sash()

        self.sec_adv.on_toggle = _after_toggle
        self.sec_opt.on_toggle = _after_toggle
        adv = self.sec_adv.body
        reserve_row = ttk.Frame(adv)
        reserve_row.pack(fill='x', padx=6, pady=(4, 3))
        ttk.Label(reserve_row, text='保留顯存（GiB）：').pack(side='left')
        self.max_vram_box = ttk.Spinbox(reserve_row, from_=0, to=MAX_VRAM_RESERVE_LIMIT, increment=0.5,
                                        width=4, textvariable=self.max_vram_reserve)
        self.max_vram_box.pack(side='left', padx=(0, 10))
        self.prefix_cache_box = ttk.Checkbutton(reserve_row, text='停用 prefix cache（僅 Qwen-Image 2.1）',
                                                variable=self.prefix_cache_disabled)
        self.prefix_cache_box.pack(side='left')
        hint(reserve_row, f'「保留顯存」填 N 送 --max-vram -N（{MAX_VRAM_RESERVE_DEFAULT}＝不送），'
                          '可填小數如 1.5。預算越小 graph cut 切得越細、較慢但不容易爆。\n'
                          '遇到 segment 1/1 (graph) failed during workspace capacity check 時建議 2～4，'
                          '因為桌面、瀏覽器與前階段留下的 CUDA VMM pool 不在後端追蹤內，只有負值會算進去。\n\n'
                          '「停用 prefix cache」送 --model-args qwen_image_2_1_prefix_cache=false，'
                          '跳過後端那份必定失敗的含快取計畫（取樣將無法重用快取）。\n'
                          '運行期間本欄鎖定，停止後才能改，重啟生效。').pack(side='left', padx=(4, 0))
        hires_row = ttk.Frame(adv)
        hires_row.pack(fill='x', padx=6, pady=(0, 5))
        ttk.Label(hires_row, text='Hi-res 放大器目錄：').pack(side='left')
        self.hires_dir_entry = ttk.Entry(hires_row, textvariable=self.hires_upscalers_dir, width=46)
        self.hires_dir_entry.pack(side='left', padx=(0, 6))
        self.hires_dir_button = ttk.Button(hires_row, text='選擇…', command=self.choose_hires_dir)
        self.hires_dir_button.pack(side='left', padx=(0, 4))
        self.hires_dir_clear = ttk.Button(hires_row, text='清空',
                                          command=lambda: self.hires_upscalers_dir.set(''))
        self.hires_dir_clear.pack(side='left')
        hint(hires_row, '送 --hires-upscalers-dir：sd-server 會把這個資料夾裡找到的放大模型'
                        '（ESRGAN／Real-ESRGAN 的 .pth，例如 RealESRGAN_x4plus_anime_6B.pth）'
                        '列進 /sdapi/v1/upscalers，網頁「進階設定：Hi-res 高解析度」的「放大方法」下拉就會出現它們，'
                        '文字生圖與圖生圖都可用。\n'
                        '留空＝不送此參數（只剩內建 Lanczos／Nearest）；填了不存在的資料夾會阻止啟動。\n'
                        '預設 IMAGE-MODELS\\upscalers（該資料夾不存在時留空）；'
                        '若你以前把 --hires-upscalers-dir 手寫在「額外指令」，載入時會自動搬進本欄。\n'
                        '以專案資料夾為相對路徑基準，執行期間鎖定、重啟生效。').pack(side='left', padx=(4, 0))

        ttk.Label(frame, text='繁中網頁：自動儲存圖片到 image-output、即時耗時、參數說明與排程。下方為本輪 server 日誌。',
                  wraplength=880).pack(anchor='w')
        buttons = ttk.Frame(actionbar)
        buttons.pack(side='left')
        self.start_btn = ttk.Button(buttons, text='▶ 啟動 Image Server', command=self.start)
        self.start_btn.pack(side='left', padx=3)
        self.stop_btn = ttk.Button(buttons, text='■ 停止', command=self.stop, state='disabled')
        self.stop_btn.pack(side='left', padx=3)
        ttk.Button(buttons, text='🌐 開繁中生圖網頁', command=self.open_web).pack(side='left', padx=3)
        ttk.Button(buttons, text='📄 開啟日誌', command=self.open_log).pack(side='left', padx=3)
        self.status = ttk.Label(actionbar, text='未執行')
        self.status.pack(side='left', padx=14)
        logframe = ttk.LabelFrame(pane, text='Image Server 日誌（可拖曳分隔線調整高度）')
        pane.add(logframe, weight=2)
        # --- 分隔線把手：細線外觀，但 ±9px 都抓得到（見上方說明）---
        self._grip = {'active': False, 'y0': 0, 'sp0': 0}

        def _grip_zone(y):
            """回傳 True 表示這個 y 落在可拖曳範圍內。"""
            try:
                return abs(y - pane.sashpos(0)) <= 9
            except tk.TclError:
                return False

        def _grip_press(event):
            if _grip_zone(event.y):
                self._grip.update(active=True, y0=event.y, sp0=pane.sashpos(0))
                return 'break'          # 擋掉原生判定，改由我們自己拖
            return None

        def _grip_drag(event):
            if self._grip['active']:
                pane.sashpos(0, self._grip['sp0'] + (event.y - self._grip['y0']))
                return 'break'
            return None

        def _grip_release(_event):
            if self._grip['active']:
                self._grip['active'] = False
                return 'break'
            return None

        def _grip_cursor(event):
            """游標進入可抓範圍時變成上下箭頭，平常恢復預設。

            要 return 'break'：不然 ttk 原生的 SetCursor 會在我們之後再跑一次，
            把游標蓋回預設（實測 ±7px 會被蓋掉，看起來像抓不到）。
            """
            try:
                pane.configure(cursor='size_ns' if _grip_zone(event.y) else '')
            except tk.TclError:
                pass
            return 'break'

        pane.bind('<Button-1>', _grip_press, add='+')
        pane.bind('<B1-Motion>', _grip_drag, add='+')
        pane.bind('<ButtonRelease-1>', _grip_release, add='+')
        pane.bind('<Motion>', _grip_cursor, add='+')

        self.log = scrolledtext.ScrolledText(logframe, height=5, wrap='word', state='disabled')
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
        self.sync_cache_controls()
        self.sync_vae_tile_controls()
        self.sync_ref_image_hint()
        self.vae_tiling.trace_add('write', lambda *_: self.sync_vae_tile_controls())
        # 切換生圖模型 → 自動套用綁定模板（沒有綁定則套預設模板）。
        self.models[self._TPL_KEY].trace_add('write', lambda *_: self._tpl_on_model_change())
        # 手動微調任何一個模板欄位 → 存回目前選的模板（與 LLM 模式的自動記憶一致）。
        for _var in (self.offload, self.cache_mode, self.cache_threshold, self.cache_warmup,
                     self.vae_tiling, self.vae_relative_tile_size, self.diffusion_fa,
                     self.conditioning_cache_size):
            _var.trace_add('write', lambda *_: self._tpl_on_param_change())
        self._tpl_refresh_ui()
        # 開機：widgets 都就緒、設定也還原完了，才放行 trace 並套用模型的模板。
        self._loading = False
        self._tpl_on_model_change(force=True)
        # 表單控件全部建好後，才把滾輪綁到整棵樹（含稍後建立的子控件）。
        _bind_wheel_tree(frame)
        fit_sash()  # 設定區先量一次（視窗尚未 map 時高度可能是 1，下面會再補一次）
        root.after(80, fit_sash)   # 視窗真的顯示出來、尺寸確定後，把分隔線定位到表單底部
        root.bind('<Map>', lambda _e: root.after(1, fit_sash), add='+')
        self._poll_after = root.after(500, self.poll)

    # ---------- 模板（image-presets.json）：參數組合 + 模型綁定 ----------
    # 模板「不含」埠與模型路徑——那是「選哪個模型」，不是「這模型要怎麼跑」。
    _TPL_KEY = 'diffusion'
    _TPL_NONE = '（不使用模板）'

    def _log(self, text):
        """寫進視窗日誌；模板操作的回饋都走這裡。"""
        try:
            self.log.config(state='normal')
            self.log.insert('end', text)
            self.log.see('end')
            self.log.config(state='disabled')
        except tk.TclError:
            pass

    def _tpl_snapshot(self):
        """目前畫面上的啟動參數 → 一份模板內容。"""
        return {
            'offload': bool(self.offload.get()),
            'cache_mode': self.cache_mode.get(),
            'cache_threshold': self.cache_threshold.get(),
            'cache_warmup': self.cache_warmup.get(),
            'vae_tiling': bool(self.vae_tiling.get()),
            'vae_relative_tile_size': self.vae_relative_tile_size.get(),
            'diffusion_fa': bool(self.diffusion_fa.get()),
            'conditioning_cache_size': self.conditioning_cache_size.get(),
        }

    def _tpl_apply(self, tpl):
        """把模板內容套到 UI；只套模板有的欄位，缺的不動。"""
        if not isinstance(tpl, dict):
            return
        for var, key in ((self.cache_threshold, 'cache_threshold'), (self.cache_warmup, 'cache_warmup'),
                         (self.vae_relative_tile_size, 'vae_relative_tile_size'),
                         (self.conditioning_cache_size, 'conditioning_cache_size')):
            value = tpl.get(key)
            if isinstance(value, str):
                var.set(value)
        if isinstance(tpl.get('cache_mode'), str) and tpl['cache_mode'] in CACHE_MODES:
            self.cache_mode.set(tpl['cache_mode'])
        for var, key in ((self.offload, 'offload'), (self.vae_tiling, 'vae_tiling'),
                         (self.diffusion_fa, 'diffusion_fa')):
            if isinstance(tpl.get(key), bool):
                var.set(tpl[key])
        # Deliberately NO sync_cache_controls()/sync_vae_tile_controls() here: those lock a
        # field once its mode is not selected, and locking is what made an applied template
        # impossible to fine-tune. The trace on cache_mode refreshes the hints on its own.

    def _tpl_current_name(self):
        if not hasattr(self, 'tpl'):
            return ''
        value = self.tpl.get()
        return '' if value in ('', self._TPL_NONE) else value

    def _tpl_default_name(self):
        name = self._presets.get('default', '')
        return name if isinstance(name, str) and name in self._presets['templates'] else ''

    def _tpl_refresh_ui(self):
        if not hasattr(self, 'cb_tpl'):
            return
        self.cb_tpl.config(values=[self._TPL_NONE] + sorted(self._presets['templates']))
        current = self._tpl_current_name()
        self.tpl.set(current if current in self._presets['templates'] else self._TPL_NONE)
        self._tpl_status()

    def _tpl_status(self, note=''):
        if not hasattr(self, 'lbl_tpl'):
            return
        default = self._tpl_default_name()
        base = f'預設模板：{default}（未綁定的模型會自動套用）' if default else '尚未設定預設模板'
        self.lbl_tpl.config(text=base + (f'　｜　{note}' if note else ''))

    def _tpl_on_pick(self, *_args):
        """下拉選模板 → 立即套用。"""
        if self._tpl_busy:
            return
        name = self._tpl_current_name()
        if not name:
            self._log('[模板] 已停用模板（目前參數不變）\n')
            return
        tpl = self._presets['templates'].get(name)
        if not isinstance(tpl, dict):
            return
        self._tpl_busy = True
        try:
            self._tpl_apply(tpl)
        finally:
            self._tpl_busy = False
        self._log(f'[模板] 已套用「{name}」\n')

    def _tpl_store_current(self, name=None, quiet=False):
        """把目前參數存回（指定或目前的）模板；自動存回走 quiet。"""
        name = name or self._tpl_current_name()
        if not name or name not in self._presets['templates']:
            return False
        self._presets['templates'][name] = self._tpl_snapshot()
        ok = save_presets(self._presets)
        if not quiet:
            self._log(f"[模板] {'已存回' if ok else '⚠ 存檔失敗：'}「{name}」\n")
        return ok

    def _tpl_on_param_change(self):
        """某個模板欄位被手動改動 → 標記為待存回。"""
        if self._tpl_busy or self._loading:
            return
        if not self._tpl_current_name():
            return
        self._tpl_dirty = True
        if self._tpl_save_job is not None:
            try:
                self.root.after_cancel(self._tpl_save_job)
            except tk.TclError:
                pass
        try:
            self._tpl_save_job = self.root.after(800, self._tpl_flush_save)
        except tk.TclError:
            self._tpl_save_job = None

    def _tpl_flush_save(self):
        """閒置 800ms 後真的把改動寫回模板檔（合併連續編輯，不每敲一鍵就寫檔）。"""
        self._tpl_save_job = None
        if not self._tpl_dirty or self._tpl_busy or self._loading:
            return
        self._tpl_dirty = False
        name = self._tpl_current_name()
        if not name:
            return
        if self._tpl_store_current(name, quiet=True):
            self._tpl_status(f'已自動存回「{name}」')

    def _tpl_save_as(self):
        from tkinter import simpledialog
        name = simpledialog.askstring('另存模板', '模板名稱', parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()
        if name in self._presets['templates'] and not messagebox.askyesno(
                '覆蓋模板', f'模板「{name}」已存在，要覆蓋嗎？'):
            return
        self._presets['templates'][name] = self._tpl_snapshot()
        if save_presets(self._presets):
            self.tpl.set(name)
            self._tpl_refresh_ui()
            self.tpl.set(name)
            self._log(f'[模板] 已另存模板「{name}」\n')
        else:
            messagebox.showerror('儲存失敗', '無法寫入 image-presets.json')

    def _tpl_rename(self):
        """改名：連綁定一起搬過去，不會留下指向舊名的孤兒綁定。"""
        from tkinter import simpledialog
        old = self._tpl_current_name()
        if not old:
            messagebox.showwarning('沒有模板', '請先在上面的下拉選一個模板。')
            return
        new = simpledialog.askstring('模板改名', '新名稱', initialvalue=old, parent=self.root)
        if not new or not new.strip():
            return
        new = new.strip()
        if new == old:
            return
        if new in self._presets['templates'] and not messagebox.askyesno(
                '名稱已存在', f'已有模板叫「{new}」，要覆蓋它嗎？'):
            return
        self._presets['templates'][new] = self._presets['templates'].pop(old)
        for key, value in list(self._presets['bind'].items()):
            if value == old:
                self._presets['bind'][key] = new
        if self._presets.get('default') == old:
            self._presets['default'] = new
        if save_presets(self._presets):
            self._tpl_refresh_ui()
            self.tpl.set(new)
            self._log(f'[模板] 已改名：{old} → {new}\n')
        else:
            messagebox.showerror('儲存失敗', '無法寫入 image-presets.json')

    def _tpl_delete(self):
        name = self._tpl_current_name()
        if not name:
            return
        if not messagebox.askyesno('刪除模板', f'要刪除模板「{name}」嗎？\n（有綁定它的模型會自動改回不套用）'):
            return
        self._presets['templates'].pop(name, None)
        for key in [k for k, v in self._presets['bind'].items() if v == name]:
            self._presets['bind'].pop(key, None)
        if self._presets.get('default') == name:
            self._presets['default'] = ''
        if save_presets(self._presets):
            self._log(f'[模板] 已刪除模板「{name}」\n')
            self._tpl_refresh_ui()

    def _tpl_set_default(self):
        name = self._tpl_current_name()
        if not name:
            messagebox.showwarning('沒有模板', '請先在上面的下拉選一個模板。')
            return
        self._presets['default'] = name
        if save_presets(self._presets):
            self._log(f'[模板] 已把「{name}」設為預設模板（未綁定的模型會自動套用）\n')
            self._tpl_refresh_ui()

    def _tpl_model_key_of(self, value):
        """綁定用的鍵：正規化後的生圖模型路徑（沒有則用檔名）。"""
        value = str(value or '').strip()
        if not value:
            return ''
        path = pathlib.Path(value)
        if not path.is_absolute():
            path = BASE / path
        return os.path.normcase(str(path))

    def _tpl_bind_model(self):
        """把「目前生圖模型」綁到「目前選的模板」。"""
        key = self._tpl_model_key_of(self.models[self._TPL_KEY].get())
        if not key:
            messagebox.showwarning('沒有生圖模型', '請先在「生圖模型」欄位選一個模型。')
            return
        name = self._tpl_current_name()
        if not name:
            messagebox.showwarning('沒有模板', '請先在上面的下拉選一個模板（或用「另存模板」建立）。')
            return
        self._presets['bind'][key] = name
        if save_presets(self._presets):
            self._log(f'[模板] 已綁定：{pathlib.Path(key).name} → 「{name}」（以後選到它自動套用）\n')
        else:
            messagebox.showerror('儲存失敗', '無法寫入 image-presets.json')

    def _tpl_unbind(self):
        key = self._tpl_model_key_of(self.models[self._TPL_KEY].get())
        if not key:
            return
        hits = [k for k in self._presets['bind'] if k == key or k == os.path.normcase(pathlib.Path(key).name)]
        if not hits:
            messagebox.showinfo('沒有綁定', f'{pathlib.Path(key).name} 目前沒有綁定任何模板。')
            return
        for hit in hits:
            self._presets['bind'].pop(hit, None)
        if save_presets(self._presets):
            self._log(f'[模板] 已解除綁定：{pathlib.Path(key).name}\n')
            self._tpl_model_key = ''
            self._tpl_on_model_change()

    def _tpl_on_model_change(self, force=False):
        """生圖模型變了（或開機 force）→ 套用綁定的模板；沒綁定則套預設模板；都沒有就不動。"""
        if self._tpl_busy or (self._loading and not force):
            return
        key = self._tpl_model_key_of(self.models[self._TPL_KEY].get())
        if not key:
            return
        changed = key != self._tpl_model_key
        self._tpl_model_key = key
        if not changed:
            return
        name = self._presets['bind'].get(key)
        source = '此模型綁定'
        if not (name and name in self._presets['templates']):
            name = self._tpl_default_name()
            source = '預設模板'
        if name and name in self._presets['templates']:
            self._tpl_busy = True
            try:
                self.tpl.set(name)
                self._tpl_apply(self._presets['templates'][name])
            finally:
                self._tpl_busy = False
            self._log(f'[模板] 切換生圖模型 → 自動套用{source}「{name}」\n')
        else:
            self.tpl.set(self._TPL_NONE)
            self._tpl_status()

    def browse_runtime(self):
        path = filedialog.askopenfilename(title='選擇 sd-server.exe', filetypes=[('執行檔', '*.exe')])
        if path:
            self.runtime.set(str(pathlib.Path(path).parent))

    def browse_model(self, key):
        path = filedialog.askopenfilename(title='選擇模型檔案', initialdir=str(BASE / 'IMAGE-MODELS'),
              filetypes=[('模型檔案', '*.gguf *.safetensors *.ckpt *.pt *.pth'), ('所有檔案', '*.*')])
        if path:
            self.models[key].set(path)

    def check_model_choice(self, key):
        """Strip the dropdown's 不適用 marker, then warn if the pick cannot fill the field.

        The marker is display-only.  Selecting an entry used to write the decorated string
        ('...safetensors  ✗ 不適用') into the field, so the stored path pointed at a file that
        does not exist and sd-server failed with 模型格式不符 on a name that looked right.
        The value is kept after stripping (it may be a deliberate experiment) but the user is
        told now, not after sd-server has spent time loading and then died on the metadata.
        """
        value = self.models[key].get().strip()
        if value.endswith(NOT_FOR_FIELD_MARK):
            value = value[:-len(NOT_FOR_FIELD_MARK)].rstrip()
            self.models[key].set(value)
        if not value or model_fits_field(BASE, key, value):
            return
        label = dict((f[0], f[2]) for f in FIELDS).get(key, key)
        messagebox.showwarning(
            '這個檔不適用於這一欄',
            f'「{label}」選到的是不適用的檔案：\n\n{value}\n\n'
            '這個檔看起來是 LoRA／放大器／mmproj，不是該欄要的模型，'
            'sd-server 會以 model metadata validation failed 結束（跑不起來）。\n\n'
            '建議按欄位旁的「清空」把這欄留空，或改選標示沒有「✗ 不適用」的檔案。\n'
            '（若你確定要用它，仍可保留，啟動時由 sd-server 判定。）')

    def refresh(self):
        self.runtime_box['values'] = [name for name, _ in find_image_runtimes(BASE)]
        for key, cb in self.entries.items():
            cb['values'] = model_fieldui_labels(BASE, key, extra=[self.models[key].get()])

    def sync_cache_controls(self):
        """欄位一律保持可編輯——只在對應模式下才真正送出。原本「目前模式用不到就鎖死」
        會讓套用模板後的欄位改不動，故改為：仍可先填好，只用提示標示目前不送／值不合法。"""
        mode = self.cache_mode.get()
        spectrum = mode == 'spectrum'
        block = mode in CACHE_OPTION_MODES
        self.spectrum_w_entry.config(state='normal')
        for widget in (self.cache_threshold_entry, self.cache_warmup_entry):
            widget.config(state='normal')
        inactive = []
        if not spectrum:
            inactive.append('Spectrum W')
        if not block:
            inactive.append('threshold／warmup')
        hint, color = '', '#555'
        if spectrum and not valid_spectrum_w(self.spectrum_w.get()):
            hint, color = 'W 需介於 0.05～1.0', '#c0392b'
        elif block and not valid_cache_threshold(self.cache_threshold.get()):
            hint, color = 'threshold 請留空或填 0～100', '#c0392b'
        elif block and not valid_cache_warmup(self.cache_warmup.get()):
            hint, color = 'warmup 請留空或填 0～50 整數', '#c0392b'
        elif inactive:
            hint = '目前模式不送：' + '、'.join(inactive) + '（仍可先填好）'
        self.cache_hint.config(text=hint, foreground=color)

    def sync_vae_tile_controls(self):
        """分塊大小只在「分塊處理」開啟時有意義；填錯當場標紅。"""
        text = self.vae_relative_tile_size.get().strip()
        if text and not valid_vae_relative_tile_size(text):
            self.vae_tile_hint.config(text='請留空、填 0.01～1.0（如 0.5），或尺寸（如 512x768）',
                                      foreground='#c0392b')
        elif text and not self.vae_tiling.get():
            self.vae_tile_hint.config(text='需搭配「分塊處理」才會送出', foreground='#b9770e')
        else:
            self.vae_tile_hint.config(text='', foreground='#555')

    def sync_ref_image_hint(self):
        """參考圖參數欄下方永遠顯示預設值；格式不對時當場標紅，提醒可按「還原預設」。"""
        text = self.ref_image_args.get().strip()
        if text and not valid_ref_image_args(text):
            self.ref_image_hint.config(
                text=f'格式請為 key=value（逗號分隔），或留空；預設值：{REF_IMAGE_ARGS_DEFAULT}',
                foreground='#c0392b')
        else:
            self.ref_image_hint.config(text=f'預設值：{REF_IMAGE_ARGS_DEFAULT}', foreground='#555')

    def settings(self):
        return {'port': self.port.get(), 'offload': self.offload.get(), 'cache_mode': self.cache_mode.get(),
                'spectrum_w': self.spectrum_w.get(),
                'diffusion_fa': self.diffusion_fa.get(),
                'fa': self.fa.get(),
                'ref_image_args': self.ref_image_args.get(),
                'conditioning_cache_size': self.conditioning_cache_size.get(),
                'extra_args': self.extra_args.get(),
                'max_vram_reserve_gib': self.max_vram_reserve.get(),
                'prefix_cache_disabled': self.prefix_cache_disabled.get(),
                'vae_on_cpu': self.vae_on_cpu.get(),
                'vae_tiling': self.vae_tiling.get(),
                'cache_threshold': self.cache_threshold.get(),
                'cache_warmup': self.cache_warmup.get(),
                'vae_relative_tile_size': self.vae_relative_tile_size.get(),
                'hires_upscalers_dir': self.hires_upscalers_dir.get(),
                'runtime': self.runtime.get(), 'models': {key: var.get() for key, var in self.models.items()}}

    def choose_hires_dir(self):
        """Pick the Hi-res upscaler folder on disk.

        Paths inside the project are stored relative to it, so the settings file keeps working
        when the project is moved; anything else is stored as an absolute path.
        """
        current = self.hires_upscalers_dir.get().strip()
        initial = pathlib.Path(current) if current else BASE
        if not initial.is_absolute():
            initial = BASE / initial
        chosen = filedialog.askdirectory(initialdir=str(initial if initial.is_dir() else BASE),
                                         title='選擇 Hi-res 放大器目錄（放 ESRGAN／Real-ESRGAN 的 .pth 權重）')
        if not chosen:
            return
        path = pathlib.Path(chosen)
        try:
            self.hires_upscalers_dir.set(str(path.relative_to(BASE)))
        except ValueError:
            self.hires_upscalers_dir.set(str(path))

    def reset_log(self):
        """Start a fresh session: new widget content, previous run kept as server.log.prev.

        The runtime truncates server.log on every launch, so the failure evidence of the last
        run (VRAM budget and graph-cut errors) used to be lost for good.
        """
        path = BASE / 'image-output' / 'server.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.stat().st_size:
                shutil.copy2(path, path.with_name('server.log.prev'))
        except OSError:
            pass
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
            cmd = build_cmd_from_settings(BASE, self.port.get(), config)
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
            # A path we dropped must be visible: the page would otherwise silently show only
            # the built-in upscalers and look like the external model "does nothing".
            hires_note = hires_dir_warning(config.get('hires_upscalers_dir'), BASE)
            if hires_note:
                self.logfile.write('[GGUFRun] 警告：' + hires_note + '\n')
            self.logfile.flush()
            self.launch_offset = self.logfile.tell()
            self.refresh_log()
            self.ensure_save_service(port, logpath)
            self.proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=self.logfile,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.start_btn.config(state='disabled')
            self.port_entry.config(state='disabled')
            self.cache_box.config(state='disabled')
            self.fa_box.config(state='disabled')
            self.fa_all_box.config(state='disabled')
            self.ref_image_entry.config(state='disabled')
            self.ref_image_reset.config(state='disabled')
            self.conditioning_box.config(state='disabled')
            self.extra_args_entry.config(state='disabled')
            self.max_vram_box.config(state='disabled')
            self.prefix_cache_box.config(state='disabled')
            self.vae_cpu_box.config(state='disabled')
            self.vae_tiling_box.config(state='disabled')
            self.hires_dir_entry.config(state='disabled')
            self.hires_dir_button.config(state='disabled')
            self.hires_dir_clear.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.status.config(text=f'啟動中 PID {self.proc.pid}；請等待模型載入。')
        except Exception as exc:
            self.record_failure(exc)

    def ensure_save_service(self, port, logpath):
        """Return the save service, reusing the running one.

        The generation page receives this service's port and token in its URL, so recreating it
        on every launch — as this used to — killed an open page: /mode stopped answering and the
        page disabled 「♻ 釋放 LoRA 記憶體」 even while a LoRA was resident.  The service now
        outlives the sd-server process and is only stopped when the window closes.
        """
        if self.save_service is None or self.save_service.httpd is None:
            self.save_service = ImageSaveService(BASE / 'image-output', f'http://127.0.0.1:{port}',
                                                 mode_status=self.mode_status,
                                                 request_mode=self.request_mode, log_path=logpath,
                                                 request_restart=self.request_restart).start()
        else:
            self.save_service.origin = f'http://127.0.0.1:{port}'
            self.save_service.log_path = pathlib.Path(logpath)
        return self.save_service

    def sync_offload_snapshot(self, *_args):
        """Mirror the Tk checkbox for the save service's HTTP thread (never read Tk there)."""
        self.offload_snapshot = bool(self.offload.get())

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
        return {'vision': vision, 'offload': self.launch_settings['offload'] if running
                else self.offload_snapshot,
                'switching': switching, 'ready': ready, 'error': error,
                'lora_resident': running and lora_resident(BASE / 'image-output' / 'server.log',
                                                           since=self.launch_offset)}

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
            self.restart_note = ''
        try:
            self.root.after(0, lambda: self.begin_switch(vision, desired_offload))
        except (RuntimeError, tk.TclError):
            with self.mode_lock:
                self.switching = False
            return False, '控制窗已關閉'
        return True, '切換中；正在安全重啟 Image Server'

    def request_restart(self):
        """Relaunch the Image Server in place so resident LoRA weights are released.

        The generation page asks for this: sd-server keeps a LoRA it applied at runtime in
        the process (weights grow by GBs and nothing but a new process returns them), so
        turning the page's LoRA selector back to «off» alone never frees memory.
        """
        with self.mode_lock:
            if self.switching:
                return False, '伺服器正在切換，請稍候'
            if self.proc is None or self.proc.poll() is not None:
                return False, 'Image Server 未執行'
            self.switching = True
            self.mode_error = ''
            self.restart_note = '重新啟動 Image Server 以釋放常駐 LoRA 記憶體；不關閉存圖服務與佇列。'
            vision = self.vision_loaded
            offload = self.launch_settings['offload']
        try:
            self.root.after(0, lambda: self.begin_switch(vision, offload))
        except (RuntimeError, tk.TclError):
            with self.mode_lock:
                self.switching = False
                self.restart_note = ''
            return False, '控制窗已關閉'
        return True, '正在重新啟動 Image Server 以釋放 LoRA 記憶體'

    def begin_switch(self, vision, offload):
        if not self.switching or self.proc is None:
            return
        self.switch_started = time.monotonic()
        self.status.config(text=self.restart_note or '切換模式：等待舊 Image Server 退出；不關閉存圖服務與佇列。')
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
            cmd = build_cmd_from_settings(BASE, port, config, vision=vision, offload=offload)
            self.logfile.write('\n[GGUFRun] ' + (self.restart_note or
                               ('切換到' + ('Qwen 指令修圖（帶 mmproj）' if vision else '無視覺權重模式'))) +
                               ('；CPU offload 開啟：' if offload else '；自動適配：') + subprocess.list2cmdline(cmd) + '\n')
            self.logfile.flush()
            self.proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=self.logfile,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.launch_offset = self.logfile.tell()
            config['offload'] = offload
            self.offload.set(offload)
            try:
                save_settings(config)
            except OSError as exc:
                self.logfile.write(f'[GGUFRun] 無法保存顯存策略設定（下次啟動需重選）：{exc}\n')
                self.logfile.flush()
            note = self.restart_note
            with self.mode_lock:
                self.vision_loaded = vision
                self.switching = False
                self.mode_error = ''
                self.restart_note = ''
            self.status.config(text='Image Server 重新載入中：常駐 LoRA 記憶體已隨新行程釋放。' if note
                               else ('Image Server 重新載入中；mmproj ' + ('按需載入' if vision else '已卸載')))
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
        # The save service stays up: an open page keeps polling /mode and recovers once a later
        # launch succeeds (only closing the window stops it).
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
        # The save service outlives the sd-server process on purpose: an open generation page
        # holds its port and token, so it must keep answering /mode after a stop/start cycle.
        if self.save_service:
            self.save_service.log_path = BASE / 'image-output' / 'server.log'
        if self.logfile:
            self.logfile.close()
            self.logfile = None
        self.refresh_log()
        self.start_btn.config(state='normal')
        self.port_entry.config(state='normal')
        self.cache_box.config(state='readonly')
        self.fa_box.config(state='normal')
        self.fa_all_box.config(state='normal')
        self.ref_image_entry.config(state='normal')
        self.ref_image_reset.config(state='normal')
        self.conditioning_box.config(state='normal')
        self.extra_args_entry.config(state='normal')
        self.max_vram_box.config(state='normal')
        self.prefix_cache_box.config(state='normal')
        self.vae_cpu_box.config(state='normal')
        self.vae_tiling_box.config(state='normal')
        self.hires_dir_entry.config(state='normal')
        self.hires_dir_button.config(state='normal')
        self.hires_dir_clear.config(state='normal')
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
        # Flush a pending auto-save-back before the app goes away.
        if getattr(self, '_tpl_save_job', None) is not None:
            try:
                self.root.after_cancel(self._tpl_save_job)
            except tk.TclError:
                pass
            self._tpl_save_job = None
        if getattr(self, '_tpl_dirty', False):
            try:
                self._tpl_flush_save()
            except Exception:
                pass
        try:
            save_settings(self.settings())
        except OSError as exc:
            self.record_failure(exc)
        self.stop()
        if self.save_service:
            self.save_service.close()
            self.save_service = None
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    ImageApp(root)
    root.mainloop()
