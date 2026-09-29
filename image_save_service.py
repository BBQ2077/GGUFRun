"""Loopback-only PNG saver and progress reader for the local sd-server HTML page."""
import base64
import binascii
import json
import math
import os
import pathlib
import re
import secrets
import struct
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
MAX_REQUEST = 40 * 1024 * 1024
MAX_IMAGE = 30 * 1024 * 1024
LOG_TAIL = 65536
BAR = re.compile(r'\|\s*(\d+)\s*/\s*(\d+)\b')
REQUEST_START = re.compile(r'generate_image \d+x\d+')
# Ordered decisive markers: sd-server interleaves tensor-loading lines during sampling
# (CPU offload reloads tensors between steps), so the phase must follow the generation
# sequence rather than whichever marker happens to be printed last.
PHASES = (('generate_image completed', 'done'), ('decode_first_stage completed', 'done'),
          ('latent 1 decoded', 'decode'), ('decoding 1 latents', 'decode'),
          ('sampling completed', 'decode'), ('generating image: ', 'sampling'),
          ('get_learned_condition completed', 'conditioning'),
          ('encode_first_stage completed', 'encode'),
          ('preprocess init[', 'preprocess'), ('preprocess ref[', 'preprocess'))


def parse_progress(text, steps=0):
    """Describe the live phase of the newest generation in a server.log tail.

    Only the newest `generate_image WxH` segment is inspected, so a finished or older
    task in the same tail cannot be reported as the current phase, and tensor-loading
    lines inside that segment cannot be mistaken for the sampling phase. A step count
    is only read from a bar whose total equals the running task's step count: the log
    reuses `n/n` bars for tensor loading and VAE decoding.
    """
    try:
        steps = max(0, int(steps))
    except (TypeError, ValueError):
        steps = 0
    if not text:
        return {'phase': 'unknown', 'step': 0, 'total': steps, 'label': '等待伺服器日誌'}
    starts = [match.start() for match in REQUEST_START.finditer(text)]
    preprocess = max((text.rfind(marker) for marker in ('preprocess ref[', 'preprocess init[')), default=-1)
    if starts:
        previous = starts[-2] if len(starts) > 1 else -1
        # `preprocess` is printed just before `generate_image WxH` of the same request,
        # so start the segment there when it is newer than the previous request.
        begin = preprocess if preprocess > previous else starts[-1]
    else:
        begin = preprocess if preprocess >= 0 else 0
    segment = text[begin:]
    latest = None
    for marker, name in PHASES:
        index = segment.rfind(marker)
        if index >= 0 and (latest is None or index > latest[0]):
            latest = (index, name)
    phase = latest[1] if latest else ('ready' if 'listening on' in text else 'preparing')
    step = 0
    if phase == 'sampling' and steps:
        start = segment.rfind('generating image')
        bars = [match for match in BAR.finditer(segment, start if start >= 0 else 0)]
        current = next((match for match in reversed(bars) if int(match.group(2)) == steps), None)
        step = int(current.group(1)) if current else 0
    labels = {'preparing': '準備中（載入權重）', 'loading': '載入運算圖',
              'preprocess': '預處理原圖', 'encode': '編碼原圖',
              'conditioning': '解讀提示與參考圖', 'decode': '解碼影像', 'done': '完成，保存中'}
    if phase == 'sampling':
        label = f'取樣 {step}/{steps} 步' if steps else '取樣中'
    else:
        label = labels.get(phase, '伺服器已就緒')
    return {'phase': phase, 'step': step, 'total': steps, 'label': label}


class ImageSaveService:
    def __init__(self, output_dir, origin, mode_status=None, request_mode=None, log_path=None,
                 request_restart=None):
        self.output_dir = pathlib.Path(output_dir)
        self.mode_status = mode_status
        self.request_mode = request_mode
        self.request_restart = request_restart
        self.log_path = pathlib.Path(log_path) if log_path else self.output_dir / 'server.log'

        self.origin = origin
        self.token = secrets.token_urlsafe(32)
        self.httpd = None
        self.thread = None
        self.url = None

    def start(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def allowed(self):
                return self.headers.get('Origin') == service.origin

            def cors(self):
                self.send_header('Access-Control-Allow-Origin', service.origin)
                self.send_header('Vary', 'Origin')

            def reply(self, code, data):
                body = json.dumps(data, ensure_ascii=False).encode('utf-8')
                self.send_response(code)
                if self.allowed():
                    self.cors()
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self):
                if not self.allowed() or self.path not in ('/save', '/mode', '/restart') or self.headers.get('Access-Control-Request-Method') != 'POST':
                    self.reply(403, {'error': '來源不符'})
                    return
                self.send_response(204)
                self.cors()
                self.send_header('Access-Control-Allow-Methods', 'POST')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-GGUFRun-Token')
                self.send_header('Content-Length', '0')
                self.end_headers()

            def do_POST(self):
                if (self.path not in ('/save', '/mode', '/restart') or not self.allowed() or
                        self.headers.get('X-GGUFRun-Token') != service.token or
                        self.headers.get('Content-Type', '').split(';')[0].lower() != 'application/json'):
                    self.reply(403, {'error': '拒絕未授權儲存'})
                    return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if self.path == '/restart':
                        # Resident LoRA weights (applied at runtime) can only be returned by a
                        # new sd-server process, so the page asks the control window to relaunch.
                        if not service.request_restart:
                            self.reply(409, {'error': '此控制窗不支援重新啟動'})
                            return
                        if not 0 < size <= 128:
                            raise ValueError('重啟要求格式錯誤')
                        request = json.loads(self.rfile.read(size))
                        if (not isinstance(request, dict) or
                                not isinstance(request.get('reason', ''), str) or
                                len(request.get('reason', '')) > 32):
                            raise ValueError('重啟要求格式錯誤')
                        accepted, message = service.request_restart()
                        self.reply(202 if accepted else 409, {'status': message})
                        return
                    if self.path == '/mode':
                        if not service.request_mode:
                            self.reply(409, {'error': '此控制窗不支援模式切換'})
                            return
                        if not 0 < size <= 128:
                            raise ValueError('模式資料格式錯誤')
                        request = json.loads(self.rfile.read(size))
                        if not isinstance(request, dict) or request.get('mode') not in ('txt2img', 'img2img', 'qwen_edit'):
                            raise ValueError('生成模式不正確')
                        if 'offload' in request:
                            raise ValueError('顯存策略請於 Image 控制窗設定，網頁僅能切換生成模式')
                        accepted, message = service.request_mode(request['mode'])
                        self.reply(202 if accepted else 409, {'status': message})
                        return
                    if not 0 < size <= MAX_REQUEST:
                        raise ValueError('圖片資料太大或缺少長度')
                    data = json.loads(self.rfile.read(size))
                    encoded = data['image']
                    if not isinstance(encoded, str):
                        raise ValueError('圖片資料格式錯誤')
                    raw = base64.b64decode(encoded, validate=True)
                    if len(raw) > MAX_IMAGE or len(raw) < 45 or not raw.startswith(PNG_MAGIC + b'\x00\x00\x00\rIHDR') or not raw.endswith(b'IEND\xaeB`\x82'):
                        raise ValueError('不是有效 PNG 或檔案過大')
                    width, height = struct.unpack('>II', raw[16:24])
                    if not (0 < width <= 16384 and 0 < height <= 16384):
                        raise ValueError('PNG 尺寸不合理')
                    if data.get('width') != width or data.get('height') != height:
                        raise ValueError('所填尺寸與 PNG 實際尺寸不符，請刷新頁面後重試')
                    steps = data.get('steps')
                    seed = data.get('seed')
                    cfg = data.get('cfg_scale')
                    elapsed = data.get('duration_ms')
                    sampler = data.get('sampler', 'default')
                    if type(steps) is not int or not 1 <= steps <= 150:
                        raise ValueError('步數不正確')
                    if type(seed) is not int or not -1 <= seed <= 2**32 - 1:
                        raise ValueError('種子不正確')
                    if type(cfg) not in (int, float) or not math.isfinite(cfg) or not 0 <= cfg <= 30:
                        raise ValueError('CFG 不正確')
                    if type(elapsed) is not int or not 0 <= elapsed <= 43200000:
                        raise ValueError('耗時不正確')
                    if not isinstance(sampler, str) or not re.fullmatch(r'[A-Za-z0-9_+. -]{1,64}', sampler):
                        raise ValueError('取樣器名稱不正確')
                    sampler = re.sub(r'[^a-z0-9]+', '', sampler.lower()) or 'default'
                    mode = data.get('mode', 'txt2img')
                    if mode not in ('txt2img', 'img2img', 'qwen_edit'):
                        raise ValueError('生成模式不正確')
                    strength = data.get('denoising_strength')
                    if mode == 'img2img' and (type(strength) not in (int, float) or
                                              not math.isfinite(strength) or not 0 <= strength <= 1):
                        raise ValueError('圖生圖改動強度不正確')
                    mode_label = f'-img2img-dn{strength:g}' if mode == 'img2img' else ('-qwenedit' if mode == 'qwen_edit' else '')
                    # Viggle turbo runs get a mode tag so a fused-LoRA result is not mistaken for a
                    # plain Euler render: '-6step' for the author's 6-step schedule, '-turbo' otherwise.
                    if data.get('viggle') is True:
                        mode_label += '-6step' if steps == 6 else '-turbo'
                    cfg_text = f'{cfg:g}'
                    seed_text = str(seed) if seed >= 0 else 'random-unknown'
                    seconds = round(elapsed / 1000)
                    service.output_dir.mkdir(parents=True, exist_ok=True)
                    name = (f'image-{datetime.now():%Y%m%d-%H%M%S}-{width}x{height}'
                            f'-{steps}steps-cfg{cfg_text}-seed{seed_text}-{sampler}{mode_label}'
                            f'-{seconds}s-{secrets.token_hex(6)}.png')
                    path = service.output_dir / name
                    try:
                        with path.open('xb') as output:
                            output.write(raw)
                    except Exception:
                        path.unlink(missing_ok=True)
                        raise
                    self.reply(200, {'filename': name})
                except (ValueError, KeyError, TypeError, binascii.Error) as exc:
                    self.reply(400, {'error': str(exc)})
                except (OSError, json.JSONDecodeError) as exc:
                    self.reply(500, {'error': '寫入圖片失敗：' + str(exc)})

            def log_tail(self):
                """Read the last chunk of the server log without blocking on a busy writer."""
                try:
                    with open(service.log_path, 'rb') as stream:
                        stream.seek(0, os.SEEK_END)
                        size = stream.tell()
                        stream.seek(max(0, size - LOG_TAIL))
                        return stream.read().decode('utf-8', 'replace')
                except OSError:
                    return ''

            def do_GET(self):
                parsed = urlsplit(self.path)
                if parsed.path == '/mode':
                    if not service.mode_status or not self.allowed() or parse_qs(parsed.query).get('token') != [service.token]:
                        self.reply(403, {'error': '拒絕未授權讀取'})
                    else:
                        self.reply(200, service.mode_status())
                    return
                if parsed.path == '/progress':
                    if not self.allowed() or parse_qs(parsed.query).get('token') != [service.token]:
                        self.reply(403, {'error': '拒絕未授權讀取'})
                        return
                    try:
                        steps = int(parse_qs(parsed.query).get('steps', ['0'])[0])
                    except ValueError:
                        steps = 0
                    progress = parse_progress(self.log_tail(), steps)
                    self.reply(200, progress)
                    return
                name = parsed.path.removeprefix('/image/')
                valid_old = re.fullmatch(r'image-[0-9]{8}-[0-9]{6}-(?:[0-9]+|x)-[0-9a-f]{12}\.png', name)
                valid_new = re.fullmatch(r'image-[0-9]{8}-[0-9]{6}-[0-9]+x[0-9]+-[0-9]+steps-cfg[0-9.]+-seed(?:[0-9]+|random-unknown)-[a-z0-9]+(?:-img2img-dn(?:[01](?:\.[0-9]+)?)|-qwenedit)?(?:-(?:6step|turbo))?-[0-9]+s-[0-9a-f]{12}\.png', name)
                if (not parsed.path.startswith('/image/') or not (valid_old or valid_new)
                    or parse_qs(parsed.query).get('token') != [service.token]
                    or not self.allowed()):
                    self.reply(403, {'error': '拒絕未授權讀取'})
                    return
                try:
                    data = (service.output_dir / name).read_bytes()
                except OSError:
                    self.reply(404, {'error': '圖片不存在'})
                    return
                self.send_response(200)
                self.cors()
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.httpd.server_port}'
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def close(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.thread.join(timeout=3)
            self.httpd = None
            self.thread = None
