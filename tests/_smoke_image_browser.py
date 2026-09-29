"""Fake sd-server browser integration; runs locally, never loads model or touches real output."""
import asyncio
import base64
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import urlsplit

import websockets
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from image_save_service import ImageSaveService
HTML = (ROOT / 'assets/image-web.html').read_bytes()

class Fake(BaseHTTPRequestHandler):
    calls = []
    job_size = (512, 1024)
    model_name = 'fake-only'
    upscalers = None   # None → 404：模擬「伺服器還在載入模型，放大器清單尚未就緒」
    def log_message(self, *args):
        pass

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == '/sdcpp/v1/jobs/job_test':
            self.calls.append((route, None))
            image = Image.new('RGB', type(self).job_size, 'orange')
            buf = io.BytesIO(); image.save(buf, format='PNG')
            content = json.dumps({'status':'completed','result':{'images':[{'index':0,'b64_json':base64.b64encode(buf.getvalue()).decode()}]},'error':None}).encode()
        else:
            content = {'/': HTML, '/sdapi/v1/samplers': b'[{"name":"Euler"},{"name":"DDIM"}]',
                       '/sdapi/v1/loras': b'[{"name":"NSFW Qwen Lora","path":"NSFW Qwen Lora.safetensors"},{"name":"styleX","path":"styleX.safetensors"},{"name":"Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128-fused-gguf","path":"Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128-fused-gguf.safetensors"}]',
                       '/sdapi/v1/options': json.dumps({'sd_model_checkpoint': type(self).model_name}).encode(),
                       '/sdapi/v1/upscalers': type(self).upscalers}.get(route)
        if content is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8' if route == '/' else 'application/json')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        if self.path not in ('/sdapi/v1/txt2img', '/sdapi/v1/img2img', '/sdcpp/v1/img_gen'):
            self.send_error(404)
            return
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.calls.append((self.path, request))
        if self.path == '/sdcpp/v1/img_gen':
            if request.get('init_image'):
                # 圖生圖 Hi-res: /sdapi/v1/img2img drops enable_hr in sd-server, so the page sends
                # the native request instead — init_image + first-pass strength + hires block.
                assert request['init_image'].startswith('data:image/png;base64,')
                assert 'mask_image' not in request and 'ref_images' not in request
                hs = request['hires']
                assert hs['enabled'] is True
                assert request['seed'] >= 0 and 'lora' not in request, request
                if request['strength'] == 0:
                    # Pure Hi-res: strength 0 skips stage 1 entirely (server logs
                    # 'target t_enc is 0 steps') and only upscale + low-denoise 2nd pass run.
                    assert hs['scale'] == 1, hs
                    assert (hs['target_width'], hs['target_height']) == (request['width'], request['height']), hs
                else:
                    assert request['strength'] == 0.65, request['strength']
                    assert hs == {'enabled': True, 'upscaler': 'Lanczos', 'scale': 1.5,
                                  'target_width': 384, 'target_height': 384, 'steps': 10,
                                  'denoising_strength': 0.5}, hs
                assert request['sample_params'] == {'sample_steps': 20, 'sample_method': 'euler', 'guidance': {'txt_cfg': 1}}
                type(self).job_size = (hs['target_width'], hs['target_height'])
            else:
                type(self).job_size = (request['width'],request['height'])
                assert request['init_image'] is None
                if request.get('hires'):
                    # native txt2img hires (turbo path): result is the upscaled target size
                    assert request['hires']['enabled'] is True
                    type(self).job_size = (request['hires']['target_width'], request['hires']['target_height'])
                if request.get('lora') or request.get('sample_params',{}).get('custom_sigmas'):
                    # Turbo text2img uses the native branch (it needs custom_sigmas + hires), so it
                    # no longer carries the Qwen-Edit ref_images field.  A fused viggle GGUF carries
                    # the same sigmas but no LoRA (the turbo weights already live in the checkpoint).
                    assert request.get('ref_images', []) == []
                    if any('viggle' in item['path'] for item in request.get('lora') or []):
                        assert request['lora'][-1] == {'path':'Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128-fused-gguf.safetensors','multiplier':1}
                        previous=request['lora'][:-1]
                        assert len(previous)<=1 and all(item['path']=='NSFW Qwen Lora.safetensors' and 0<=item['multiplier']<=2 for item in previous), previous
                    else:
                        assert not any('viggle' in item['path'] for item in request.get('lora') or []), '融合模型不該再掛 Viggle LoRA'
                    params=request['sample_params']
                    assert params['sample_steps'] in (6,8) and params['sample_method']=='euler'
                    assert params['guidance']=={'txt_cfg':1}
                    assert len(params['custom_sigmas'])==params['sample_steps']+1 and params['custom_sigmas'][-1]==0
                    # 6-step turbo and Hi-res may now be combined: both blocks must survive.
                    if request.get('hires'):
                        assert request['hires']['enabled'] is True
                        assert request['hires']['upscaler']=='Lanczos'
                else:
                    assert len(request['ref_images']) in (1, 2)
                    assert request['ref_images'][0].startswith('data:image/png;base64,')
                    assert request['sample_params']=={'sample_steps':20,'sample_method':'euler','guidance':{'txt_cfg':1}}
                    assert request['width']==512 and request['height']==1024 and request['prompt']=='只把裙子改成黑色'
            content=json.dumps({'id':'job_test','status':'queued','poll_url':'/sdcpp/v1/jobs/job_test'}).encode()
            self.send_response(202)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(content)))
            self.end_headers();self.wfile.write(content)
            return
        if self.path.endswith('img2img'):
            assert request['denoising_strength'] == 0.65 and len(request['init_images']) == 1
            assert request['init_images'][0].startswith('data:image/png;base64,')
            assert 'sourceKey' not in request and 'maskKey' not in request and 'offload' not in request and 'mode' not in request
            if 'mask' in request:
                assert request['mask'].startswith('data:image/png;base64,')
                assert request['prompt'] == 'the same image'
        else:
            assert 'init_images' not in request and 'mask' not in request and 'offload' not in request
            for entry in request.get('lora') or []:
                assert entry['path'].endswith('.safetensors') and 0 <= entry['multiplier'] <= 2, entry
            if request.get('enable_hr'):
                assert request['hr_upscaler']=='Lanczos' and request['hr_scale']==1.5
                assert request['hr_steps']==10 and request['denoising_strength']==0.5
                assert (request['hr_resize_x'],request['hr_resize_y'])==(384,384)
        assert request['steps'] == 20 and request['cfg_scale'] == 1
        assert request['sampler_name'].lower() == 'euler' and request['seed'] >= 0
        width,height=(request['hr_resize_x'],request['hr_resize_y']) if request.get('enable_hr') else (request['width'],request['height'])
        image = Image.new('RGB', (width, height), 'orange')
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        content = json.dumps({'images': [base64.b64encode(buf.getvalue()).decode()]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

async def main():
    with tempfile.TemporaryDirectory(dir=ROOT / 'tools', prefix='_browser_smoke_') as tmp:
        directory = pathlib.Path(tmp)
        fake = ThreadingHTTPServer(('127.0.0.1', 0), Fake)
        Thread(target=fake.serve_forever, daemon=True).start()
        origin = f'http://127.0.0.1:{fake.server_port}'
        mode={'vision':False,'offload':True}
        def request_mode(value):
            mode['vision']=value=='qwen_edit'
            return True,'切換完成'
        saver = ImageSaveService(directory / 'saved', origin,
            mode_status=lambda:{'vision':mode['vision'],'offload':mode['offload'],'switching':False,'ready':True,'error':''},
            request_mode=request_mode,
            log_path=directory / 'server.log').start()
        (directory / 'server.log').write_text(
            '[INFO   ] image.cpp:810  - generate_image 512x1024\r\n'
            '[INFO   ] image.cpp:862  - generating image: 1/1 - seed 1\r\n'
            '  |====>         | 4/15 - 6.6s/it\r\n',
            encoding='utf-8')
        chrome = None
        try:
            profile = directory / 'profile'
            url = f'{origin}/?save_port={saver.httpd.server_port}&save_token={saver.token}'
            chrome = subprocess.Popen([
                'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
                '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
                '--remote-allow-origins=*', '--remote-debugging-port=0',
                '--user-data-dir=' + str(profile), url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            debug = profile / 'DevToolsActivePort'
            for _ in range(80):
                if debug.exists():
                    break
                await asyncio.sleep(.15)
            assert debug.exists(), 'Chrome debugger did not start'
            port = int(debug.read_text().splitlines()[0])
            tab = None
            for _ in range(80):
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=2) as response:
                    tabs = json.load(response)
                tab = next((t for t in tabs if t['type'] == 'page' and t['url'].startswith(origin)), None)
                if tab:
                    break
                await asyncio.sleep(.15)
            assert tab, tabs
            async with websockets.connect(tab['webSocketDebuggerUrl'], max_size=10000000) as ws:
                serial = 0
                async def evaluate(code):
                    nonlocal serial
                    serial += 1
                    index = serial
                    await ws.send(json.dumps({'id': index, 'method': 'Runtime.evaluate',
                                              'params': {'expression': code, 'returnByValue': True, 'awaitPromise': True}}))
                    while True:
                        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=12))
                        if message.get('id') == index:
                            result = message.get('result', {})
                            if result.get('exceptionDetails'):
                                raise RuntimeError(result['exceptionDetails'])
                            return result.get('result', {}).get('value')
                for _ in range(80):
                    if await evaluate("document.querySelector('#width')!==null"):
                        break
                    await asyncio.sleep(.15)
                else:
                    raise RuntimeError('Image HTML was not loaded: ' + str(await evaluate("JSON.stringify({href:location.href,html:document.documentElement.outerHTML.slice(0,350),ready:document.readyState})")))
                print('defaults', await evaluate("JSON.stringify([document.querySelector('#width').value,document.querySelector('#height').value,document.querySelector('#steps').value,document.querySelector('#cfg').value,document.querySelector('#sampler').value])"), flush=True)
                assert await evaluate("document.querySelector('#size-preset').value") == '512x1024'
                progress = json.loads(await evaluate("(async()=>{const r=await fetch(saveBase+'/progress?steps=15&token='+encodeURIComponent(saveToken));return JSON.stringify(await r.json());})()"))
                assert progress['phase'] == 'sampling' and progress['step'] == 4 and progress['label'] == '取樣 4/15 步', progress
                print('progress endpoint', progress['label'], flush=True)
                # 迴歸：伺服器比網頁晚就緒（按下「開網頁」時 sd-server 還在載入模型）時，
                # 舊版會靜默地只剩內建方法、且永不重抓，使用者因此看不到外部 ESRGAN 模型。
                note = ''
                for _ in range(40):
                    note = await evaluate("document.querySelector('#upscaler-note').textContent")
                    if '尚未取得' in note:
                        break
                    await asyncio.sleep(.1)
                else:
                    raise RuntimeError('放大器清單未就緒時未顯示提示訊息：' + repr(note))
                assert await evaluate("document.querySelector('#hr-upscaler').options.length") == 2
                Fake.upscalers = json.dumps([
                    {'model_name': None, 'model_path': None, 'model_url': None, 'name': 'None', 'scale': 4},
                    {'model_name': None, 'model_path': None, 'model_url': None, 'name': 'Lanczos', 'scale': 4},
                    {'model_name': None, 'model_path': None, 'model_url': None, 'name': 'Nearest', 'scale': 4},
                    {'model_name': 'ESRGAN_4x', 'model_url': None, 'scale': 4,
                     'model_path': 'D:/ExampleProject/IMAGE-MODELS/upscalers/RealESRGAN_x4plus_anime_6B.pth',
                     'name': 'RealESRGAN_x4plus_anime_6B'}]).encode()
                for _ in range(60):
                    if await evaluate("document.querySelector('#hr-upscaler').options.length") == 3:
                        break
                    await asyncio.sleep(.25)
                else:
                    raise RuntimeError('伺服器就緒後沒有自動補上放大器清單：'
                                       + str(await evaluate("document.querySelector('#upscaler-note').textContent")))
                assert await evaluate("document.querySelector('#upscaler-note').textContent") == ''
                labels = json.loads(await evaluate(
                    "JSON.stringify([...document.querySelector('#hr-upscaler').options].map(o=>o.textContent))"))
                assert labels == ['Lanczos（內建）', 'Nearest（內建）',
                                  'RealESRGAN_x4plus_anime_6B（外部模型 4×）'], labels
                print('upscalers: late-ready server list picked up automatically, note cleared', flush=True)
                await evaluate("queue.tasks=[{id:99,body:{prompt:'x'},status:'running',startedAt:Date.now()}];progressLabel='取樣 4/15 步';updateElapsed();void 0")
                elapsed = await evaluate("document.querySelector('#elapsed').textContent")
                assert '取樣 4/15 步' in elapsed and '已耗時' in elapsed, elapsed
                print('elapsed text', elapsed, flush=True)
                await evaluate("queue.tasks=[];queue.notify();stopProgress();void 0")
                assert await evaluate("document.querySelector('#elapsed').textContent") == '尚未開始生圖'
                await evaluate("document.querySelector('#size-preset').value='1024x512';document.querySelector('#size-preset').dispatchEvent(new Event('change'));void 0")
                assert await evaluate("document.querySelector('#width').value+'x'+document.querySelector('#height').value") == '1024x512'
                await evaluate("document.querySelector('#size-preset').value='2560x1440';document.querySelector('#size-preset').dispatchEvent(new Event('change'));void 0")
                assert '顯存' in await evaluate("document.querySelector('#size-warning').textContent")
                # 每個常用規格都有直式與橫式：直式是同一組數字的轉置，選了要真的填入寬高並進 draft。
                await evaluate("document.querySelector('#size-preset').value='1088x1920';document.querySelector('#size-preset').dispatchEvent(new Event('change'));void 0")
                assert await evaluate("document.querySelector('#width').value+'x'+document.querySelector('#height').value") == '1088x1920'
                assert '顯存' in await evaluate("document.querySelector('#size-warning').textContent")
                await evaluate("document.querySelector('#prompt').value='portrait preset';document.querySelector('#size-preset').value='736x1280';document.querySelector('#size-preset').dispatchEvent(new Event('change'));void 0")
                portrait = json.loads(await evaluate("JSON.stringify(draftBody())"))
                assert (portrait['width'], portrait['height']) == (736, 1280), portrait
                await evaluate("document.querySelector('#size-preset').value='512x1024';document.querySelector('#size-preset').dispatchEvent(new Event('change'));void 0")
                print('presets: portrait presets fill width/height and reach the payload', flush=True)
                await evaluate("document.querySelector('#size-preset').value='640x384';document.querySelector('#size-preset').dispatchEvent(new Event('change'));document.querySelector('#prompt').value='a bright orange';document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("document.querySelector('#tasklist').children.length===1"):
                        break
                    await asyncio.sleep(.1)
                else:
                    raise RuntimeError('Txt task not queued: ' + str(await evaluate("JSON.stringify({status:document.querySelector('#status').textContent,valid:document.querySelector('#form').checkValidity(),w:document.querySelector('#width').value,h:document.querySelector('#height').value,step:document.querySelector('#steps').value,dbg:typeof queue,source:typeof selectedSourceKey,error:document.querySelector('#queue-count').textContent})")))
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(80):
                    snap = json.loads(await evaluate("JSON.stringify({status:document.querySelector('#status').textContent,count:document.querySelector('#queue-count').textContent,preview:document.querySelector('#preview img')?.naturalWidth||0,file:document.querySelector('#result').textContent,rows:document.querySelector('#tasklist').children.length})"))
                    if snap['file'] and snap['preview'] and snap['rows'] == 0:
                        break
                    if '失敗 1' in snap['count']:
                        raise RuntimeError('Browser task failed: ' + str(snap))
                    await asyncio.sleep(.15)
                else:
                    raise RuntimeError('Browser timed out: ' + str(snap))
                files = list((directory / 'saved').glob('*.png'))
                assert len(files) == 1 and '640x384-20steps-cfg1-seed' in files[0].name and '-euler-' in files[0].name, files
                print('saved', files[0].name, 'bytes', files[0].stat().st_size, 'preview', snap['preview'], 'rows', snap['rows'], flush=True)
                await evaluate("document.querySelector('#mode').value='img2img';document.querySelector('#mode').dispatchEvent(new Event('change'));document.querySelector('#size-preset').value='512x1024';document.querySelector('#size-preset').dispatchEvent(new Event('change'));document.querySelector('#prompt').value='restyle portrait';void 0")
                assert await evaluate("document.querySelector('#source-fields').hidden") is False
                await evaluate("(async()=>{const c=document.createElement('canvas');c.width=32;c.height=64;c.getContext('2d').fillRect(0,0,32,64);const blob=await new Promise(resolve=>c.toBlob(resolve,'image/png'));const file=new File([blob],'input.png',{type:'image/png'});const dt=new DataTransfer();dt.items.add(file);document.querySelector('#source-file').files=dt.files;document.querySelector('#source-file').dispatchEvent(new Event('change'));})()")
                for _ in range(80):
                    if await evaluate("document.querySelector('#source-info').textContent.startsWith('已選')"):
                        break
                    await asyncio.sleep(.1)
                else:
                    raise RuntimeError('Browser did not store original image: '+str(await evaluate("document.querySelector('#status').textContent")))
                await evaluate("document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("document.querySelector('#tasklist').children.length===1"):
                        break
                    await asyncio.sleep(.1)
                else:
                    raise RuntimeError('Img task not queued: '+str(await evaluate("document.querySelector('#status').textContent")))
                stored = await evaluate("localStorage.getItem('ggufrun.image.queue.v1')")
                assert 'sourceKey' in stored and 'data:image/' not in stored
                await evaluate('location.reload();void 0')
                for _ in range(80):
                    if await evaluate("document.querySelector('#tasklist li')?.textContent.includes('Img2Img')"):
                        break
                    await asyncio.sleep(.15)
                else:
                    raise RuntimeError('Image-to-image task did not survive refresh')
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(80):
                    snap = json.loads(await evaluate("JSON.stringify({count:document.querySelector('#queue-count').textContent,rows:document.querySelector('#tasklist').children.length,file:document.querySelector('#result').textContent})"))
                    if len(list((directory/'saved').glob('*.png'))) == 2 and snap['rows'] == 0:
                        break
                    if '失敗 1' in snap['count']:
                        raise RuntimeError('Img task failed: '+str(snap))
                    await asyncio.sleep(.15)
                else:
                    raise RuntimeError('Img task timed out: '+str(snap))
                img_calls=[v for path,v in Fake.calls if path=='/sdapi/v1/img2img']
                assert len(img_calls)==1 and img_calls[0]['width']==512 and img_calls[0]['height']==1024
                assert any('512x1024-20steps-cfg1-' in p.name and '-img2img-dn0.65-' in p.name
                           for p in (directory/'saved').glob('*.png'))
                print('img2img: upload persisted through refresh, sent source and strength, saved 512x1024',flush=True)
                await evaluate("document.querySelector('#mode').value='qwen_edit';document.querySelector('#mode').dispatchEvent(new Event('change'));document.querySelector('#prompt').value='只把裙子改成黑色';void 0")
                assert await evaluate("document.querySelector('#denoise-fields').hidden") is True
                for _ in range(80):
                    if mode['vision']:
                        break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Qwen Edit did not request vision mode')
                await evaluate("(async()=>{const c=document.createElement('canvas');c.width=32;c.height=64;c.getContext('2d').fillRect(0,0,32,64);const blob=await new Promise(resolve=>c.toBlob(resolve,'image/png'));const dt=new DataTransfer();dt.items.add(new File([blob],'edit.png',{type:'image/png'}));document.querySelector('#edit-files').files=dt.files;document.querySelector('#edit-files').dispatchEvent(new Event('change'));})()")
                for _ in range(80):
                    if await evaluate("document.querySelector('#edit-list').textContent.includes('edit.png')"):
                        break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Edit source was not stored')
                await evaluate("document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("document.querySelector('#tasklist').children.length===1"):
                        break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Edit not queued: '+str(await evaluate("document.querySelector('#status').textContent")))
                # Existing single-reference queue entries use sourceKey rather than sourceKeys.
                await evaluate("queue.tasks[0].body.sourceKey=queue.tasks[0].body.sourceKeys[0];delete queue.tasks[0].body.sourceKeys;queue.notify();void 0")
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(80):
                    snap = json.loads(await evaluate("JSON.stringify({count:document.querySelector('#queue-count').textContent,rows:document.querySelector('#tasklist').children.length})"))
                    if len(list((directory/'saved').glob('*.png')))==3 and snap['rows']==0:break
                    if '失敗 1' in snap['count']:raise RuntimeError('Edit failed: '+str(snap))
                    await asyncio.sleep(.15)
                else:raise RuntimeError('Edit timed out: '+str(snap))
                assert len([c for c in Fake.calls if c[0]=='/sdcpp/v1/img_gen'])==1
                assert len([c for c in Fake.calls if c[0]=='/sdcpp/v1/jobs/job_test'])>=1
                assert any('-qwenedit-' in p.name for p in (directory/'saved').glob('*.png'))
                for _ in range(80):
                    if await evaluate('queue.running') is False:break
                    await asyncio.sleep(.1)
                # An over-limit batch must not add a partial selection.
                await evaluate("(async()=>{const c=document.createElement('canvas');c.width=32;c.height=64;const blob=await new Promise(r=>c.toBlob(r,'image/png'));const dt=new DataTransfer();for(let i=0;i<11;i++)dt.items.add(new File([blob],i+'.png',{type:'image/png'}));document.querySelector('#edit-files').files=dt.files;document.querySelector('#edit-files').dispatchEvent(new Event('change'));await editReady;})()")
                assert await evaluate("document.querySelectorAll('#edit-list .edit-item').length") == 1
                assert '最多 10 張' in await evaluate("document.querySelector('#status').textContent")
                # Add three distinct references, remove the mistaken middle image,
                # then clear the current selection AFTER queuing; pending references must survive refresh.
                await evaluate("(async()=>{for(const [name,color] of [['first.png','red'],['mistake.png','green'],['last.png','blue']]){const c=document.createElement('canvas');c.width=32;c.height=64;c.getContext('2d').fillStyle=color;c.getContext('2d').fillRect(0,0,32,64);const blob=await new Promise(r=>c.toBlob(r,'image/png'));const dt=new DataTransfer();dt.items.add(new File([blob],name,{type:'image/png'}));document.querySelector('#edit-files').files=dt.files;document.querySelector('#edit-files').dispatchEvent(new Event('change'));}await editReady;})()")
                assert await evaluate("document.querySelectorAll('#edit-list .edit-item').length") == 4
                await evaluate("document.querySelectorAll('#edit-list .edit-item button[aria-label^=移除]')[2].click();void 0")
                assert await evaluate("[...document.querySelectorAll('#edit-list .edit-item span')].map(x=>x.textContent).join('|')") == '1. edit.png|2. first.png|3. last.png'
                await evaluate("document.querySelectorAll('#edit-list .edit-item button[aria-label^=下移]')[1].click();void 0")
                assert await evaluate("[...document.querySelectorAll('#edit-list .edit-item span')].map(x=>x.textContent).join('|')") == '1. edit.png|2. last.png|3. first.png'
                await evaluate("document.querySelectorAll('#edit-list .edit-item button[aria-label^=移除]')[0].click();document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===1"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Multiple references not queued: '+str(await evaluate("document.querySelector('#status').textContent")))
                assert await evaluate("queue.tasks[0].body.sourceKeys.length") == 2
                for _ in range(80):
                    if await evaluate("document.querySelectorAll('#tasklist .task-refs img').length") == 2:break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Pending references have no preview')
                assert await evaluate("[...document.querySelectorAll('#tasklist .task-refs img')].map(x=>x.alt).join('|')") == '參考圖 1|參考圖 2'
                assert 'data:image/' not in await evaluate("localStorage.getItem('ggufrun.image.queue.v1')")
                await evaluate("document.querySelector('#clear-edit').click();void 0")
                assert await evaluate("document.querySelectorAll('#edit-list .edit-item').length") == 0
                await evaluate("document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if '至少一張' in await evaluate("document.querySelector('#status').textContent"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Empty reference selection was accepted')
                assert await evaluate("queue.tasks.length") == 1
                await evaluate('location.reload();void 0')
                for _ in range(80):
                    if await evaluate("document.querySelector('#tasklist')?.children.length===1"):break
                    await asyncio.sleep(.15)
                else:raise RuntimeError('Pending multiple references lost on reload')
                for _ in range(80):
                    if await evaluate("[...document.querySelectorAll('#tasklist .task-refs img')].every(x=>x.naturalWidth===32) && document.querySelectorAll('#tasklist .task-refs img').length===2"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Pending reference previews did not load after refresh')
                preview_colors=json.loads(await evaluate("JSON.stringify([...document.querySelectorAll('#tasklist .task-refs img')].map(img=>{const c=document.createElement('canvas');c.width=1;c.height=1;c.getContext('2d').drawImage(img,0,0);return [...c.getContext('2d').getImageData(0,0,1,1).data].slice(0,3)}))"))
                assert preview_colors==[[0,0,255],[255,0,0]], preview_colors
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(80):
                    if len(list((directory/'saved').glob('*.png')))==4 and await evaluate('queue.running') is False:break
                    if '失敗 1' in await evaluate("document.querySelector('#queue-count').textContent"):
                        raise RuntimeError('Multiple reference task failed: '+str(await evaluate("document.querySelector('#status').textContent")))
                    await asyncio.sleep(.15)
                else:raise RuntimeError('Multiple reference task timed out')
                refs=[v for path,v in Fake.calls if path=='/sdcpp/v1/img_gen'][-1]['ref_images']
                assert len(refs)==2
                colors=[]
                for ref in refs:
                    image=Image.open(io.BytesIO(base64.b64decode(ref.split(',',1)[-1])))
                    colors.append(image.getpixel((1,1))[:3])
                assert colors==[(0,0,255),(255,0,0)], colors
                print('qwen multi: remove single/clear, queue survives reload, ordered red/blue only', flush=True)
                assert await evaluate("document.querySelector('#unload-vision')===null")
                await evaluate("document.querySelector('#mode').value='img2img';document.querySelector('#mode').dispatchEvent(new Event('change'));void 0")
                for _ in range(80):
                    if not mode['vision']:break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Selecting Img2Img did not unload vision')
                await evaluate("document.querySelector('#mode').value='txt2img';document.querySelector('#mode').dispatchEvent(new Event('change'));void 0")
                print('qwen edit: reference upload and save; selecting Img2Img unloaded mock vision',flush=True)
                await evaluate('location.reload();void 0')
                for _ in range(80):
                    if await evaluate("document.querySelector('#tasklist')?.children.length===0 && (document.querySelector('#preview img')?.naturalWidth||0)===0 && document.querySelector('#elapsed').textContent==='尚未開始生圖'"):
                        break
                    await asyncio.sleep(.15)
                else:
                    raise RuntimeError('Refresh kept completed rows or restored a stale preview/elapsed: '+str(await evaluate("JSON.stringify({rows:document.querySelector('#tasklist').children.length,preview:document.querySelector('#preview img')?.naturalWidth||0,elapsed:document.querySelector('#elapsed').textContent})")))
                print('refresh: clean task list, blank preview and no stale elapsed', flush=True)
                await evaluate("localStorage.setItem('ggufrun.image.queue.v1',JSON.stringify([{id:11,body:{prompt:'old finished'},status:'done'},{id:12,body:{prompt:'waiting'},status:'pending'}]));location.reload();void 0")
                for _ in range(80):
                    if await evaluate("document.querySelector('#tasklist li strong')?.textContent.includes('waiting')"):
                        break
                    await asyncio.sleep(.15)
                else:
                    raise RuntimeError('Pending task did not survive refresh')
                row = await evaluate("document.querySelector('#tasklist li strong').textContent")
                assert row.startswith('目前 #1'), row
                print('migration: pending item shown as', row, flush=True)
                snap = json.loads(await evaluate("JSON.stringify({collapsed:document.querySelector('#hires-fields').tagName==='DETAILS'&&!document.querySelector('#hires-fields').open,disabledOff:['hr-upscaler','hr-scale','hr-steps','hr-denoise'].every(id=>document.getElementById(id).disabled)})"))
                assert snap['collapsed'] and snap['disabledOff'], 'Hi-res 進階設定未摺疊或子欄位未停用：'+str(snap)
                await evaluate("document.querySelector('#enable-hr').checked=true;document.querySelector('#enable-hr').dispatchEvent(new Event('change'));void 0")
                enabled = await evaluate("['hr-upscaler','hr-scale','hr-steps','hr-denoise'].every(id=>!document.getElementById(id).disabled)")
                assert enabled, '勾選「啟用 Hi-res」後子欄位仍為停用'
                print('hires: collapsed advanced section, opt-in enables its four options', flush=True)
                await evaluate("queue.remove(queue.tasks[0].id);document.querySelector('#width').value='256';document.querySelector('#height').value='256';document.querySelector('#prompt').value='hires test';document.querySelector('#hr-scale').value='1.5';document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("document.querySelector('#tasklist').children.length===1"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Hi-res task not queued: '+str(await evaluate("document.querySelector('#status').textContent")))
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(80):
                    if any('384x384-' in p.name for p in (directory/'saved').glob('*.png')):break
                    await asyncio.sleep(.15)
                else:raise RuntimeError('Hi-res did not save enlarged dimensions: '+str(await evaluate("document.querySelector('#status').textContent")))
                assert any(path=='/sdapi/v1/txt2img' and data.get('enable_hr') for path,data in Fake.calls if isinstance(data,dict))
                print('hires: native txt2img fields and output filename 384x384 verified',flush=True)
                # 1.5x of a 32-aligned 288px side is 432px: reject non-aligned Hi-res output.
                assert await evaluate("(()=>{document.querySelector('#width').value='288';document.querySelector('#height').value='256';document.querySelector('#hr-scale').value='1.5';try{draftBody();return false}catch(e){return e.message.includes('32 的倍數')}})()")
                await evaluate("document.querySelector('#width').value='256';void 0")
                # 圖生圖 Hi-res：sdapi 的 img2img 在後端忽略 enable_hr，所以頁面改走原生 img_gen
                # （init_image + strength + hires）；同一組 Hi-res 欄位在兩種模式都要能用。
                await evaluate("document.querySelector('#mode').value='img2img';document.querySelector('#mode').dispatchEvent(new Event('change'));void 0")
                for _ in range(80):
                    if await evaluate("document.querySelector('#hires-fields').hidden===false && document.querySelector('#hr-help').textContent.includes('/sdcpp/v1/img_gen')"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Img2Img 未顯示 Hi-res 區塊或說明未切換：'+str(await evaluate("JSON.stringify({hidden:document.querySelector('#hires-fields').hidden,help:document.querySelector('#hr-help').textContent.slice(0,40)})")))
                assert await evaluate("document.querySelector('#hr-label').textContent.includes('第一段')"), 'Hi-res 標題未依模式切換'
                await evaluate("document.querySelector('#hr-scale').value='1.5';document.querySelector('#hr-upscaler').value='Lanczos';void 0")
                await evaluate("(async()=>{const c=document.createElement('canvas');c.width=32;c.height=64;const x=c.getContext('2d');x.fillStyle='white';x.fillRect(0,0,32,64);const blob=await new Promise(resolve=>c.toBlob(resolve,'image/png'));const dt=new DataTransfer();dt.items.add(new File([blob],'hires-source.png',{type:'image/png'}));document.querySelector('#source-file').files=dt.files;document.querySelector('#source-file').dispatchEvent(new Event('change'));})()")
                for _ in range(80):
                    if await evaluate("document.querySelector('#source-info').textContent.startsWith('已選')"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Hi-res 圖生圖原圖未存入瀏覽器')
                await evaluate("document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("document.querySelector('#tasklist').children.length===1"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Img2Img Hi-res 任務未加入佇列：'+str(await evaluate("document.querySelector('#status').textContent")))
                assert '384×384' in await evaluate("document.querySelector('#tasklist .task-summary').textContent"), '佇列摘要未顯示 Hi-res 目標尺寸'
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(160):
                    if await evaluate('queue.running') is False and any('-img2img-dn0.65-' in p.name for p in (directory/'saved').glob('*.png')):break
                    if '失敗 1' in await evaluate("document.querySelector('#queue-count').textContent"):
                        raise RuntimeError('Img2Img Hi-res 失敗：'+str(await evaluate("document.querySelector('#tasklist .task-error')?.textContent")))
                    await asyncio.sleep(.15)
                else:raise RuntimeError('Img2Img Hi-res 未完成：'+str(await evaluate("document.querySelector('#status').textContent")))
                assert any('384x384-' in p.name and '-img2img-dn0.65-' in p.name for p in (directory/'saved').glob('*.png')), \
                    [p.name for p in (directory/'saved').glob('*.png')]
                assert not any(path=='/sdapi/v1/img2img' and data.get('enable_hr') for path,data in Fake.calls if isinstance(data,dict))
                print('img2img hires: native init_image+strength+hires sent, saved 384x384',flush=True)
                # 遮罩 Inpaint 與 Hi-res 不能同時使用（原生 hires 的第二輪不會重套遮罩）。
                await evaluate("(async()=>{const c=document.createElement('canvas');c.width=32;c.height=64;c.getContext('2d').fillRect(0,0,32,64);const blob=await new Promise(resolve=>c.toBlob(resolve,'image/png'));const dt=new DataTransfer();dt.items.add(new File([blob],'hires-mask.png',{type:'image/png'}));document.querySelector('#mask-file').files=dt.files;document.querySelector('#mask-file').dispatchEvent(new Event('change'));})()")
                await evaluate("document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===1"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('遮罩 + Hi-res 任務未加入佇列')
                native_before=len([1 for p,_ in Fake.calls if p=='/sdcpp/v1/img_gen'])
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(120):
                    error=await evaluate("document.querySelector('#tasklist .task-error')?.textContent||''")
                    if '遮罩' in error:break
                    await asyncio.sleep(.15)
                else:raise RuntimeError('遮罩 + Hi-res 未被擋下：'+str(error))
                assert len([1 for p,_ in Fake.calls if p=='/sdcpp/v1/img_gen'])==native_before, '遮罩 + Hi-res 不該送出請求'
                await evaluate("queue.remove(queue.tasks[0].id);document.querySelector('#mask-file').value='';document.querySelector('#enable-hr').checked=false;document.querySelector('#enable-hr').dispatchEvent(new Event('change'));document.querySelector('#mode').value='txt2img';document.querySelector('#mode').dispatchEvent(new Event('change'));void 0")
                print('img2img hires: mask+Hi-res refused locally with a clear reason',flush=True)
                assert await evaluate("document.querySelector('#offload')===null")
                assert mode['offload'] is True
                await evaluate("document.querySelector('#enable-hr').checked=false;document.querySelector('#enable-hr').dispatchEvent(new Event('change'));document.querySelector('#viggle').value='6step';document.querySelector('#viggle').dispatchEvent(new Event('change'));document.querySelector('#width').value='256';document.querySelector('#height').value='256';document.querySelector('#prompt').value='turbo apple';document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.some(t=>t.body.viggleTurbo)"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Viggle task not queued: '+str(await evaluate("document.querySelector('#status').textContent")))
                assert json.loads(await evaluate("JSON.stringify(queue.tasks.find(t=>t.body.viggleTurbo).body)"))['steps']==6
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===0") and any(v.get('lora') for p,v in Fake.calls if p=='/sdcpp/v1/img_gen'):break
                    await asyncio.sleep(.15)
                else:raise RuntimeError('Viggle task not saved: '+str(await evaluate("document.querySelector('#status').textContent")))
                assert mode['offload'] is True
                await evaluate("document.querySelector('#viggle').value='off';document.querySelector('#viggle').dispatchEvent(new Event('change'));void 0")
                print('viggle: opt-in sends fused LoRA, 7 sigmas, 6 Euler steps and saves image',flush=True)
                # Browser -> queue snapshot -> fake native endpoint -> saved PNG: 8-step dense text.
                await evaluate("document.querySelector('#viggle').value='6step';document.querySelector('#viggle').dispatchEvent(new Event('change'));document.querySelector('#steps').value='8';document.querySelector('#prompt').value='dense text';document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===1"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Viggle 8-step task not queued: '+str(await evaluate("document.querySelector('#status').textContent")))
                assert json.loads(await evaluate("JSON.stringify(queue.tasks[0].body)"))['steps']==8
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===0 && !queue.running"):break
                    await asyncio.sleep(.15)
                else:raise RuntimeError('Viggle 8-step task not saved: '+str(await evaluate("document.querySelector('#status').textContent")))
                eight=[v for p,v in Fake.calls if p=='/sdcpp/v1/img_gen' and v.get('prompt')=='dense text'][-1]
                assert eight['sample_params']['sample_steps']==8 and len(eight['sample_params']['custom_sigmas'])==9
                await evaluate("document.querySelector('#viggle').value='off';document.querySelector('#viggle').dispatchEvent(new Event('change'));void 0")
                print('viggle: 8 Euler steps, 9 sigmas and saved image',flush=True)

                # 6-step turbo + Hi-res：以前前端硬擋，後端其實可以。兩個區塊都要留著送出。
                await evaluate("document.querySelector('#viggle').value='6step';document.querySelector('#viggle').dispatchEvent(new Event('change'));document.querySelector('#enable-hr').checked=true;document.querySelector('#enable-hr').dispatchEvent(new Event('change'));document.querySelector('#width').value='256';document.querySelector('#height').value='256';document.querySelector('#hr-scale').value='2';document.querySelector('#hr-upscaler').value='Lanczos';document.querySelector('#prompt').value='turbo hires apple';document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===1"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('turbo+Hi-res 任務未被接受：'+str(await evaluate("document.querySelector('#status').textContent")))
                turbo_body=json.loads(await evaluate("JSON.stringify(queue.tasks[0].body)"))
                assert turbo_body.get('viggleTurbo') is True and turbo_body.get('enable_hr') is True, turbo_body
                assert (turbo_body.get('hr_resize_x'),turbo_body.get('hr_resize_y'))==(512,512), turbo_body
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(160):
                    if await evaluate('queue.running') is False and any('512x512-' in p.name for p in (directory/'saved').glob('*.png')):break
                    if '失敗 1' in await evaluate("document.querySelector('#queue-count').textContent"):
                        raise RuntimeError('turbo+Hi-res 失敗：'+str(await evaluate("document.querySelector('#tasklist .task-error')?.textContent")))
                    await asyncio.sleep(.15)
                else:raise RuntimeError('turbo+Hi-res 未完成：'+str(await evaluate("document.querySelector('#status').textContent")))
                turbo_hires=[v for p,v in Fake.calls if p=='/sdcpp/v1/img_gen' and v.get('lora') and v.get('hires')][-1]
                assert len(turbo_hires['sample_params']['custom_sigmas'])==7
                assert turbo_hires['hires']['scale']==2 and turbo_hires['hires']['target_width']==512
                print('viggle+hires: 6-step sigmas and Hi-res block sent together, saved 512x512',flush=True)
                await evaluate("document.querySelector('#viggle').value='off';document.querySelector('#viggle').dispatchEvent(new Event('change'));document.querySelector('#enable-hr').checked=false;document.querySelector('#enable-hr').dispatchEvent(new Event('change'));void 0")
                # 純 Hi-res：改動強度 0 → 第一段不重畫（後端 t_enc 0 步），只放大＋低去噪。
                await evaluate("document.querySelector('#mode').value='img2img';document.querySelector('#mode').dispatchEvent(new Event('change'));document.querySelector('#enable-hr').checked=true;document.querySelector('#enable-hr').dispatchEvent(new Event('change'));document.querySelector('#width').value='256';document.querySelector('#height').value='256';document.querySelector('#hr-scale').value='1';document.querySelector('#hr-upscaler').value='Lanczos';document.querySelector('#denoising-strength').value='0';document.querySelector('#prompt').value='';void 0")
                await evaluate("(async()=>{const c=document.createElement('canvas');c.width=64;c.height=64;c.getContext('2d').fillRect(0,0,64,64);const blob=await new Promise(r=>c.toBlob(r,'image/png'));const dt=new DataTransfer();dt.items.add(new File([blob],'pure-hires.png',{type:'image/png'}));document.querySelector('#source-file').files=dt.files;document.querySelector('#source-file').dispatchEvent(new Event('change'));})()")
                for _ in range(80):
                    if await evaluate("document.querySelector('#source-info').textContent.startsWith('已選')"):break
                    await asyncio.sleep(.1)
                await evaluate("document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===1"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('純 Hi-res 任務未加入：'+str(await evaluate("document.querySelector('#status').textContent")))
                pure=json.loads(await evaluate("JSON.stringify(queue.tasks[0].body)"))
                assert pure.get('denoising_strength')==0 and pure.get('hr_scale')==1 and pure.get('hrNative') is True, pure
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(160):
                    if await evaluate('queue.running') is False and len([1 for p,v in Fake.calls if p=='/sdcpp/v1/img_gen' and v.get('strength')==0])==1:break
                    if '失敗 1' in await evaluate("document.querySelector('#queue-count').textContent"):
                        raise RuntimeError('純 Hi-res 失敗：'+str(await evaluate("document.querySelector('#tasklist .task-error')?.textContent")))
                    await asyncio.sleep(.15)
                else:raise RuntimeError('純 Hi-res 未完成：'+str(await evaluate("document.querySelector('#status').textContent")))
                assert any(path=='/sdapi/v1/img2img' and data.get('enable_hr') for path,data in Fake.calls if isinstance(data,dict)) is False
                print('pure hires: strength 0 keeps stage 1 out and only runs upscale + 2nd pass',flush=True)
                await evaluate("document.querySelector('#enable-hr').checked=false;document.querySelector('#enable-hr').dispatchEvent(new Event('change'));document.querySelector('#viggle').value='off';document.querySelector('#viggle').dispatchEvent(new Event('change'));document.querySelector('#denoising-strength').value='0.65';document.querySelector('#hr-scale').value='1.5';document.querySelector('#mode').value='txt2img';document.querySelector('#mode').dispatchEvent(new Event('change'));void 0")
                # Reproduce a cached image returned 1920x1088 for a 1920x1080 request.
                calls_before=len(Fake.calls)
                await evaluate("(async()=>{const c=document.createElement('canvas');c.width=1920;c.height=1088;const img=c.toDataURL('image/png').split(',')[1];const body={prompt:'rounded height',width:1920,height:1080,steps:6,cfg_scale:1,seed:3236843848,sampler_name:'Euler',viggleTurbo:true};queue.tasks=[{id:987654,body,status:'failed',error:'所填尺寸與 PNG 實際尺寸不符',startedAt:Date.now()}];unsavedImages.set(987654,img);queue.notify();queue.retry(987654);document.querySelector('#run-queue').click();})()")
                for _ in range(80):
                    if any('1920x1088-' in p.name for p in (directory/'saved').glob('*.png')) and await evaluate('queue.running') is False:break
                    await asyncio.sleep(.15)
                else:raise RuntimeError('Cached rounded PNG did not save: '+str(await evaluate("JSON.stringify({status:document.querySelector('#status').textContent,rows:queue.tasks.map(t=>({status:t.status,error:t.error})),file:document.querySelector('#result').textContent})")))
                assert len(Fake.calls)==calls_before, 'Retry generated a duplicate image'
                print('cached PNG retry: 1920x1080 requested, 1920x1088 saved without regeneration',flush=True)
                # Multi-LoRA: standing NSFW selector, collapsible list, sdapi field and 6-step Viggle combination.
                lora_ui = json.loads(await evaluate("JSON.stringify({nsfw:[...document.querySelector('#nsfw-lora').options].map(o=>o.value),disabled:document.querySelector('#nsfw-lora').disabled,rows:[...document.querySelectorAll('#lora-list .lora-item')].map(r=>r.dataset.path),collapsed:document.querySelector('#lora-fields').tagName==='DETAILS'&&!document.querySelector('#lora-fields').open,warning:document.querySelector('#lora-warning').textContent})"))
                assert lora_ui['nsfw'] == ['off','NSFW Qwen Lora.safetensors'], lora_ui
                assert lora_ui['disabled'] is False and lora_ui['rows'] == ['NSFW Qwen Lora.safetensors','styleX.safetensors'], lora_ui
                assert lora_ui['collapsed'] and lora_ui['warning'] == '', lora_ui
                await evaluate("document.querySelector('#width').value='256';document.querySelector('#height').value='256';document.querySelector('#prompt').value='standing nsfw';document.querySelector('#nsfw-lora').value='NSFW Qwen Lora.safetensors';document.querySelector('#nsfw-strength').value='1.3';document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===1"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('NSFW LoRA task not queued')
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===0 && !queue.running"):break
                    await asyncio.sleep(.15)
                else:raise RuntimeError('NSFW LoRA task not finished: '+str(await evaluate("document.querySelector('#status').textContent")))
                nsfw_calls=[v for p,v in Fake.calls if p=='/sdapi/v1/txt2img' and v.get('lora')]
                assert nsfw_calls and nsfw_calls[-1]['lora'] == [{'path':'NSFW Qwen Lora.safetensors','multiplier':1.3}], nsfw_calls[-1] if nsfw_calls else None
                print('multi-lora: standing NSFW selector sent through sdapi lora array',flush=True)
                await evaluate("document.querySelector('#viggle').value='6step';document.querySelector('#viggle').dispatchEvent(new Event('change'));document.querySelector('#prompt').value='viggle plus nsfw';document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===1"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Viggle+NSFW task not queued: '+str(await evaluate("document.querySelector('#status').textContent")))
                assert json.loads(await evaluate("JSON.stringify(queue.tasks[0].body.lora)")) == [{'path':'NSFW Qwen Lora.safetensors','multiplier':1.3},{'path':'Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128-fused-gguf.safetensors','multiplier':1}]
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===0 && !queue.running"):break
                    await asyncio.sleep(.15)
                else:raise RuntimeError('Viggle+NSFW task not finished: '+str(await evaluate("document.querySelector('#status').textContent")))
                combined=[v for p,v in Fake.calls if p=='/sdcpp/v1/img_gen' and v.get('lora') and len(v['lora'])==2][-1]
                assert combined['lora'][0] == {'path':'NSFW Qwen Lora.safetensors','multiplier':1.3} and combined['sample_params']['sample_steps']==6, combined['lora']
                await evaluate("document.querySelector('#viggle').value='off';document.querySelector('#viggle').dispatchEvent(new Event('change'));document.querySelector('#nsfw-lora').value='off';document.querySelectorAll('#lora-list .lora-item input[type=checkbox]')[1].click();void 0")
                await evaluate("document.querySelector('#prompt').value='list lora only';document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===1"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Checklist LoRA task not queued: '+str(await evaluate("document.querySelector('#status').textContent")))
                assert json.loads(await evaluate("JSON.stringify(queue.tasks[0].body.lora)")) == [{'path':'styleX.safetensors','multiplier':1}]
                assert 'LoRA' in await evaluate("document.querySelector('#tasklist .task-summary').textContent")
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===0 && !queue.running"):break
                    await asyncio.sleep(.15)
                else:raise RuntimeError('Checklist LoRA task not finished')
                listed=[v for p,v in Fake.calls if p=='/sdapi/v1/txt2img' and v.get('lora')][-1]
                assert listed['lora'] == [{'path':'styleX.safetensors','multiplier':1}] and listed['steps']==20, listed
                assert await evaluate("document.querySelector('#nsfw-lora').value") == 'off'
                print('multi-lora: checklist sends one LoRA, queue summary shows it, base 20 steps kept',flush=True)
                await evaluate("document.querySelector('#viggle').value='off';document.querySelector('#viggle').dispatchEvent(new Event('change'));document.querySelector('#nsfw-strength').value='1';document.querySelectorAll('#lora-list .lora-item input[type=checkbox]')[1].click();void 0")
                await evaluate("document.querySelector('#mode').value='img2img';document.querySelector('#mode').dispatchEvent(new Event('change'));document.querySelector('#enable-hr').checked=false;document.querySelector('#prompt').value='';void 0")
                await evaluate("(async()=>{const c=document.createElement('canvas');c.width=32;c.height=64;const x=c.getContext('2d');x.fillStyle='white';x.fillRect(0,0,32,64);const blob=await new Promise(resolve=>c.toBlob(resolve,'image/png'));for(const id of ['source-file','mask-file']){const dt=new DataTransfer();dt.items.add(new File([blob],id+'.png',{type:'image/png'}));document.querySelector('#'+id).files=dt.files;document.querySelector('#'+id).dispatchEvent(new Event('change'));}})()")
                for _ in range(80):
                    if await evaluate("document.querySelector('#source-info').textContent.startsWith('已選')"):break
                    await asyncio.sleep(.1)
                await evaluate("document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("document.querySelector('#tasklist').children.length===1"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('Mask task not queued: '+str(await evaluate("document.querySelector('#status').textContent")))
                assert await evaluate("queue.tasks[0].body.maskKey?.length>0")
                await evaluate("queue.tasks[0].body.offload=false;void 0")  # Simulate a task saved by the older web UI.
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(80):
                    if len([v for p,v in Fake.calls if p=='/sdapi/v1/img2img' and 'mask' in v])==1 and await evaluate('queue.running') is False:break
                    await asyncio.sleep(.15)
                else:raise RuntimeError('Mask task not sent: '+str(await evaluate("document.querySelector('#status').textContent")))
                assert await evaluate("document.querySelector('#tasklist').children.length") == 0
                assert mode['offload'] is True
                print('img2img mask: native mask and blank prompt; legacy queue offload ignored',flush=True)
                # 載入作者轉好的 Viggle 融合 GGUF：頁面要自動開 Viggle、自動 6 步，且不可再疊 LoRA（會套兩次）。
                await evaluate("document.querySelector('#mode').value='txt2img';document.querySelector('#mode').dispatchEvent(new Event('change'));document.querySelector('#enable-hr').checked=false;document.querySelector('#enable-hr').dispatchEvent(new Event('change'));document.querySelector('#viggle').value='off';document.querySelector('#viggle').dispatchEvent(new Event('change'));document.querySelector('#steps').value='20';document.querySelectorAll('#lora-list .lora-item input[type=checkbox]').forEach(c=>{if(c.checked)c.click();});void 0")
                Fake.model_name='qwen-image-2.1-viggle-turbo-fused-Q4_K_M'
                await evaluate("document.querySelector('#prompt').value='fused gguf apple';void 0")
                await evaluate("discover();void 0")
                for _ in range(40):
                    if await evaluate("document.querySelector('#viggle').value==='6step'"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('融合 GGUF 未自動切到 Viggle：'+str(await evaluate("document.querySelector('#model').textContent")))
                assert await evaluate("document.querySelector('#steps').value")=='6', await evaluate("document.querySelector('#steps').value")
                assert 'Viggle 融合模型' in await evaluate("document.querySelector('#model').textContent")
                await evaluate("document.querySelector('#form').requestSubmit();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===1"):break
                    await asyncio.sleep(.1)
                else:raise RuntimeError('融合 GGUF 任務未入列：'+str(await evaluate("document.querySelector('#status').textContent")))
                assert await evaluate("queue.tasks[0].body.viggleTurbo===true")
                assert await evaluate("!queue.tasks[0].body.lora||!queue.tasks[0].body.lora.some(l=>/viggle/i.test(l.path))"), '融合 GGUF 不該再加掛 Viggle LoRA'
                calls_before=len(Fake.calls)
                await evaluate("document.querySelector('#run-queue').click();void 0")
                for _ in range(80):
                    if await evaluate("queue.tasks.length===0 && !queue.running"):break
                    await asyncio.sleep(.15)
                else:raise RuntimeError('融合 GGUF 未完成：'+str(await evaluate("document.querySelector('#status').textContent")))
                fused=[v for p,v in Fake.calls[calls_before:] if p=='/sdcpp/v1/img_gen'][-1]
                assert fused['sample_params']['sample_steps']==6 and fused['sample_params']['sample_method']=='euler', fused['sample_params']
                assert not fused.get('lora'), fused.get('lora')
                assert await evaluate("document.querySelector('#tasklist').children.length")==0
                print('fused viggle gguf: auto 6 steps, no duplicate LoRA, saved image',flush=True)
                Fake.model_name='fake-only'
        finally:
            if chrome and chrome.poll() is None:
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(chrome.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
            saver.close()
            fake.shutdown()
            fake.server_close()

if __name__ == '__main__':
    asyncio.run(main())
