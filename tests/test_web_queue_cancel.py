"""Browser test: an 「中斷待確認」 row must offer an obvious cancel, and cancelling must stick.

Uses the fake sd-server + real headless Chrome (never generates, never touches the GPU or the
real image-output), so it is skipped when Chrome or websockets is unavailable.
"""
import asyncio
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from threading import Thread

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

try:
    import websockets
except ImportError:  # pragma: no cover - environment without the browser harness deps
    websockets = None

CHROME = pathlib.Path('C:/Program Files (x86)/Google/Chrome/Application/chrome.exe')

if websockets is not None:
    _spec = importlib.util.spec_from_file_location('_smoke_cancel', ROOT / 'tests/_smoke_image_browser.py')
    smoke = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(smoke)
    from image_save_service import ImageSaveService


@unittest.skipUnless(CHROME.exists() and websockets is not None,
                     'needs Chrome and websockets (browser harness)')
class QueueCancelBrowserTests(unittest.TestCase):
    """The page is the only place the queue lives, so the cancel button is verified in a browser."""

    def test_interrupted_row_offers_cancel_and_cancelling_clears_it(self):
        result = asyncio.run(self._probe())
        self.assertEqual(result['status_text'], '中斷待確認')
        self.assertTrue(any('取消' in label and '不重送' in label for label in result['buttons']),
                        f'a cancel button is missing: {result["buttons"]}')
        self.assertTrue(any('重試' in label for label in result['buttons']), result['buttons'])
        self.assertEqual(result['rows_after_cancel'], 0)
        self.assertIn('中斷待確認 0', result['count_after_cancel'])
        self.assertEqual(result['stored_after_cancel'], '[]')
        self.assertTrue(any('取消' in label for label in result['failed_buttons']),
                        f'a failed row must be cancellable too: {result["failed_buttons"]}')
        self.assertEqual(result['retried'], 'pending')  # 重試 still requeues

    async def _probe(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tools', prefix='_cancel_test_') as tmp:
            directory = pathlib.Path(tmp)
            fake = ThreadingHTTPServer(('127.0.0.1', 0), smoke.Fake)
            Thread(target=fake.serve_forever, daemon=True).start()
            origin = f'http://127.0.0.1:{fake.server_port}'
            saver = ImageSaveService(directory / 'saved', origin,
                                     mode_status=lambda: {'vision': False, 'offload': True, 'switching': False,
                                                          'ready': True, 'error': '', 'lora_resident': True},
                                     request_mode=lambda value: (True, '切換完成'),
                                     log_path=directory / 'server.log').start()
            (directory / 'server.log').write_text('[INFO   ] image.cpp:810  - generate_image 512x512\r\n',
                                                  encoding='utf-8')
            chrome = None
            try:
                profile = directory / 'profile'
                url = f'{origin}/?save_port={saver.httpd.server_port}&save_token={saver.token}'
                chrome = subprocess.Popen([
                    str(CHROME), '--headless=new', '--disable-gpu', '--no-first-run',
                    '--no-default-browser-check', '--remote-allow-origins=*',
                    '--remote-debugging-port=0', '--user-data-dir=' + str(profile), url],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                debug = profile / 'DevToolsActivePort'
                for _ in range(80):
                    if debug.exists():
                        break
                    await asyncio.sleep(.15)
                self.assertTrue(debug.exists(), 'Chrome debugger did not start')
                port = int(debug.read_text().splitlines()[0])
                tab = None
                for _ in range(80):
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=2) as response:
                        tabs = json.load(response)
                    tab = next((t for t in tabs if t['type'] == 'page' and t['url'].startswith(origin)), None)
                    if tab:
                        break
                    await asyncio.sleep(.15)
                self.assertIsNotNone(tab, tabs)
                async with websockets.connect(tab['webSocketDebuggerUrl'], max_size=10000000) as websocket:
                    serial = 0

                    async def evaluate(code):
                        nonlocal serial
                        serial += 1
                        index = serial
                        await websocket.send(json.dumps({
                            'id': index, 'method': 'Runtime.evaluate',
                            'params': {'expression': code, 'returnByValue': True, 'awaitPromise': True}}))
                        while True:
                            message = json.loads(await asyncio.wait_for(websocket.recv(), timeout=15))
                            if message.get('id') == index:
                                answer = message.get('result', {})
                                if answer.get('exceptionDetails'):
                                    raise RuntimeError(answer['exceptionDetails'])
                                return answer.get('result', {}).get('value')

                    for _ in range(80):
                        if await evaluate("document.querySelector('#width')!==null"):
                            break
                        await asyncio.sleep(.15)
                    else:
                        raise RuntimeError('Image HTML was not loaded')

                    await evaluate("queue.tasks=[{id:7,body:{prompt:'中斷的任務',mode:'txt2img',width:512,"
                                   "height:512,steps:6,cfg_scale:1,seed:1},status:'interrupted'}];queue.notify();void 0")
                    labels = json.loads(await evaluate(
                        "JSON.stringify([...document.querySelectorAll('#tasklist button')].map(b=>b.textContent))"))
                    status_text = await evaluate("document.querySelector('#tasklist .task-head span').textContent")
                    await evaluate("(async()=>{const b=[...document.querySelectorAll('#tasklist button')]"
                                   ".find(x=>/取消/.test(x.textContent));if(b)b.click();return !!b;})()")
                    rows_after = await evaluate("document.querySelector('#tasklist').children.length")
                    count_after = await evaluate("document.querySelector('#queue-count').textContent")
                    stored_after = await evaluate("localStorage.getItem('ggufrun.image.queue.v1')")
                    # A failed row keeps its own wording, and the retry path stays available.
                    await evaluate("queue.tasks=[{id:8,body:{prompt:'失敗的任務',mode:'txt2img',width:512,"
                                   "height:512,steps:6,cfg_scale:1,seed:2},status:'failed',error:'boom'}];"
                                   "queue.notify();void 0")
                    failed_labels = json.loads(await evaluate(
                        "JSON.stringify([...document.querySelectorAll('#tasklist button')].map(b=>b.textContent))"))
                    await evaluate("queue.retry(8);queue.notify();void 0")
                    retried = await evaluate("queue.tasks[0].status")
                    return {'buttons': labels, 'status_text': status_text, 'rows_after_cancel': rows_after,
                            'count_after_cancel': count_after, 'stored_after_cancel': stored_after,
                            'failed_buttons': failed_labels, 'retried': retried}
            finally:
                if chrome:
                    chrome.terminate()
                    try:
                        chrome.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        chrome.kill()
                saver.close()
                fake.shutdown()
                fake.server_close()


if __name__ == '__main__':
    unittest.main()
