import base64
import json
import pathlib
import tempfile
import unittest
import urllib.error
import urllib.request

from image_save_service import ImageSaveService, parse_progress

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Yf+mwAAAABJRU5ErkJggg==')


class SaveServiceTests(unittest.TestCase):
    def test_saves_distinct_png_files_and_serves_preview(self):
        with tempfile.TemporaryDirectory() as td:
            service = ImageSaveService(pathlib.Path(td), 'http://127.0.0.1:18436')
            service.start()
            try:
                for _ in range(2):
                    req = urllib.request.Request(service.url + '/save',
                        data=json.dumps({'image': base64.b64encode(PNG).decode(), 'task_id': 7,
                            'width': 1, 'height': 1, 'steps': 15, 'cfg_scale': 1,
                            'seed': 42, 'sampler': 'Euler', 'duration_ms': 1450}).encode(),
                        headers={'Origin': service.origin, 'X-GGUFRun-Token': service.token,
                                 'Content-Type': 'application/json'}, method='POST')
                    with urllib.request.urlopen(req, timeout=4) as response:
                        self.assertEqual(response.status, 200)
                        filename = json.load(response)['filename']
                    self.assertRegex(filename, r'^image-\d{8}-\d{6}-1x1-15steps-cfg1-seed42-euler-1s-[0-9a-f]{12}\.png$')
                    self.assertEqual((pathlib.Path(td) / filename).read_bytes(), PNG)
                self.assertEqual(len(list(pathlib.Path(td).glob('*.png'))), 2)
                get = urllib.request.Request(service.url + '/image/' + filename + '?token=' + service.token,
                                              headers={'Origin': service.origin})
                with urllib.request.urlopen(get, timeout=4) as response:
                    self.assertEqual(response.read(), PNG)
            finally:
                service.close()

    def test_img2img_saves_strength_in_filename_and_serves_preview(self):
        with tempfile.TemporaryDirectory() as td:
            service = ImageSaveService(pathlib.Path(td), 'http://127.0.0.1:18436').start()
            try:
                payload = dict(image=base64.b64encode(PNG).decode(), width=1, height=1,
                               steps=15, cfg_scale=1, seed=42, sampler='Euler', duration_ms=1450,
                               mode='img2img', denoising_strength=.65)
                req = urllib.request.Request(service.url+'/save', data=json.dumps(payload).encode(),
                    headers={'Origin': service.origin, 'X-GGUFRun-Token':service.token,
                             'Content-Type':'application/json'},method='POST')
                with urllib.request.urlopen(req,timeout=4) as response:
                    filename=json.load(response)['filename']
                self.assertRegex(filename,r'-euler-img2img-dn0\.65-1s-[0-9a-f]{12}\.png$')
                get=urllib.request.Request(service.url+'/image/'+filename+'?token='+service.token,
                                           headers={'Origin':service.origin})
                with urllib.request.urlopen(get,timeout=4) as response:
                    self.assertEqual(response.read(),PNG)
                payload['denoising_strength']=2
                req=urllib.request.Request(service.url+'/save',data=json.dumps(payload).encode(),
                    headers={'Origin':service.origin,'X-GGUFRun-Token':service.token,'Content-Type':'application/json'},method='POST')
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(req,timeout=4)
                caught.exception.close()
            finally:
                service.close()

    def test_mode_endpoint_requires_origin_and_token(self):
        calls = []
        with tempfile.TemporaryDirectory() as td:
            origin = 'http://127.0.0.1:18436'
            service = ImageSaveService(pathlib.Path(td), origin,
                mode_status=lambda: {'vision': False, 'ready': True, 'switching': False, 'error': ''},
                request_mode=lambda mode: (calls.append(mode) or True, '切換中')).start()
            try:
                url = service.url + '/mode'
                good = {'Origin': origin, 'X-GGUFRun-Token': service.token, 'Content-Type': 'application/json'}
                with urllib.request.urlopen(urllib.request.Request(url, data=b'{"mode":"qwen_edit"}', headers=good)) as response:
                    self.assertEqual(response.status, 202)
                self.assertEqual(calls, ['qwen_edit'])
                with urllib.request.urlopen(urllib.request.Request(url + '?token=' + service.token, headers={'Origin': origin})) as response:
                    self.assertFalse(json.load(response)['vision'])
                for headers in ({'Origin': origin, 'Content-Type': 'application/json'},
                                {'Origin': 'http://evil', 'X-GGUFRun-Token': service.token, 'Content-Type': 'application/json'}):
                    with self.assertRaises(urllib.error.HTTPError) as cm:
                        urllib.request.urlopen(urllib.request.Request(url, data=b'{"mode":"qwen_edit"}', headers=headers))
                    cm.exception.close()
                    self.assertEqual(cm.exception.code, 403)
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(urllib.request.Request(url, data=b'{"mode":"wrong"}', headers=good))
                cm.exception.close()
                self.assertEqual(cm.exception.code, 400)
                self.assertEqual(calls, ['qwen_edit'])
            finally:
                service.close()

    def test_mode_endpoint_rejects_offload_from_web_and_only_switches_modes(self):
        calls = []
        with tempfile.TemporaryDirectory() as td:
            service = ImageSaveService(pathlib.Path(td), 'http://127.0.0.1:18436',
                mode_status=lambda: {'vision': False, 'ready': True},
                request_mode=lambda mode: (calls.append(mode) or True, 'ok')).start()
            try:
                headers = {'Origin': service.origin, 'X-GGUFRun-Token': service.token, 'Content-Type': 'application/json'}
                for payload in ({'mode': 'img2img', 'offload': False}, {'mode': 'qwen_edit', 'offload': True}):
                    req = urllib.request.Request(service.url + '/mode', data=json.dumps(payload).encode(), headers=headers)
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(req, timeout=4)
                    self.assertEqual(caught.exception.code, 400)
                    caught.exception.close()
                self.assertEqual(calls, [])
                req = urllib.request.Request(service.url + '/mode', data=b'{"mode":"img2img"}', headers=headers)
                with urllib.request.urlopen(req, timeout=4) as response:
                    self.assertEqual(response.status, 202)
                self.assertEqual(calls, ['img2img'])
            finally:
                service.close()

    def test_progress_reports_phase_steps_and_ignores_non_sampling_bars(self):
        sample = ('[INFO   ] image.cpp:810  - generate_image 512x1024\r\n'
                  '  |####          | 27/297 - 1.1GB/s\r\n'
                  '[INFO   ] image.cpp:862  - generating image: 1/1 - seed 11\r\n'
                  '  |====>         | 1/12 - 7.23s/it\r\n'
                  '  |=========>    | 5/12 - 6.61s/it\r\n')
        state = parse_progress(sample, steps=12)
        self.assertEqual((state['phase'], state['step'], state['total']), ('sampling', 5, 12))
        self.assertEqual(state['label'], '取樣 5/12 步')
        self.assertEqual(parse_progress(sample, steps=0)['label'], '取樣中')
        # Bars for tensor loading and VAE decoding reuse `n/n`; only the task's own step
        # total may be read as sampling progress, even when loading lines arrive later.
        noise = ('[INFO   ] image.cpp:810  - generate_image 512x1024\r\n'
                 '[INFO   ] image.cpp:862  - generating image: 1/1 - seed 11\r\n'
                 '  |=========>    | 7/12 - 6.83s/it\r\n'
                 '[INFO   ] model_loader.cpp:1309 - loading tensors completed, taking 3.53s\r\n'
                 '  |##            | 4/297 - 900MB/s\r\n')
        state = parse_progress(noise, steps=12)
        self.assertEqual((state['phase'], state['step']), ('sampling', 7))
        # A finished request in the tail must not be reported as the current phase.
        finished = noise + ('[INFO   ] image.cpp:895  - sampling completed, taking 18.88s\r\n'
                            '[INFO   ] image.cpp:616  - latent 1 decoded, taking 2.62s\r\n'
                            '[INFO   ] image.cpp:1046 - generate_image completed in 29.62s\r\n')
        state = parse_progress(finished, steps=12)
        self.assertEqual((state['phase'], state['step']), ('done', 0))
        for text, phase in (('[INFO   ] image_preprocess.cpp:268  - preprocess ref[0]: 512x1024', 'preprocess'),
                            ('[INFO   ] image.cpp:525  - get_learned_condition completed, taking 6.70s', 'conditioning'),
                            ('[INFO   ] image.cpp:396  - encode_first_stage completed, taking 1.04s', 'encode'),
                            ('[INFO   ] diffusion_engine.cpp:773  - loading llm vision', 'preparing'),
                            ('[INFO   ] main.cpp:149  - listening on: http://127.0.0.1:18436', 'ready')):
            self.assertEqual(parse_progress(text, steps=12)['phase'], phase, text)
        state = parse_progress('', steps=12)
        self.assertEqual((state['phase'], state['step'], state['total']), ('unknown', 0, 12))

    def test_progress_endpoint_requires_token_and_reports_phase(self):
        with tempfile.TemporaryDirectory() as td:
            log = pathlib.Path(td) / 'server.log'
            log.write_text('[INFO   ] image.cpp:810  - generate_image 512x1024\r\n'
                           '[INFO   ] image.cpp:862  - generating image: 1/1 - seed 11\r\n'
                           '  |====>         | 3/12 - 6.6s/it\r\n', encoding='utf-8')
            service = ImageSaveService(pathlib.Path(td) / 'out', 'http://127.0.0.1:18436',
                                       log_path=log).start()
            try:
                url = service.url + '/progress?steps=12&token=' + service.token
                with urllib.request.urlopen(urllib.request.Request(url, headers={'Origin': service.origin}), timeout=4) as response:
                    state = json.load(response)
                self.assertEqual((state['phase'], state['step'], state['total']),
                                 ('sampling', 3, 12))
                for candidate in (service.url + '/progress?steps=12',
                                  service.url + '/progress?steps=12&token=wrong'):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(urllib.request.Request(candidate, headers={'Origin': service.origin}), timeout=4)
                    self.assertEqual(caught.exception.code, 403)
                    caught.exception.close()
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(urllib.request.Request(url, headers={'Origin': 'http://evil'}), timeout=4)
                self.assertEqual(caught.exception.code, 403)
                caught.exception.close()
            finally:
                service.close()

    def test_rejects_mismatched_png_dimensions_and_unsafe_filename_fields(self):
        with tempfile.TemporaryDirectory() as td:
            service = ImageSaveService(pathlib.Path(td), 'http://127.0.0.1:18436').start()
            try:
                for bad in ({'width': 512}, {'sampler': '../evil'}, {'steps': 0}):
                    payload = dict(image=base64.b64encode(PNG).decode(), width=1, height=1,
                                   steps=15, cfg_scale=1, seed=42, sampler='euler', duration_ms=1450)
                    payload.update(bad)
                    req = urllib.request.Request(service.url + '/save', data=json.dumps(payload).encode(),
                          headers={'Origin': service.origin, 'X-GGUFRun-Token': service.token,
                                   'Content-Type': 'application/json'}, method='POST')
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(req, timeout=4)
                    caught.exception.close()
                self.assertEqual(list(pathlib.Path(td).glob('*.png')), [])
            finally:
                service.close()

    def test_denies_wrong_origin_token_and_non_png(self):
        with tempfile.TemporaryDirectory() as td:
            service = ImageSaveService(pathlib.Path(td), 'http://127.0.0.1:18436')
            service.start()
            try:
                for origin, token, image in [('http://malicious.invalid', service.token, PNG),
                                             (service.origin, 'wrong', PNG),
                                             (service.origin, service.token, b'not a png')]:
                    req = urllib.request.Request(service.url + '/save',
                          data=json.dumps({'image': base64.b64encode(image).decode()}).encode(),
                          headers={'Origin': origin, 'X-GGUFRun-Token': token, 'Content-Type': 'application/json'},
                          method='POST')
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(req, timeout=4)
                    caught.exception.close()
                self.assertEqual(list(pathlib.Path(td).glob('*.png')), [])
            finally:
                service.close()

    def test_restart_endpoint_requires_authorised_json_and_reports_result(self):
        calls = []
        with tempfile.TemporaryDirectory() as td:
            service = ImageSaveService(pathlib.Path(td), 'http://127.0.0.1:18436',
                request_restart=lambda: (calls.append('restart') or True, '正在重新啟動')).start()
            try:
                url = service.url + '/restart'
                good = {'Origin': service.origin, 'X-GGUFRun-Token': service.token, 'Content-Type': 'application/json'}
                with urllib.request.urlopen(urllib.request.Request(url, data=b'{"reason":"lora"}', headers=good), timeout=4) as response:
                    self.assertEqual(response.status, 202)
                    self.assertEqual(json.load(response)['status'], '正在重新啟動')
                self.assertEqual(calls, ['restart'])
                for headers in ({'Origin': service.origin, 'Content-Type': 'application/json'},
                                {'Origin': 'http://evil', 'X-GGUFRun-Token': service.token, 'Content-Type': 'application/json'},
                                {'Origin': service.origin, 'X-GGUFRun-Token': service.token, 'Content-Type': 'text/plain'}):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(urllib.request.Request(url, data=b'{"reason":"lora"}', headers=headers), timeout=4)
                    self.assertEqual(caught.exception.code, 403)
                    caught.exception.close()
                for payload in (b'{"reason":7}', b'[]', b'{"reason":"' + b'x' * 40 + b'"}'):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(urllib.request.Request(url, data=payload, headers=good), timeout=4)
                    self.assertEqual(caught.exception.code, 400)
                    caught.exception.close()
                self.assertEqual(calls, ['restart'])
                preflight = urllib.request.Request(url, method='OPTIONS',
                            headers={'Origin': service.origin, 'Access-Control-Request-Method': 'POST'})
                with urllib.request.urlopen(preflight, timeout=4) as response:
                    self.assertEqual(response.status, 204)
            finally:
                service.close()

    def test_restart_endpoint_surfaces_refusal_and_unsupported_controller(self):
        with tempfile.TemporaryDirectory() as td:
            service = ImageSaveService(pathlib.Path(td), 'http://127.0.0.1:18436',
                request_restart=lambda: (False, 'Image Server 未執行')).start()
            try:
                headers = {'Origin': service.origin, 'X-GGUFRun-Token': service.token, 'Content-Type': 'application/json'}
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(urllib.request.Request(service.url + '/restart', data=b'{"reason":"lora"}', headers=headers), timeout=4)
                body = json.loads(caught.exception.read().decode('utf-8'))
                self.assertEqual(caught.exception.code, 409)
                caught.exception.close()
                self.assertIn('未執行', body['status'])
            finally:
                service.close()
        with tempfile.TemporaryDirectory() as td:
            plain = ImageSaveService(pathlib.Path(td), 'http://127.0.0.1:18436').start()
            try:
                headers = {'Origin': plain.origin, 'X-GGUFRun-Token': plain.token, 'Content-Type': 'application/json'}
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(urllib.request.Request(plain.url + '/restart', data=b'{"reason":"lora"}', headers=headers), timeout=4)
                self.assertEqual(caught.exception.code, 409)
                caught.exception.close()
            finally:
                plain.close()

    def test_cors_preflight_only_allows_configured_origin(self):
        with tempfile.TemporaryDirectory() as td:
            service = ImageSaveService(pathlib.Path(td), 'http://127.0.0.1:18436')
            service.start()
            try:
                for path in ('/save', '/mode', '/restart'):
                    req = urllib.request.Request(service.url + path, method='OPTIONS',
                                headers={'Origin': service.origin, 'Access-Control-Request-Method': 'POST'})
                    with urllib.request.urlopen(req, timeout=4) as response:
                        self.assertEqual(response.headers['Access-Control-Allow-Origin'], service.origin)
                req = urllib.request.Request(service.url + '/save', method='OPTIONS',
                            headers={'Origin': 'http://evil', 'Access-Control-Request-Method': 'POST'})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(req, timeout=4)
                self.assertEqual(caught.exception.code, 403)
                caught.exception.close()
            finally:
                service.close()
