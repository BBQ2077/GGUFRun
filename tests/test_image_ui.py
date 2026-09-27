import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_SPEC = importlib.util.spec_from_file_location('gguf_ui_image_hook', ROOT / 'gguf-ui.py')
gguf_ui = importlib.util.module_from_spec(MAIN_SPEC)
MAIN_SPEC.loader.exec_module(gguf_ui)
SPEC = importlib.util.spec_from_file_location('image_ui', ROOT / 'image-ui.py')
image_ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(image_ui)


class LayoutTests(unittest.TestCase):
    def test_llm_scan_uses_llm_models_only_and_runtime_subdirectories(self):
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            (base / 'LLM-MODELS').mkdir()
            (base / 'IMAGE-MODELS').mkdir()
            (base / 'LLM-MODELS' / 'my-model.gguf').write_bytes(b'GGUF')
            (base / 'IMAGE-MODELS' / 'diffusion.gguf').write_bytes(b'GGUF')
            (base / 'RUNTIMES' / 'llama-official').mkdir(parents=True)
            (base / 'RUNTIMES' / 'llama-official' / 'llama-server.exe').touch()
            with mock.patch.object(gguf_ui, 'BASE', str(base)), mock.patch.object(gguf_ui, 'LLM_MODELS', str(base / 'LLM-MODELS')), mock.patch.object(gguf_ui, 'RUNTIME_ROOT', str(base / 'RUNTIMES')), mock.patch.object(gguf_ui, 'REGISTRY', str(base / 'models.json')):
                self.assertEqual([x[0] for x in gguf_ui.list_models()], ['my-model.gguf'])
                self.assertEqual(gguf_ui.first_model_path(), str(base / 'LLM-MODELS' / 'my-model.gguf'))
                self.assertEqual(gguf_ui.find_runtimes(), [('llama-official/', str(base / 'RUNTIMES' / 'llama-official'))])


class ImageHookTests(unittest.TestCase):
    def test_main_app_launches_separate_image_window(self):
        app = gguf_ui.App.__new__(gguf_ui.App)
        with mock.patch.object(gguf_ui.subprocess, 'Popen') as popen:
            app.open_image()
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], gguf_ui.sys.executable)
        self.assertEqual(argv[1], str(ROOT / 'image-ui.py'))


class ImageRuntimeTests(unittest.TestCase):
    def test_fused_viggle_lora_directory_is_available_only_when_installed(self):
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            cmd = image_ui.build_server_cmd(base, 18436)
            self.assertNotIn('--lora-model-dir', cmd)
            adapter = base / 'IMAGE-MODELS/loras/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128-fused-gguf.safetensors'
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(b'fixture')
            cmd = image_ui.build_server_cmd(base, 18436)
            self.assertEqual(cmd[cmd.index('--lora-model-dir') + 1], str(adapter.parent))
            self.assertIn('--offload-to-cpu', cmd)

    def test_settings_path_is_resolved_when_called_not_at_import(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / 'image-settings.json'
            path.write_text('{"port": "25791"}', encoding='utf-8')
            with mock.patch.object(image_ui, 'SETTINGS', path):
                self.assertEqual(image_ui.load_settings()['port'], '25791')

    def test_save_settings_uses_redirected_path_without_touching_project(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / 'image-settings.json'
            with mock.patch.object(image_ui, 'SETTINGS', path), \
                 mock.patch.object(pathlib.Path, 'write_text', autospec=True) as write, \
                 mock.patch.object(image_ui.os, 'replace') as replace:
                image_ui.save_settings({'port': '25791'})
            self.assertEqual(write.call_args.args[0], path.with_suffix('.json.tmp'))
            self.assertEqual(replace.call_args.args, (path.with_suffix('.json.tmp'), path))

    def test_scan_only_image_runtimes_and_models(self):
        with tempfile.TemporaryDirectory() as td:
            b = pathlib.Path(td)
            for name in ('RUNTIMES/llama/llama-server.exe', 'RUNTIMES/sd-cpu/sd-server.exe',
                         'RUNTIMES/sd-cuda/sd-server.exe', 'IMAGE-MODELS/a.gguf',
                         'IMAGE-MODELS/sub/b.safetensors', 'LLM-MODELS/no.gguf'):
                p = b / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            self.assertEqual([n for n, _ in image_ui.find_image_runtimes(b)], ['sd-cpu/', 'sd-cuda/'])
            self.assertEqual([p.name for p in image_ui.find_image_models(b)], ['a.gguf', 'b.safetensors'])

    def test_custom_paths_are_forwarded_without_preflight_file_checks(self):
        with tempfile.TemporaryDirectory() as td:
            b = pathlib.Path(td)
            missing = b / 'not-downloaded.gguf'
            cmd = image_ui.build_server_cmd(b, 18436, True, runtime='sd-cuda/',
                       models={'diffusion': str(missing), 'llm': '', 'vae': ''})
            self.assertEqual(cmd[0], str(b / 'RUNTIMES/sd-cuda/sd-server.exe'))
            self.assertEqual(cmd[cmd.index('--diffusion-model') + 1], str(missing))
            self.assertNotIn('--llm', cmd)
            self.assertNotIn('--vae', cmd)
            self.assertIn('--serve-html-path', cmd)
            self.assertEqual(cmd[cmd.index('--listen-port') + 1], '18436')
            self.assertIn('--offload-to-cpu', cmd)

    def test_full_checkpoint_and_extra_encoder_fields_are_supported(self):
        b = pathlib.Path('D:/ExampleProject')
        cmd = image_ui.build_server_cmd(b, 18436, False, runtime='custom/', models={
            'model': 'C:/outside/full.safetensors', 'clip_l': 'C:/outside/clip.gguf',
            'clip_g': 'C:/outside/clipg.gguf', 't5xxl': 'C:/outside/t5.gguf'})
        self.assertIn('--model', cmd)
        self.assertNotIn('--diffusion-model', cmd)
        for flag in ('--clip_l', '--clip_g', '--t5xxl'):
            self.assertIn(flag, cmd)
        self.assertNotIn('--offload-to-cpu', cmd)

    def test_spectrum_cache_is_opt_in_and_validated(self):
        baseline = image_ui.build_server_cmd(ROOT, 18436, cache_mode='off')
        self.assertNotIn('--cache-mode', baseline)
        self.assertNotIn('--cache-option', baseline)
        spectrum = image_ui.build_server_cmd(ROOT, 18436, cache_mode='spectrum')
        self.assertEqual(spectrum[spectrum.index('--cache-mode') + 1], 'spectrum')
        with self.assertRaises(ValueError):
            image_ui.build_server_cmd(ROOT, 18436, cache_mode='unknown')

    def test_spectrum_w_defaults_to_010_and_is_configurable(self):
        """Only Spectrum accepts W; the shipped default is the quality-safe 0.10."""
        self.assertEqual(image_ui.format_spectrum_w('0.10'), '0.1')
        default = image_ui.build_server_cmd(ROOT, 18436, cache_mode='spectrum')
        self.assertEqual(default[default.index('--cache-option') + 1], 'w=0.1')
        self.assertEqual(float(image_ui.SPECTRUM_W_DEFAULT), 0.10)
        custom = image_ui.build_server_cmd(ROOT, 18436, cache_mode='spectrum', spectrum_w='0.25')
        self.assertEqual(custom[custom.index('--cache-option') + 1], 'w=0.25')
        # Integer-ish and padded input normalises to a compact decimal.
        normalised = image_ui.build_server_cmd(ROOT, 18436, cache_mode='spectrum', spectrum_w='0.20')
        self.assertEqual(normalised[normalised.index('--cache-option') + 1], 'w=0.2')
        for value in ('0', '0.04', '1.01', 'abc', '', None, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                image_ui.build_server_cmd(ROOT, 18436, cache_mode='spectrum', spectrum_w=value)

    def test_other_cache_modes_never_receive_w(self):
        """W is Spectrum-only: other accelerators must not leak the option."""
        for mode in ('easycache', 'taylorseer', 'cache-dit'):
            with self.subTest(mode=mode):
                cmd = image_ui.build_server_cmd(ROOT, 18436, cache_mode=mode)
                self.assertEqual(cmd[cmd.index('--cache-mode') + 1], mode)
                self.assertNotIn('--cache-option', cmd)
        for mode in image_ui.CACHE_MODES:
            with self.subTest(mode=mode):
                cmd = image_ui.build_server_cmd(ROOT, 18436, cache_mode=mode)
                if mode == 'off':
                    self.assertNotIn('--cache-mode', cmd)
                else:
                    self.assertIn('--cache-mode', cmd)

    def test_flash_attention_and_conditioning_cache_options(self):
        plain = image_ui.build_server_cmd(ROOT, 18436)
        self.assertNotIn('--diffusion-fa', plain)
        self.assertNotIn('--conditioning-cache-size', plain)
        enabled = image_ui.build_server_cmd(ROOT, 18436, cache_mode='spectrum',
                        diffusion_fa=True, conditioning_cache_size='8')
        self.assertIn('--diffusion-fa', enabled)
        self.assertEqual(enabled[enabled.index('--conditioning-cache-size')+1], '8')
        self.assertEqual(enabled[enabled.index('--cache-mode')+1], 'spectrum')
        for size in ('0', '4', '12', '128', '2147483647'):
            cmd = image_ui.build_server_cmd(ROOT, 18436, conditioning_cache_size=size)
            self.assertEqual(cmd[cmd.index('--conditioning-cache-size')+1], size)
        for value in ('-1', '2147483648', 'nope', '4.5', '', ' 4', '04', 4, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                image_ui.build_server_cmd(ROOT, 18436, conditioning_cache_size=value)

    def test_conditioning_cache_runtime_capability_check(self):
        import subprocess
        with mock.patch.object(image_ui.subprocess, 'run') as run:
            run.return_value.returncode = 0
            run.return_value.stdout = '--conditioning-cache-size <int> option'
            run.return_value.stderr = ''
            self.assertTrue(image_ui.supports_conditioning_cache('server.exe'))
            run.return_value.stdout = '--cache-mode spectrum'
            self.assertFalse(image_ui.supports_conditioning_cache('server.exe'))
            run.side_effect = subprocess.TimeoutExpired('server.exe', 5)
            self.assertFalse(image_ui.supports_conditioning_cache('server.exe'))

    def test_mmproj_is_explicit_and_only_enabled_for_qwen_edit(self):
        b = pathlib.Path('D:/ExampleProject')
        models = {'diffusion': 'IMAGE-MODELS/model.gguf', 'llm': 'IMAGE-MODELS/llm.gguf',
                  'vae': 'IMAGE-MODELS/vae.safetensors',
                  'llm_vision': 'IMAGE-MODELS/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf'}
        plain = image_ui.build_server_cmd(b, 18436, models=models)
        edit = image_ui.build_server_cmd(b, 18436, models=models, vision=True)
        self.assertNotIn('--llm_vision', plain)
        self.assertEqual(edit[edit.index('--llm_vision') + 1], str(b / models['llm_vision']))
        with self.assertRaises(ValueError):
            image_ui.build_server_cmd(b, 18436, models={'llm_vision': ''}, vision=True)

    def test_widgets_grey_spectrum_w_outside_spectrum_and_pass_it_through(self):
        """W entry is only editable for spectrum; start() forwards the value for Spectrum only."""
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            settings = base / 'image-settings.json'
            fake = mock.Mock(pid=111)
            fake.poll.return_value = None
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', settings), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui.subprocess, 'Popen', return_value=fake) as popen:
                root = tk.Tk()
                try:
                    app = image_ui.ImageApp(root)
                    app.offload.set(False)
                    self.assertEqual(app.spectrum_w.get(), '0.10')
                    self.assertEqual(str(app.spectrum_w_entry['state']), 'disabled')
                    app.cache_mode.set('spectrum')
                    self.assertEqual(str(app.spectrum_w_entry['state']), 'normal')
                    app.spectrum_w.set('0.05')
                    app.start()
                    cmd = popen.call_args.args[0]
                    self.assertEqual(cmd[cmd.index('--cache-option') + 1], 'w=0.05')
                    self.assertEqual(json.loads(settings.read_text(encoding='utf-8'))['spectrum_w'], '0.05')
                    app.stop()
                    app.cache_mode.set('taylorseer')
                    self.assertEqual(str(app.spectrum_w_entry['state']), 'disabled')
                    app.start()
                    cmd = popen.call_args.args[0]
                    self.assertEqual(cmd[cmd.index('--cache-mode') + 1], 'taylorseer')
                    self.assertNotIn('--cache-option', cmd)
                finally:
                    if 'app' in locals():
                        app.close()
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_invalid_spectrum_w_blocks_start_and_surfaces_error(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            settings = base / 'image-settings.json'
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', settings), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui.subprocess, 'Popen') as popen:
                root = tk.Tk()
                try:
                    app = image_ui.ImageApp(root)
                    app.offload.set(False)
                    app.cache_mode.set('spectrum')
                    app.spectrum_w.set('9')
                    app.start()
                    popen.assert_not_called()
                    self.assertIn('0.05', app.status.cget('text'))
                    self.assertFalse(settings.exists())
                finally:
                    if 'app' in locals():
                        app.close()
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_mode_switch_keeps_save_service_and_updates_process_flags(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            vision = base / 'IMAGE-MODELS' / 'vision.gguf'
            vision.parent.mkdir(parents=True)
            vision.write_bytes(b'GGUF')
            settings = base / 'image-settings.json'
            fake_proc = mock.Mock(pid=111, returncode=0)
            fake_proc.poll.side_effect = [None, 0]
            replacement = mock.Mock(pid=222)
            replacement.poll.return_value = None
            plain_proc = mock.Mock(pid=333)
            plain_proc.poll.return_value = None
            callbacks = []
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', settings), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui.subprocess, 'Popen', side_effect=[fake_proc, replacement, plain_proc]) as popen:
                root = tk.Tk()
                try:
                    app = image_ui.ImageApp(root)
                    app.models['llm_vision'].set(str(vision))
                    app.offload.set(False)
                    app.cache_mode.set('spectrum')
                    app.diffusion_fa.set(True)
                    app.conditioning_cache_size.set('default')
                    app.start()
                    self.assertEqual(popen.call_args.args[0][popen.call_args.args[0].index('--cache-mode') + 1], 'spectrum')
                    self.assertEqual(json.loads(settings.read_text(encoding='utf-8'))['cache_mode'], 'spectrum')
                    self.assertIn('--diffusion-fa', popen.call_args.args[0])
                    self.assertTrue(json.loads(settings.read_text(encoding='utf-8'))['diffusion_fa'])
                    app.diffusion_fa.set(False)
                    app.cache_mode.set('off')  # Unsaved UI change must not affect a mode switch in flight.
                    saver = app.save_service
                    with mock.patch.object(root, 'after', side_effect=lambda _ms, fn: callbacks.append(fn)):
                        accepted, _ = app.request_mode('qwen_edit')
                        self.assertTrue(accepted)
                        callbacks.pop(0)()
                        self.assertTrue(app.switching)
                        callbacks.pop(0)()
                    self.assertIs(app.save_service, saver)
                    self.assertTrue(app.vision_loaded)
                    self.assertFalse(app.switching)
                    argv = popen.call_args.args[0]
                    self.assertEqual(argv[argv.index('--llm_vision') + 1], str(vision))
                    self.assertEqual(argv[argv.index('--cache-mode') + 1], 'spectrum')
                    self.assertIn('--diffusion-fa', argv)
                    self.assertNotIn('--offload-to-cpu', argv)
                    replacement.poll.side_effect = [None, 0]
                    with mock.patch.object(root, 'after', side_effect=lambda _ms, fn: callbacks.append(fn)):
                        accepted, _ = app.request_mode('txt2img')
                        self.assertTrue(accepted)
                        callbacks.pop(0)()
                        callbacks.pop(0)()
                    self.assertIs(app.save_service, saver)
                    self.assertFalse(app.vision_loaded)
                    self.assertNotIn('--llm_vision', popen.call_args.args[0])
                    self.assertIn('--cache-mode', popen.call_args.args[0])
                    self.assertNotIn('--offload-to-cpu', popen.call_args.args[0])
                    self.assertFalse(app.mode_status()['offload'])
                    app.stop()
                    self.assertIsNone(app.save_service)
                finally:
                    if 'app' in locals():
                        app.close()
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_supported_conditioning_cache_survives_mode_switch(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            vision = base / 'vision.gguf'
            vision.touch()
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', base / 'image-settings.json'), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui, 'supports_conditioning_cache', return_value=True), \
                 mock.patch.object(image_ui.subprocess, 'Popen') as popen:
                root = tk.Tk()
                try:
                    app = image_ui.ImageApp(root)
                    self.assertFalse(app.diffusion_fa.get())
                    self.assertEqual(app.conditioning_cache_size.get(), 'default')
                    app.models['llm_vision'].set(str(vision))
                    app.conditioning_box.delete(0, 'end')
                    app.conditioning_box.insert(0, '12')
                    self.assertEqual(app.conditioning_cache_size.get(), '12')
                    app.cache_mode.set('spectrum')
                    app.diffusion_fa.set(True)
                    app.start()
                    argv = popen.call_args.args[0]
                    self.assertEqual(argv[argv.index('--conditioning-cache-size')+1], '12')
                    self.assertEqual(json.loads((base/'image-settings.json').read_text(encoding='utf-8'))['conditioning_cache_size'], '12')
                    self.assertIn('--diffusion-fa', argv)
                    self.assertEqual(str(app.conditioning_box.cget('state')), 'disabled')
                    app.conditioning_cache_size.set('default')
                    proc = app.proc
                    proc.poll.side_effect = [None, 0]
                    callbacks = []
                    with mock.patch.object(root, 'after', side_effect=lambda _ms, fn: callbacks.append(fn)):
                        self.assertTrue(app.request_mode('qwen_edit')[0])
                        callbacks.pop(0)()
                        callbacks.pop(0)()
                    argv = popen.call_args.args[0]
                    self.assertEqual(argv[argv.index('--conditioning-cache-size')+1], '12')
                    self.assertIn('--diffusion-fa', argv)
                    app.proc.poll.side_effect = None
                    app.proc.poll.return_value = 0
                    app.stop()
                    self.assertEqual(str(app.conditioning_box.cget('state')), 'normal')
                    app.close()
                finally:
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_numeric_conditioning_setting_reloads_and_invalid_entry_is_blocked(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            path = base / 'image-settings.json'
            path.write_text(json.dumps({'conditioning_cache_size': '12'}), encoding='utf-8')
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', path), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui.subprocess, 'Popen') as popen:
                root = tk.Tk()
                try:
                    app = image_ui.ImageApp(root)
                    self.assertEqual(app.conditioning_cache_size.get(), '12')
                    app.conditioning_box.delete(0, 'end')
                    app.conditioning_box.insert(0, '-3')
                    app.start()
                    popen.assert_not_called()
                    self.assertIn('default', str(app.status.cget('text')))
                    self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['conditioning_cache_size'], '12')
                    app.conditioning_cache_size.set('12')
                    app.close()
                finally:
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_unsupported_conditioning_cache_does_not_start_or_save(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', base / 'image-settings.json'), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui, 'supports_conditioning_cache', return_value=False), \
                 mock.patch.object(image_ui.subprocess, 'Popen') as popen:
                root = tk.Tk()
                try:
                    app = image_ui.ImageApp(root)
                    app.conditioning_cache_size.set('4')
                    app.start()
                    popen.assert_not_called()
                    self.assertFalse((base / 'image-settings.json').exists())
                    self.assertIn('不支援', app.status.cget('text'))
                    app.conditioning_cache_size.set('default')
                    app.close()
                finally:
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_invalid_ports_are_rejected(self):
        for port in (0, 65536, 'bad'):
            with self.subTest(port=port), self.assertRaises(ValueError):
                image_ui.build_server_cmd(ROOT, port, False)

    def test_log_reader_shows_recent_history_and_new_content(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / 'server.log'
            path.write_bytes(b'old\n' * 15000 + '最近一行\n'.encode('utf-8'))
            text, position = image_ui.read_log_delta(path, None)
            self.assertIn('最近一行', text)
            self.assertLess(len(text), 30000)
            with path.open('ab') as out:
                out.write('新增錯誤\n'.encode('utf-8'))
            extra, position = image_ui.read_log_delta(path, position)
            self.assertEqual(extra, '新增錯誤\n')
            self.assertEqual(image_ui.read_log_delta(path, position), ('', position))

    def test_large_log_append_keeps_initial_error_and_recent_tail(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / 'server.log'
            path.write_bytes(b'previous')
            _, pos = image_ui.read_log_delta(path, None)
            with path.open('ab') as out:
                out.write(b'\n[ERROR] missing model\n' + b'help text\n' * 5000 + b'\nfinal diagnostic\n')
            text, end = image_ui.read_log_delta(path, pos)
            self.assertIn('[ERROR] missing model', text)
            self.assertIn('final diagnostic', text)
            self.assertLess(len(text), 26000)
            self.assertEqual(end, path.stat().st_size)

    def test_log_reader_recovers_from_truncation_and_absent_log(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / 'server.log'
            self.assertEqual(image_ui.read_log_delta(path, None), ('', 0))
            path.write_text('abcdef', encoding='utf-8')
            _, position = image_ui.read_log_delta(path, None)
            path.write_text('短', encoding='utf-8')
            text, position = image_ui.read_log_delta(path, position)
            self.assertEqual(text, '短')
            self.assertEqual(position, len('短'.encode('utf-8')))

    def test_image_controller_starts_and_stops_save_service(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', base / 'image-settings.json'), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui.subprocess, 'Popen') as popen:
                proc = popen.return_value
                proc.poll.return_value = 0
                proc.pid = 1234
                root = tk.Tk()
                try:
                    app = image_ui.ImageApp(root)
                    app.start()
                    self.assertIsNotNone(app.save_service)
                    self.assertTrue(app.save_service.url.startswith('http://127.0.0.1:'))
                    app.stop()
                    self.assertIsNone(app.save_service)
                finally:
                    if 'app' in locals():
                        app.close()
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_opening_controller_clears_old_log_when_port_is_free(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            path = base / 'image-output/server.log'
            path.parent.mkdir()
            path.write_text('OLD SESSION FAILURE\n', encoding='utf-8')
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', base / 'image-settings.json'), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False):
                root = tk.Tk()
                try:
                    app = image_ui.ImageApp(root)
                    self.assertEqual(path.read_text(encoding='utf-8'), '')
                    self.assertNotIn('OLD SESSION FAILURE', app.log.get('1.0', 'end'))
                    app.close()
                finally:
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_busy_port_does_not_truncate_or_stream_another_servers_log(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            path = base / 'image-output/server.log'
            path.parent.mkdir()
            path.write_text('EXTERNAL RUN\n', encoding='utf-8')
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', base / 'image-settings.json'), \
                 mock.patch.object(image_ui, 'port_busy', return_value=True):
                root = tk.Tk()
                try:
                    app = image_ui.ImageApp(root)
                    with path.open('a', encoding='utf-8') as f:
                        f.write('EXTERNAL NEW LINE\n')
                    app.poll()
                    self.assertIn('EXTERNAL RUN', path.read_text(encoding='utf-8'))
                    self.assertEqual(app.log.get('1.0', 'end').strip(), '')
                    app.close()
                finally:
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_restarting_server_in_same_window_starts_clean_log(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', base / 'image-settings.json'), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False):
                root = tk.Tk()
                try:
                    app = image_ui.ImageApp(root)
                    path = base / 'image-output/server.log'
                    path.write_text('OLD RUN\n', encoding='utf-8')
                    app.refresh_log()
                    with mock.patch.object(image_ui.subprocess, 'Popen', side_effect=OSError('test launch failed')):
                        app.start()
                    text = path.read_text(encoding='utf-8')
                    self.assertNotIn('OLD RUN', text)
                    self.assertNotIn('OLD RUN', app.log.get('1.0', 'end'))
                    self.assertIn('test launch failed', text)
                    self.assertIn('test launch failed', app.log.get('1.0', 'end'))
                    app.close()
                finally:
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_log_visible_inside_image_controller(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td, mock.patch.object(image_ui, 'BASE', pathlib.Path(td)), mock.patch.object(image_ui, 'SETTINGS', pathlib.Path(td) / 'image-settings.json'):
            root = tk.Tk()
            try:
                app = image_ui.ImageApp(root)
                root.update()
                path = pathlib.Path(td) / 'image-output/server.log'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('啟動失敗：模型缺檔\n', encoding='utf-8')
                app.refresh_log()
                self.assertIn('模型缺檔', app.log.get('1.0', 'end'))
                self.assertEqual(app.log.cget('state'), 'disabled')
                app.close()
                self.assertTrue((pathlib.Path(td) / 'image-settings.json').is_file())
            finally:
                try:
                    root.destroy()
                except tk.TclError:
                    pass

    def test_quick_sizes_defaults_and_clean_elapsed_label(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        for fragment in ('id="size-preset"', 'value="512x1024" selected',
                         '1024×512', '512×1024', '640×384', '864×480', '1280×736',
                         '1920×1088', '2560×1440', '3840×2176',
                         'id="steps" type="number" min="1" max="150" value="20"',
                         'id="cfg" type="number" min="0" max="30" step="0.1" value="1"'):
            self.assertIn(fragment, html)
        self.assertNotIn('無準確百分比', html)

    def test_quick_sizes_are_32_aligned_and_custom_inputs_step_by_32(self):
        import re
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        values = re.findall(r'<option value="(\d+)x(\d+)"', html)
        self.assertGreaterEqual(len(values), 9)
        for w, h in values:
            self.assertEqual(int(w) % 32, 0, (w, h))
            self.assertEqual(int(h) % 32, 0, (w, h))
        for name in ('width', 'height'):
            self.assertRegex(html, rf'id="{name}" type="number" min="64" max="4096" step="32"')
        self.assertIn('value="1920x1088"', html)

    def test_img2img_uses_blob_store_and_one_image_per_task(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        for fragment in ('id="mode"', 'id="source-file"', 'id="denoising-strength"',
                         'indexedDB.open', 'init_images=[', "'/sdapi/v1/img2img'",
                         'denoising_strength', 'sourceKey'):
            self.assertIn(fragment, html)

    def test_qwen_multi_reference_ui_and_cleanup(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        for fragment in ('id="edit-files"', 'multiple>', 'id="edit-list"', 'id="clear-edit"',
                         'body.sourceKeys', 'ref_images:', 'selectedEditRefs',
                         't.body?.sourceKeys?.includes(key)'):
            self.assertIn(fragment, html)

    def test_qwen_edit_uses_native_img_gen_without_init_image(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        self.assertNotIn('/v1/images/edits', html)
        for fragment in ("'/sdcpp/v1/img_gen'", 'init_image:null', 'ref_images:refs',
                         'sample_params', 'txt_cfg', 'poll_url', '\\/sdcpp\\/v1\\/jobs\\/'):
            self.assertIn(fragment, html)
        self.assertIn("'/sdapi/v1/img2img'", html)

    def test_image_controller_heading_uses_legible_ui_font(self):
        text = (ROOT / 'image-ui.py').read_text(encoding='utf-8')
        self.assertIn("font=('Microsoft JhengHei UI', 10)", text)
        self.assertNotIn("font=('', 11, 'bold')", text)

    def test_browser_prefers_x86_chrome_with_light_profile(self):
        from types import SimpleNamespace
        app = SimpleNamespace(port=SimpleNamespace(get=lambda: '18436'),
                              proc=SimpleNamespace(poll=lambda: None),
                              save_service=SimpleNamespace(httpd=SimpleNamespace(server_port=55123), token='a'*32),
                              status=SimpleNamespace(config=lambda **kw: None))
        chrome = r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
        edge = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
        with mock.patch.object(image_ui, 'port_busy', return_value=True), \
             mock.patch.object(image_ui.os.path, 'isfile', side_effect=lambda path: path in (chrome, edge)), \
             mock.patch.object(image_ui.subprocess, 'Popen') as popen:
            image_ui.ImageApp.open_web(app)
        args = popen.call_args.args[0]
        self.assertEqual(args[0], chrome)
        for flag in ('--disable-extensions', '--disable-background-networking', '--no-first-run', '--no-default-browser-check'):
            self.assertIn(flag, args)
        self.assertTrue(any('_image_chrome_profile' in arg for arg in args))
        self.assertTrue(any(arg.startswith('--app=http://127.0.0.1:18436/?') for arg in args))

    def test_browser_uses_existing_edge_profile_only_if_chrome_missing(self):
        from types import SimpleNamespace
        app = SimpleNamespace(port=SimpleNamespace(get=lambda: '18436'),
                              proc=SimpleNamespace(poll=lambda: None),
                              save_service=SimpleNamespace(httpd=SimpleNamespace(server_port=55123), token='a'*32),
                              status=SimpleNamespace(config=lambda **kw: None))
        edge = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
        with mock.patch.object(image_ui, 'port_busy', return_value=True), \
             mock.patch.object(image_ui.os.path, 'isfile', side_effect=lambda path: path == edge), \
             mock.patch.object(image_ui.subprocess, 'Popen') as popen:
            image_ui.ImageApp.open_web(app)
        args = popen.call_args.args[0]
        self.assertEqual(args[0], edge)
        self.assertTrue(any('_image_webview_profile' in arg for arg in args))

    def test_frontend_prioritizes_image_and_auto_save(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        self.assertLess(html.index('id="run-queue"'), html.index('id="preview"'))
        self.assertLess(html.index('id="preview"'), html.index('id="tasklist"'))
        self.assertIn('id="elapsed"', html)
        self.assertIn('save_port', html)
        self.assertIn("'/save'", html)
        self.assertIn('await saveResponse.json()', html)
        self.assertIn('image-output', html)

    def test_frontend_does_not_restore_last_image_on_load(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        self.assertNotIn('lastSaved', html)
        self.assertNotIn('ggufrun.image.last', html)
        self.assertIn('開頁一律空白', html)

    def test_hi_res_is_collapsed_behind_explicit_opt_in(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        self.assertIn('<details id="hires-fields">', html)
        self.assertIn('進階設定', html)
        self.assertNotIn('id="enable-hr" type="checkbox" checked', html)
        for control in ('hr-upscaler', 'hr-scale', 'hr-steps', 'hr-denoise'):
            self.assertIn(f'id="{control}"', html)
        self.assertGreaterEqual(html.count('disabled>'), 4, 'Hi-res 子欄位預設須為停用')
        self.assertIn('syncHrOptions()', html)
        self.assertIn("addEventListener('change',syncHrOptions)", html)

    def test_chinese_web_frontend_calls_generation_api_and_explains_parts(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        for term in ('lang="zh-Hant"', '文字生圖', '文字編碼器', 'VAE', '/sdapi/v1/txt2img', '自動保存'):
            self.assertIn(term, html)
        self.assertNotIn('<script src=', html)
    def test_every_generation_parameter_has_a_help_control(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        for name in ('prompt', 'negative', 'width', 'height', 'steps', 'cfg', 'seed', 'sampler'):
            with self.subTest(name=name):
                self.assertIn(f'aria-label="說明：{name}"', html)
        self.assertIn('class="hinttext"', html)

    def test_queue_actions_and_bulk_input_are_present(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        for label in ('批量加入', '開始／繼續', '暫停下一張', '待執行', '中斷待確認', 'queue.start()', 'queue.pause()'):
            self.assertIn(label, html)


if __name__ == '__main__':
    unittest.main()
