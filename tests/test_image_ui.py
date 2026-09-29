import importlib.util
import json
import os
import pathlib
import re
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


_SHARED_ROOT = None


def hidden_root():
    """The one Tk root the whole test run shares.

    A run used to build a fresh, mapped Tk window for every test that needed real
    widgets -- 21 windows flashing open and shut, because ImageApp.close() tears
    its root down. Here the root is created once and handed out again on every
    call: its children are dropped between tests instead of the window itself, so
    the run opens exactly one window and closes it when the process exits. Alpha
    0 keeps that window mapped (winfo_ismapped / winfo_rooty assertions still
    hold) but paints nothing on screen.
    """
    global _SHARED_ROOT
    import tkinter as tk

    def drop_children():
        for child in _SHARED_ROOT.winfo_children():
            try:
                child.destroy()
            except tk.TclError:
                pass

    if _SHARED_ROOT is None or not _SHARED_ROOT.winfo_exists():
        _SHARED_ROOT = tk.Tk()
        try:
            _SHARED_ROOT.attributes('-alpha', 0.0)
        except tk.TclError:
            pass
        _SHARED_ROOT.destroy = drop_children  # never tear the shared window down
    else:
        drop_children()
    return _SHARED_ROOT


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
        """W is Spectrum-only: other accelerators must not leak the w= option.

        sd-cli documents threshold=/warmup= as shared by dbcache/taylorseer/cache-dit,
        so those modes may legitimately carry --cache-option -- but never a spectrum w=.
        """
        for mode in ('easycache', 'taylorseer', 'cache-dit'):
            with self.subTest(mode=mode):
                cmd = image_ui.build_server_cmd(ROOT, 18436, cache_mode=mode)
                self.assertEqual(cmd[cmd.index('--cache-mode') + 1], mode)
                options = cmd[cmd.index('--cache-option') + 1] if '--cache-option' in cmd else ''
                keys = [kv.split('=')[0] for kv in options.split(',') if kv]
                self.assertNotIn('w', keys)
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
                root = hidden_root()
                try:
                    app = image_ui.ImageApp(root)
                    app.offload.set(False)
                    self.assertEqual(app.spectrum_w.get(), '0.10')
                    # Fields stay editable in every mode (no lockout); only the hint changes.
                    self.assertEqual(str(app.spectrum_w_entry['state']), 'normal')
                    app.cache_mode.set('spectrum')
                    self.assertEqual(str(app.spectrum_w_entry['state']), 'normal')
                    app.spectrum_w.set('0.05')
                    app.start()
                    cmd = popen.call_args.args[0]
                    self.assertEqual(cmd[cmd.index('--cache-option') + 1], 'w=0.05')
                    self.assertEqual(json.loads(settings.read_text(encoding='utf-8'))['spectrum_w'], '0.05')
                    app.stop()
                    app.cache_mode.set('taylorseer')
                    self.assertEqual(str(app.spectrum_w_entry['state']), 'normal')
                    app.start()
                    cmd = popen.call_args.args[0]
                    self.assertEqual(cmd[cmd.index('--cache-mode') + 1], 'taylorseer')
                    opts = cmd[cmd.index('--cache-option') + 1] if '--cache-option' in cmd else ''
                    self.assertNotIn('w', [kv.split('=')[0] for kv in opts.split(',') if kv])
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
                root = hidden_root()
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
                root = hidden_root()
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
                    # The page holds this service's port and token: a restart must not drop it.
                    self.assertIs(app.save_service, saver)
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
                root = hidden_root()
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
                root = hidden_root()
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
                root = hidden_root()
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

    def test_image_controller_keeps_the_save_service_alive_across_a_restart(self):
        """The page carries the save service's port and token, so /mode must survive a restart."""
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
                root = hidden_root()
                try:
                    app = image_ui.ImageApp(root)
                    app.start()
                    service = app.save_service
                    self.assertIsNotNone(service)
                    self.assertTrue(service.url.startswith('http://127.0.0.1:'))
                    app.stop()
                    self.assertIs(app.save_service, service)  # open page keeps working
                    app.start()  # ...including after the server comes back
                    self.assertIs(app.save_service, service)
                    self.assertEqual((app.save_service.httpd.server_port, app.save_service.token),
                                     (service.httpd.server_port, service.token))
                    app.close()
                    self.assertIsNone(app.save_service)
                finally:
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
                root = hidden_root()
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
                root = hidden_root()
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
                root = hidden_root()
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
            root = hidden_root()
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
                         '512×1024（直式，預設）', '1024×512（橫式）', '512×512（方形）',
                         '約 360p · 384×640（直式）', '約 360p · 640×384（橫式）',
                         '約 480p · 480×864（直式）', '約 480p · 864×480（橫式）',
                         '約 720p · 736×1280（直式）', '約 720p · 1280×736（橫式）',
                         '約 1080p · 1088×1920（直式）', '約 1080p · 1920×1088（橫式）',
                         '2K · 1440×2560（直式，高顯存）', '2K · 2560×1440（橫式，高顯存）',
                         '約 4K · 2176×3840（直式，極高顯存）', '約 4K · 3840×2176（橫式，極高顯存）',
                         'id="steps" type="number" min="1" max="150" value="20"',
                         'id="cfg" type="number" min="0" max="30" step="0.1" value="1"'):
            self.assertIn(fragment, html)
        self.assertNotIn('無準確百分比', html)

    def test_quick_sizes_are_32_aligned_and_custom_inputs_step_by_32(self):
        import re
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        values = [(int(w), int(h)) for w, h in re.findall(r'<option value="(\d+)x(\d+)"', html)]
        self.assertGreaterEqual(len(values), 15)
        for w, h in values:
            self.assertEqual(w % 32, 0, (w, h))
            self.assertEqual(h % 32, 0, (w, h))
        # 每個常用規格都要有直式與橫式：橫式就是這組數字的轉置（方形與 512×1024 那組本來就成對）。
        for w, h in values:
            self.assertIn((h, w), values, f'缺少 {h}×{w}：常用規格必須同時提供直式與橫式')
        self.assertEqual(len(values), len(set(values)), '尺寸選項有重複')
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


class ExtraArgsTests(unittest.TestCase):
    """可選「額外指令」欄位、「保留顯存」數字（預設 2 GiB，送出 --max-vram -N）與「停用 prefix cache」的啟動行為。"""

    def test_extra_args_are_optional_and_appended_verbatim(self):
        plain = image_ui.build_server_cmd(ROOT, 18436)
        self.assertEqual(image_ui.split_extra_args(''), [])
        self.assertEqual(image_ui.split_extra_args('   '), [])
        self.assertNotIn('--lora-apply-mode', plain)
        extra = ['--lora-apply-mode', 'immediately', '-t', '6']
        cmd = image_ui.build_server_cmd(ROOT, 18436, extra_args='--lora-apply-mode immediately -t 6')
        self.assertEqual(cmd, plain + extra)
        self.assertEqual(image_ui.build_server_cmd(ROOT, 18436, extra_args=None), plain)

    def test_quoted_extra_values_stay_one_token(self):
        self.assertEqual(image_ui.split_extra_args('--embd-dir "D:/my models/emb"'),
                         ['--embd-dir', 'D:/my models/emb'])
        self.assertEqual(image_ui.split_extra_args("--embd-dir 'D:/my models/emb'"),
                         ['--embd-dir', 'D:/my models/emb'])
        self.assertEqual(image_ui.split_extra_args('--threads 6   --mmap'), ['--threads', '6', '--mmap'])

    def test_malformed_or_window_managed_extra_args_are_refused(self):
        long_text = 'x' * (image_ui.MAX_EXTRA_ARGS + 1)
        too_many = ' '.join(['--mmap'] * (image_ui.MAX_EXTRA_TOKENS + 1))
        for text in ('"D:/unterminated', '--listen-port 18437', '--offload-to-cpu',
                     '--serve-html-path D:/x.html', '--vae D:/vae.safetensors', long_text, too_many):
            with self.subTest(text=text[:40]), self.assertRaises(ValueError):
                image_ui.build_server_cmd(ROOT, 18436, extra_args=text)
        for value in (6, ['--mmap'], True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                image_ui.split_extra_args(value)

    def test_vram_reserve_number_is_editable_and_defaults_to_no_flag(self):
        self.assertEqual(image_ui.MAX_VRAM_RESERVE_DEFAULT, 0)
        self.assertNotIn('--max-vram', image_ui.build_server_cmd(ROOT, 18436))
        self.assertNotIn('--max-vram', image_ui.build_server_cmd(ROOT, 18436, max_vram_reserve=0))
        for number, flag in ((1, '-1'), (2, '-2'), (5, '-5'), ('12', '-12'), (16, '-16'),
                             (1.5, '-1.5'), ('0.5', '-0.5'), (2.25, '-2.25'), ('3.0', '-3')):
            with self.subTest(number=number):
                cmd = image_ui.build_server_cmd(ROOT, 18436, max_vram_reserve=number)
                self.assertEqual(cmd[-2:], ['--max-vram', flag])
        # The runtime parses the budget as a float, so a fractional reserve is legal on its side.
        for bad in ('yes', True, False, -1, 17, 16.5, '', None, ' 2', '2.', '.5', '1.234', '1,5'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                image_ui.build_server_cmd(ROOT, 18436, max_vram_reserve=bad)
        with self.assertRaises(ValueError):
            image_ui.build_server_cmd(ROOT, 18436, max_vram_reserve=2, extra_args='--max-vram 6')
        # The window's own default reaches the launch command through the settings snapshot.
        self.assertNotIn('--max-vram', image_ui.build_cmd_from_settings(ROOT, 18436, {}))
        # Legacy on/off settings migrate: on keeps the 1 GiB it used to mean, off adopts the new default.
        self.assertEqual(image_ui.build_cmd_from_settings(
            ROOT, 18436, {'max_vram_reserve': True})[-2:], ['--max-vram', '-1'])
        self.assertEqual(image_ui.normalise_vram_reserve(False), 0)
        self.assertEqual(image_ui.normalise_vram_reserve('4'), 4)
        self.assertEqual(image_ui.normalise_vram_reserve('4.0'), 4)   # integral decimals stay ints
        self.assertEqual(image_ui.normalise_vram_reserve('1.5'), 1.5)
        self.assertEqual(image_ui.normalise_vram_reserve('junk'), 0)

    def test_vae_cpu_and_vae_tiling_checkboxes_reach_the_launch_command(self):
        """--backend vae=cpu is the runtime's replacement for the deprecated --vae-on-cpu."""
        self.assertNotIn('--backend', image_ui.build_server_cmd(ROOT, 18436))
        self.assertNotIn('--vae-tiling', image_ui.build_server_cmd(ROOT, 18436))
        self.assertEqual(image_ui.VAE_CPU_FLAG, ('--backend', 'vae=cpu'))
        self.assertEqual(image_ui.VAE_TILING_FLAG, '--vae-tiling')
        cpu = image_ui.build_server_cmd(ROOT, 18436, vae_on_cpu=True)
        self.assertEqual(cpu[cpu.index('--backend'):cpu.index('--backend') + 2], ['--backend', 'vae=cpu'])
        self.assertEqual(image_ui.build_server_cmd(ROOT, 18436, vae_tiling=True)[-1], '--vae-tiling')
        both = image_ui.build_server_cmd(ROOT, 18436, vae_on_cpu=True, vae_tiling=True)
        self.assertEqual(both[-3:], ['--backend', 'vae=cpu', '--vae-tiling'])
        for bad in ('yes', 1, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                image_ui.build_server_cmd(ROOT, 18436, vae_on_cpu=bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                image_ui.build_server_cmd(ROOT, 18436, vae_tiling=bad)
        # Free-form args may still carry them by hand, but never twice at the same time.
        self.assertIn('--backend', image_ui.build_server_cmd(ROOT, 18436, extra_args='--backend clip=cpu'))
        self.assertIn('--vae-tiling', image_ui.build_server_cmd(ROOT, 18436, extra_args='--vae-tiling'))
        for extra in ('--backend clip=cpu', '--vae-on-cpu'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                image_ui.build_server_cmd(ROOT, 18436, vae_on_cpu=True, extra_args=extra)
        with self.assertRaises(ValueError):
            image_ui.build_server_cmd(ROOT, 18436, vae_tiling=True, extra_args='--vae-tiling')
        # The mode switch reuses the saved snapshot, so both flags must survive it.
        switched = image_ui.build_cmd_from_settings(
            ROOT, 18436, {'vae_on_cpu': True, 'vae_tiling': True})
        self.assertEqual(switched[switched.index('--backend'):switched.index('--backend') + 3],
                         ['--backend', 'vae=cpu', '--vae-tiling'])

    def test_vae_tiling_defaults_on_unless_the_saved_settings_turn_it_off(self):
        """分塊解碼預設開啟：沒有存檔、或舊存檔缺這個欄位時都要送 --vae-tiling。"""
        import tkinter as tk
        self.assertIn('--vae-tiling', image_ui.build_cmd_from_settings(ROOT, 18436, {}))
        self.assertNotIn('--vae-tiling', image_ui.build_cmd_from_settings(ROOT, 18436, {'vae_tiling': False}))
        self.assertIn('--vae-tiling', image_ui.build_cmd_from_settings(ROOT, 18436, {'vae_tiling': True}))

        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            settings = base / 'image-settings.json'
            fake = mock.Mock(pid=222)
            fake.poll.return_value = None

            def launch(saved):
                """Start the window against one saved settings file; return (checkbox, command, written)."""
                settings.write_text(json.dumps(saved), encoding='utf-8')
                with mock.patch.object(image_ui, 'BASE', base), \
                     mock.patch.object(image_ui, 'SETTINGS', settings), \
                     mock.patch.object(image_ui, 'port_busy', return_value=False), \
                     mock.patch.object(image_ui.subprocess, 'Popen', return_value=fake) as popen:
                    root = hidden_root()
                    app = image_ui.ImageApp(root)
                    try:
                        app.start()
                        return (app.vae_tiling.get(), popen.call_args.args[0],
                                json.loads(settings.read_text(encoding='utf-8')))
                    finally:
                        app.close()
                        try:
                            root.destroy()
                        except tk.TclError:
                            pass

            checked, cmd, written = launch({})                   # 舊存檔：沒有這個欄位
            self.assertIs(checked, True)
            self.assertIn('--vae-tiling', cmd)
            self.assertIs(written['vae_tiling'], True)
            checked, cmd, written = launch({'vae_tiling': False})  # 使用者明確取消：尊重它
            self.assertIs(checked, False)
            self.assertNotIn('--vae-tiling', cmd)
            self.assertIs(written['vae_tiling'], False)

    def test_hand_written_vae_flags_move_into_the_checkboxes(self):
        """Old free-form spellings are lifted into the new checkboxes instead of being rejected."""
        migrate = image_ui.migrate_vae_flags
        self.assertEqual(migrate('--vae-tiling'), ('', False, True))
        self.assertEqual(migrate('--vae-tiling --auto-fit off'), ('--auto-fit off', False, True))
        self.assertEqual(migrate('--backend vae=cpu'), ('', True, False))
        self.assertEqual(migrate('--backend=vae=cpu --vae-on-cpu'), ('', True, False))
        self.assertEqual(migrate('--backend clip=cpu'), ('--backend clip=cpu', False, False))
        self.assertEqual(migrate('--auto-fit off "a b"'), ('--auto-fit off "a b"', False, False))
        self.assertEqual(migrate(None), ('', False, False))
        self.assertEqual(migrate(''), ('', False, False))

    def test_hires_upscaler_directory_reaches_the_launch_command(self):
        """ESRGAN/Real-ESRGAN weights are found through a directory flag, so it must be wired."""
        self.assertEqual(image_ui.HIRES_UPSCALERS_DIR_FLAG, '--hires-upscalers-dir')
        self.assertEqual(image_ui.HIRES_UPSCALERS_DIR_DEFAULT, 'IMAGE-MODELS/upscalers')
        self.assertNotIn('--hires-upscalers-dir', image_ui.build_server_cmd(ROOT, 18436))
        self.assertNotIn('--hires-upscalers-dir', image_ui.build_cmd_from_settings(ROOT, 18436, {}))
        # Relative values follow the same project-root rule as the model fields.
        relative = image_ui.build_server_cmd(ROOT, 18436, hires_upscalers_dir='assets')
        self.assertEqual(relative[-2:], ['--hires-upscalers-dir', str(ROOT / 'assets')])
        with tempfile.TemporaryDirectory() as td:
            upscalers = pathlib.Path(td) / 'upscalers'
            upscalers.mkdir()
            cmd = image_ui.build_server_cmd(ROOT, 18436, hires_upscalers_dir=str(upscalers))
            self.assertEqual(cmd[-2:], ['--hires-upscalers-dir', str(upscalers)])
            snapshot = image_ui.build_cmd_from_settings(
                ROOT, 18436, {'hires_upscalers_dir': str(upscalers)})
            self.assertEqual(snapshot[-2:], ['--hires-upscalers-dir', str(upscalers)])
            # A wrong path no longer refuses the launch: the flag is dropped and the caller
            # logs why.  Starting beats blocking over a folder typo.
            for bad in ('IMAGE-MODELS/no-such-folder', str(pathlib.Path(td) / 'absent'),
                        str(ROOT / 'README.md')):
                with self.subTest(bad=bad):
                    cmd = image_ui.build_server_cmd(ROOT, 18436, hires_upscalers_dir=bad)
                    self.assertNotIn('--hires-upscalers-dir', cmd)
                    # the caller can report the exact folder it could not use
                    warned = image_ui.hires_dir_warning(bad, ROOT)
                    self.assertIn('放大器目錄不存在', warned)
                    self.assertIn(pathlib.Path(bad).name, warned)
            with self.assertRaises(ValueError):
                image_ui.build_server_cmd(ROOT, 18436, hires_upscalers_dir=str(upscalers),
                                          extra_args='--hires-upscalers-dir D:/other')

    def test_hand_written_hires_upscaler_directory_moves_into_its_field(self):
        """A spelling typed into 「額外指令」 is lifted into the field instead of being rejected."""
        migrate = image_ui.migrate_hires_upscalers_dir
        self.assertEqual(migrate('--hires-upscalers-dir D:/models/upscalers'),
                         ('', 'D:/models/upscalers'))
        self.assertEqual(migrate('--hires-upscalers-dir=D:/models/x --auto-fit off'),
                         ('--auto-fit off', 'D:/models/x'))
        self.assertEqual(migrate('--auto-fit off'), ('--auto-fit off', ''))
        self.assertEqual(migrate('--hires-upscalers-dir'), ('--hires-upscalers-dir', ''))
        self.assertEqual(migrate(None), ('', ''))
        self.assertEqual(migrate(''), ('', ''))

    def test_hires_upscaler_directory_widget_locks_while_running(self):
        """The folder is a launch-time flag: prefilled only when it exists, locked while running."""
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            (base / image_ui.HIRES_UPSCALERS_DIR_DEFAULT).mkdir(parents=True)
            settings = base / 'image-settings.json'
            fake = mock.Mock(pid=222)
            fake.poll.return_value = None
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', settings), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui.subprocess, 'Popen', return_value=fake) as popen:
                root = hidden_root()
                try:
                    app = image_ui.ImageApp(root)
                    self.assertEqual(app.hires_upscalers_dir.get(), image_ui.HIRES_UPSCALERS_DIR_DEFAULT)
                    app.start()
                    cmd = popen.call_args.args[0]
                    self.assertEqual(cmd[cmd.index('--hires-upscalers-dir') + 1],
                                     str(base / image_ui.HIRES_UPSCALERS_DIR_DEFAULT))
                    for widget in (app.hires_dir_entry, app.hires_dir_button, app.hires_dir_clear):
                        self.assertEqual(str(widget['state']), 'disabled')
                    self.assertEqual(json.loads(settings.read_text(encoding='utf-8'))['hires_upscalers_dir'],
                                     image_ui.HIRES_UPSCALERS_DIR_DEFAULT)
                    app.stop()
                    for widget in (app.hires_dir_entry, app.hires_dir_button, app.hires_dir_clear):
                        self.assertEqual(str(widget['state']), 'normal')
                    # Clearing the field is the documented way to stop sending the flag.
                    app.hires_upscalers_dir.set('')
                    app.start()
                    self.assertNotIn('--hires-upscalers-dir', popen.call_args.args[0])
                finally:
                    if 'app' in locals():
                        app.close()
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_absent_hires_upscaler_directory_still_starts_and_logs_the_reason(self):
        """A wrong folder must not block the launch: sd-server ignores it and still serves.

        It used to refuse to start over a typo, which made the whole window unusable for a
        cosmetic mistake.  The flag is now dropped, the reason goes to the log, and the
        server starts normally — the dropdown simply has no external models.
        """
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            (base / 'image-output').mkdir(parents=True, exist_ok=True)
            settings = base / 'image-settings.json'
            fake = mock.Mock(pid=333)
            fake.poll.return_value = None
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', settings), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui.subprocess, 'Popen', return_value=fake) as popen:
                root = hidden_root()
                try:
                    app = image_ui.ImageApp(root)
                    self.assertEqual(app.hires_upscalers_dir.get(), '')
                    app.hires_upscalers_dir.set('IMAGE-MODELS/does-not-exist')
                    app.start()
                    popen.assert_called_once()          # the server really launched
                    cmd = popen.call_args.args[0]
                    self.assertNotIn('--hires-upscalers-dir', cmd)
                    log = (base / 'image-output' / 'server.log').read_text(encoding='utf-8')
                    self.assertIn('放大器目錄不存在', log)
                    self.assertIn('does-not-exist', log)
                    self.assertNotIn('啟動失敗', app.status.cget('text'))
                    app.stop()
                finally:
                    if 'app' in locals():
                        app.close()
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_prefix_cache_can_be_disabled_for_qwen_image_21(self):
        """Qwen-Image 2.1 plans a prefix-cache graph first and always drops it on 8 GB."""
        self.assertEqual(image_ui.PREFIX_CACHE_KEY, 'qwen_image_2_1_prefix_cache')
        self.assertNotIn('--model-args', image_ui.build_server_cmd(ROOT, 18436))
        cmd = image_ui.build_server_cmd(ROOT, 18436, prefix_cache_disabled=True)
        self.assertEqual(cmd[-2:], ['--model-args', 'qwen_image_2_1_prefix_cache=false'])
        # The backend parses a strict boolean, so anything else is refused before launch.
        for bad in ('true', 1, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                image_ui.build_server_cmd(ROOT, 18436, prefix_cache_disabled=bad)
        with self.assertRaises(ValueError):
            image_ui.build_server_cmd(ROOT, 18436, prefix_cache_disabled=True,
                                      extra_args='--model-args other=1')
        snapshot = image_ui.build_cmd_from_settings(
            ROOT, 18436, {'prefix_cache_disabled': True, 'max_vram_reserve_gib': '3'})
        self.assertEqual(snapshot[-4:], ['--max-vram', '-3', '--model-args',
                                         'qwen_image_2_1_prefix_cache=false'])

    def test_saved_snapshot_keeps_extra_args_across_a_mode_switch(self):
        """A Qwen-Edit switch relaunches from the saved snapshot, so extra args must survive."""
        settings = {'port': '18436', 'offload': True, 'runtime': 'custom/',
                    'models': {'llm_vision': 'IMAGE-MODELS/mmproj.gguf'}, 'cache_mode': 'spectrum',
                    'spectrum_w': '0.2', 'diffusion_fa': True, 'conditioning_cache_size': '8',
                    'extra_args': '--auto-fit off', 'max_vram_reserve_gib': '2',
                    'prefix_cache_disabled': True}
        cmd = image_ui.build_cmd_from_settings(ROOT, 18436, settings, vision=True, offload=True)
        for flag in ('--llm_vision', '--offload-to-cpu', '--cache-mode', '--diffusion-fa',
                     '--conditioning-cache-size'):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index('--max-vram') + 1], '-2')
        self.assertIn('qwen_image_2_1_prefix_cache=false', cmd)
        self.assertEqual(cmd[-2:], ['--auto-fit', 'off'])
        plain = image_ui.build_cmd_from_settings(ROOT, 18436, settings, offload=False)
        self.assertNotIn('--offload-to-cpu', plain)
        self.assertEqual(plain[-6:], ['--max-vram', '-2', '--model-args',
                                      'qwen_image_2_1_prefix_cache=false', '--auto-fit', 'off'])

    def test_widgets_forward_extra_args_reserve_number_and_prefix_cache_then_persist(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            settings = base / 'image-settings.json'
            fake = mock.Mock(pid=222)
            fake.poll.return_value = None
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', settings), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui.subprocess, 'Popen', return_value=fake) as popen:
                root = hidden_root()
                try:
                    app = image_ui.ImageApp(root)
                    self.assertEqual(app.extra_args.get(), '')
                    # Reserve number: editable, and 0 by default (flag not sent at all).
                    self.assertEqual(app.max_vram_reserve.get(), '0')
                    self.assertFalse(app.prefix_cache_disabled.get())
                    self.assertFalse(app.vae_on_cpu.get())
                    self.assertTrue(app.vae_tiling.get())   # 分塊解碼是預設：新安裝沒存檔也要開
                    app.offload.set(False)
                    app.extra_args.set('--auto-fit off')
                    app.max_vram_reserve.set('3')
                    app.prefix_cache_disabled.set(True)
                    app.vae_on_cpu.set(True)
                    app.vae_tiling.set(True)
                    app.start()
                    self.assertEqual(popen.call_args.args[0][-11:],
                                     ['--backend', 'vae=cpu', '--vae-tiling', '--ref-image-args',
                                      image_ui.REF_IMAGE_ARGS_DEFAULT, '--max-vram', '-3', '--model-args',
                                      'qwen_image_2_1_prefix_cache=false', '--auto-fit', 'off'])
                    saved = json.loads(settings.read_text(encoding='utf-8'))
                    self.assertEqual(saved['extra_args'], '--auto-fit off')
                    self.assertEqual(saved['max_vram_reserve_gib'], '3')
                    self.assertTrue(saved['prefix_cache_disabled'])
                    self.assertTrue(saved['vae_on_cpu'])
                    self.assertTrue(saved['vae_tiling'])
                    locked = (app.extra_args_entry, app.max_vram_box, app.prefix_cache_box,
                              app.vae_cpu_box, app.vae_tiling_box)
                    for widget in locked:
                        self.assertEqual(str(widget.cget('state')), 'disabled')
                    app.stop()
                    for widget in locked:
                        self.assertEqual(str(widget.cget('state')), 'normal')
                    app.max_vram_reserve.set('lots')  # Not a number: refuse and keep the saved settings
                    app.start()
                    self.assertEqual(popen.call_count, 1)
                    self.assertIn('保留顯存', app.status.cget('text'))
                    app.max_vram_reserve.set('3')
                    app.extra_args.set('--listen-port 18437')  # Managed flag: refuse before launch
                    app.start()
                    self.assertEqual(popen.call_count, 1)
                    self.assertIn('--listen-port', app.status.cget('text'))
                    self.assertEqual(json.loads(settings.read_text(encoding='utf-8'))['extra_args'],
                                     '--auto-fit off')
                finally:
                    if 'app' in locals():
                        app.close()
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_lora_resident_flag_tracks_only_the_current_server_session(self):
        with tempfile.TemporaryDirectory() as td:
            log = pathlib.Path(td) / 'server.log'
            self.assertFalse(image_ui.lora_resident(log))  # missing log: nothing resident
            log.write_text('[INFO   ] apply_loras completed, taking 0.04s\n', encoding='utf-8')
            self.assertTrue(image_ui.lora_resident(log))
            log.write_text('[INFO   ] apply_loras completed\n' + 'x' * 4096, encoding='utf-8')
            self.assertFalse(image_ui.lora_resident(log, limit=128))  # stale head must not count
            self.assertFalse(image_ui.lora_resident(log, since=32, limit=4096))  # before `since` never counts
            self.assertTrue(image_ui.lora_resident(log, limit=8192))  # ...but within the tail window it does
            log.write_text('x' * 4096 + '\n[INFO   ] apply_loras completed\n', encoding='utf-8')
            self.assertTrue(image_ui.lora_resident(log, limit=128))  # fresh tail does count

    def test_restart_request_relaunches_server_and_clears_resident_lora(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            settings = base / 'image-settings.json'
            exited = {'value': False}
            fake_proc = mock.Mock(pid=111, returncode=0)
            fake_proc.poll.side_effect = lambda: 0 if exited['value'] else None
            replacement = mock.Mock(pid=222)
            replacement.poll.return_value = None
            callbacks = []
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', settings), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui.subprocess, 'Popen', side_effect=[fake_proc, replacement]) as popen:
                root = hidden_root()
                try:
                    app = image_ui.ImageApp(root)
                    app.start()
                    argv = list(popen.call_args.args[0])
                    self.assertFalse(app.mode_status()['lora_resident'])
                    self.assertTrue(callable(app.save_service.request_restart))
                    log = base / 'image-output' / 'server.log'
                    with log.open('a', encoding='utf-8') as handle:
                        handle.write('[INFO   ] diffusion_engine.cpp:1808 - apply_loras completed, taking 0.04s\n')
                    self.assertTrue(app.mode_status()['lora_resident'])
                    saver = app.save_service
                    with mock.patch.object(root, 'after', side_effect=lambda _ms, fn: callbacks.append(fn)):
                        accepted, message = app.request_restart()
                        self.assertTrue(accepted)
                        self.assertIn('LoRA', message)
                        exited['value'] = True
                        callbacks.pop(0)()  # begin_switch: stop the old process
                        self.assertTrue(app.switching)
                        callbacks.pop(0)()  # finish_switch: relaunch with the same settings
                    self.assertFalse(app.switching)
                    self.assertEqual(popen.call_count, 2)
                    self.assertEqual(popen.call_args.args[0], argv)
                    self.assertIs(app.save_service, saver)
                    self.assertEqual(app.restart_note, '')
                    self.assertFalse(app.mode_status()['lora_resident'])  # restart released it
                    self.assertIn('釋放常駐 LoRA', log.read_text(encoding='utf-8'))
                    self.assertTrue(app.save_service.token)
                    app.stop()
                finally:
                    if 'app' in locals():
                        app.close()
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_restart_request_is_refused_when_no_server_is_running(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', base / 'image-settings.json'), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False):
                root = hidden_root()
                try:
                    app = image_ui.ImageApp(root)
                    accepted, message = app.request_restart()
                    self.assertFalse(accepted)
                    self.assertIn('未執行', message)
                    self.assertFalse(app.mode_status()['lora_resident'])
                finally:
                    if 'app' in locals():
                        app.close()
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_mode_endpoint_keeps_answering_after_the_server_restarts(self):
        """The page's release-LoRA button reads /mode with the port and token it was opened with."""
        import tkinter as tk
        import urllib.request
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', base / 'image-settings.json'), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui.subprocess, 'Popen') as popen:
                popen.return_value.poll.return_value = None  # the server is up in this session
                popen.return_value.pid = 1234
                root = hidden_root()
                try:
                    app = image_ui.ImageApp(root)
                    app.start()
                    port, token = app.save_service.httpd.server_port, app.save_service.token
                    app.stop()
                    app.start()  # control window 停止 → 啟動 must not invalidate the open page
                    with (base / 'image-output' / 'server.log').open('a', encoding='utf-8') as handle:
                        handle.write('[INFO   ] diffusion_engine.cpp:1808 - apply_loras completed, '
                                     'taking 0.04s\n')
                    request = urllib.request.Request(
                        f'http://127.0.0.1:{port}/mode?token={token}',
                        headers={'Origin': f'http://127.0.0.1:{app.port.get()}'})
                    with urllib.request.urlopen(request, timeout=5) as response:
                        payload = json.loads(response.read().decode('utf-8'))
                    self.assertTrue(payload['lora_resident'])
                finally:
                    if 'app' in locals():
                        app.close()
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_web_page_explains_why_the_release_button_is_disabled(self):
        """Disabled must be self-explanatory: no resident LoRA, a running queue, or lost access."""
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        for fragment in ('沒有套用過 LoRA', '按鈕要等本行程真的套用過 LoRA 才會亮',
                         '有任務在生成', '讀不到控制窗的存圖服務', '重新開啟本頁'):
            self.assertIn(fragment, html)

    def test_web_page_release_flow_uses_restart_endpoint_without_launch_flags(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        for fragment in ('id="release-lora"', 'id="lora-memory"', "saveBase+'/restart'",
                         'lora_resident', 'refreshLoraState'):
            self.assertIn(fragment, html)
        self.assertNotIn('--max-vram', html)  # VRAM reserve stays a control-window launch switch


class ViggleHiresAndFilenameTests(unittest.TestCase):
    """Hi-res 第二輪可沿用 turbo sigma；Viggle 產圖檔名帶 -6step／-turbo。"""

    def test_hires_can_carry_the_first_round_turbo_sigma(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        for fragment in ('id="hr-turbo-sigma"', '第二輪沿用第一輪 turbo sigma',
                         "body.hrTurboSigma=$('hr-turbo-sigma').checked&&$('viggle').value==='6step'",
                         'if(viggleTurbo&&body.hrTurboSigma)',
                         'native.hires.custom_sigmas=viggleSigmas(body.hr_resize_x,body.hr_resize_y,body.hr_steps)'):
            self.assertIn(fragment, html)
        # 第二輪 sigma 只在有開 Viggle 且使用者勾選時才送；沒勾就維持自行推導排程。
        self.assertIn('if(viggleTurbo&&body.hrTurboSigma)', html)
        # 沿用 sigma 只在 Viggle 開啟時可用。
        self.assertIn("$('hr-turbo-sigma').disabled=!$('enable-hr').checked||!viggleOn", html)

    def test_viggle_runs_are_tagged_in_the_saved_filename(self):
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        self.assertIn('viggle:Boolean(body.viggleTurbo)', html)
        # 別再把存圖 payload 寫成區塊外的裸變數：那正是先前存圖全數失敗的 ReferenceError 來源。
        self.assertNotIn('viggle:Boolean(viggleTurbo)', html)
        service = (ROOT / 'image_save_service.py').read_text(encoding='utf-8')
        self.assertIn("if data.get('viggle') is True:", service)
        self.assertIn("mode_label += '-6step' if steps == 6 else '-turbo'", service)


class FlashAttentionAndRefImageTests(unittest.TestCase):
    """--fa（全流程）與 --ref-image-args（參考圖參數）是兩個新的原生旗標。"""

    def test_fa_flag_defaults_on_and_sits_beside_diffusion_fa(self):
        """--fa 預設開啟、與 --diffusion-fa 並存；兩者互不取代。"""
        self.assertEqual(image_ui.FA_FLAG, '--fa')
        # 預設（沒指定）＝ 送出 --fa；--diffusion-fa 仍需自行勾選。
        plain = image_ui.build_server_cmd(ROOT, 18436)
        self.assertIn('--fa', plain)
        self.assertNotIn('--diffusion-fa', plain)
        # 兩者同時送。
        both = image_ui.build_server_cmd(ROOT, 18436, fa=True, diffusion_fa=True)
        self.assertIn('--fa', both)
        self.assertIn('--diffusion-fa', both)
        # 明確關掉 --fa，但 --diffusion-fa 仍在。
        self.assertNotIn('--fa', image_ui.build_server_cmd(ROOT, 18436, fa=False))
        self.assertNotIn('--fa', image_ui.build_server_cmd(ROOT, 18436, fa=False, diffusion_fa=True))
        for bad in ('yes', 1, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                image_ui.build_server_cmd(ROOT, 18436, fa=bad)

    def test_fa_survives_the_settings_snapshot_and_migrates_from_extra_args(self):
        # 舊存檔沒有 fa 欄位 → 預設開啟。
        self.assertIn('--fa', image_ui.build_cmd_from_settings(ROOT, 18436, {}))
        # 存檔明確關掉 → 不送。
        self.assertNotIn('--fa', image_ui.build_cmd_from_settings(ROOT, 18436, {'fa': False}))
        self.assertIn('--fa', image_ui.build_cmd_from_settings(ROOT, 18436, {'fa': True}))
        # 手寫在額外指令的 --fa 會被搬進欄位，剩下的照舊。
        self.assertEqual(image_ui.migrate_fa_flag('--fa --auto-fit off'), ('--auto-fit off', True))
        self.assertEqual(image_ui.migrate_fa_flag('--auto-fit off'), ('--auto-fit off', False))
        self.assertEqual(image_ui.migrate_fa_flag(None), ('', False))

    def test_ref_image_args_defaults_and_reaches_the_launch_command(self):
        self.assertEqual(image_ui.REF_IMAGE_ARGS_FLAG, '--ref-image-args')
        self.assertEqual(image_ui.REF_IMAGE_ARGS_DEFAULT, 'preset=qwen,vae_input_max_pixels=800000')
        # 沒填 → 不送；填了 → 原樣送成一組旗標＋值。
        self.assertNotIn('--ref-image-args', image_ui.build_server_cmd(ROOT, 18436))
        cmd = image_ui.build_server_cmd(ROOT, 18436, ref_image_args=image_ui.REF_IMAGE_ARGS_DEFAULT)
        self.assertEqual(cmd[cmd.index('--ref-image-args') + 1], image_ui.REF_IMAGE_ARGS_DEFAULT)
        # 舊存檔沒有這個欄位時沿用實測預設，切換模式重啟不會把它弄丟。
        self.assertEqual(
            image_ui.build_cmd_from_settings(ROOT, 18436, {}).count('--ref-image-args'), 1)
        self.assertNotIn('--ref-image-args',
                         image_ui.build_cmd_from_settings(ROOT, 18436, {'ref_image_args': ''}))
        # 格式驗證：合法 key=value 清單、空字串都收；旗標名或沒有 = 的拒絕。
        self.assertTrue(image_ui.valid_ref_image_args(image_ui.REF_IMAGE_ARGS_DEFAULT))
        self.assertTrue(image_ui.valid_ref_image_args(''))
        self.assertTrue(image_ui.valid_ref_image_args('a=1, b=2'))
        self.assertFalse(image_ui.valid_ref_image_args(None))
        for bad in ('--fa', 'preset', 'a=1,,b=2', 1):
            with self.subTest(bad=bad):
                self.assertFalse(image_ui.valid_ref_image_args(bad))
                with self.assertRaises(ValueError):
                    image_ui.build_server_cmd(ROOT, 18436, ref_image_args=bad)
        # 兩處都填會被擋下，避免同一旗標送兩次。
        with self.assertRaises(ValueError):
            image_ui.build_server_cmd(ROOT, 18436, ref_image_args='a=1',
                                      extra_args='--ref-image-args b=2')

    def test_hand_written_ref_image_args_moves_into_the_field(self):
        migrate = image_ui.migrate_ref_image_args
        self.assertEqual(migrate('--ref-image-args "preset=qwen,vae_input_max_pixels=800000"'),
                         ('', 'preset=qwen,vae_input_max_pixels=800000'))
        self.assertEqual(migrate('--ref-image-args=preset=qwen --auto-fit off'),
                         ('--auto-fit off', 'preset=qwen'))
        self.assertEqual(migrate('--fa --ref-image-args "preset=qwen"'),
                         ('--fa', 'preset=qwen'))
        self.assertEqual(migrate('--ref-image-args --fa'), ('--fa', ''))  # bare flag keeps next switch
        self.assertEqual(migrate('--auto-fit off'), ('--auto-fit off', ''))
        self.assertEqual(migrate(None), ('', ''))

    def test_new_widgets_exist_lock_while_running_and_restore_the_default(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            settings = base / 'image-settings.json'
            fake = mock.Mock(pid=222)
            fake.poll.return_value = None
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', settings), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui.subprocess, 'Popen', return_value=fake) as popen:
                root = hidden_root()
                try:
                    app = image_ui.ImageApp(root)
                    self.assertTrue(app.fa.get())  # 預設開啟
                    self.assertEqual(app.ref_image_args.get(), image_ui.REF_IMAGE_ARGS_DEFAULT)
                    self.assertIn('預設值', app.ref_image_hint.cget('text'))
                    app.start()
                    cmd = popen.call_args.args[0]
                    self.assertIn('--fa', cmd)
                    self.assertEqual(cmd[cmd.index('--ref-image-args') + 1], image_ui.REF_IMAGE_ARGS_DEFAULT)
                    for widget in (app.fa_all_box, app.ref_image_entry, app.ref_image_reset):
                        self.assertEqual(str(widget.cget('state')), 'disabled')
                    saved = json.loads(settings.read_text(encoding='utf-8'))
                    self.assertIs(saved['fa'], True)
                    self.assertEqual(saved['ref_image_args'], image_ui.REF_IMAGE_ARGS_DEFAULT)
                    app.stop()
                    for widget in (app.fa_all_box, app.ref_image_entry, app.ref_image_reset):
                        self.assertEqual(str(widget.cget('state')), 'normal')
                    # 「還原預設」把欄位拉回預設值。
                    app.ref_image_args.set('garbage')
                    app.ref_image_reset.invoke()
                    self.assertEqual(app.ref_image_args.get(), image_ui.REF_IMAGE_ARGS_DEFAULT)
                    self.assertIn('預設值', app.ref_image_hint.cget('text'))
                    # 壞掉的內容：提示轉紅，且啟動被擋下。
                    app.ref_image_args.set('not-key-value')
                    self.assertIn('key=value', app.ref_image_hint.cget('text'))
                    app.start()
                    self.assertEqual(popen.call_count, 1)
                    self.assertIn('--ref-image-args', app.status.cget('text'))
                finally:
                    if 'app' in locals():
                        app.close()
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_hint_popup_text_is_selectable(self):
        """說明彈窗必須能反白選取複製；ttk.Label 做不到，所以用唯讀 Text。"""
        import tkinter as tk
        root = hidden_root()
        opener = tk.Frame(root)
        opener.pack()

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        image_ui._popup_hint(opener, '測試說明\n第二行')
        popups = [w for w in descendants(opener) if isinstance(w, tk.Toplevel)]
        self.assertEqual(len(popups), 1)
        texts = [w for w in descendants(popups[0]) if isinstance(w, tk.Text)]
        self.assertEqual(len(texts), 1)
        box = texts[0]
        self.assertEqual(box.cget('state'), 'disabled')
        self.assertIn('測試說明', box.get('1.0', 'end'))
        for button in [w for w in descendants(popups[0]) if isinstance(w, tk.ttk.Button)]:
            self.assertIn(button.cget('text'), ('複製全部', '關閉'))
        popups[0].destroy()


class ControlWindowTests(unittest.TestCase):
    """The action bar and the previous run's log must survive a window that is too small."""

    def test_action_buttons_stay_above_the_log_and_the_form_scrolls(self):
        """The config form used to be clipped by the paned window, hiding 「▶ 啟動 Image Server」."""
        import tkinter as tk

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', base / 'image-settings.json'), \
                 mock.patch.object(image_ui, 'port_busy', return_value=True):
                root = hidden_root()
                try:
                    app = image_ui.ImageApp(root)
                    root.geometry('880x620')  # too short for the whole form
                    root.update()
                    canvases = [wat for wat in descendants(root) if isinstance(wat, tk.Canvas)]
                    self.assertTrue(canvases)
                    form = canvases[0]
                    box = form.bbox('all')
                    self.assertGreater(box[3] - box[1], form.winfo_height())  # scrolls, no clipping
                    for widget in (app.start_btn, app.stop_btn, app.status):
                        self.assertTrue(widget.winfo_ismapped(), widget)
                    ancestor = app.start_btn
                    while ancestor is not root:
                        self.assertNotIsInstance(ancestor, tk.Canvas)  # never scrolls out of sight
                        ancestor = ancestor.master
                    # Above the log pane, not below it.
                    self.assertLess(app.start_btn.winfo_rooty(), app.log.winfo_rooty())
                finally:
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass

    def test_opening_the_window_keeps_the_previous_run_as_a_backup(self):
        """A fresh window truncates server.log, so the last failure must be copied aside first."""
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            (base / 'image-output').mkdir()
            log = base / 'image-output' / 'server.log'
            log.write_text('[ERROR] segment 1/1 (graph) failed during workspace capacity check\n',
                           encoding='utf-8')
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', base / 'image-settings.json'), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False):
                root = hidden_root()
                try:
                    image_ui.ImageApp(root)
                    self.assertEqual(log.read_text(encoding='utf-8'), '')
                    self.assertIn('workspace capacity check',
                                  (base / 'image-output' / 'server.log.prev').read_text(encoding='utf-8'))
                finally:
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass


class ModelCandidateTests(unittest.TestCase):
    """The dropdown lists every file, but each field marks the files that cannot fill it.

    Picking a LoRA into VAE (or an upscaler) kills sd-server at startup with
    'model metadata validation failed', so the UI must warn before the user does it.
    """

    def _models(self, td):
        base = pathlib.Path(td)
        for name in ('qwen_image_2.1_vae_bf16.safetensors',
                     'qwen-image-2.1-Q4_K_M.gguf',
                     'Qwen3VL-8B-Instruct-Q4_K_M.gguf',
                     'mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf',
                     'loras/NSFW Qwen Lora.safetensors',
                     'loras/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128-fused-gguf.safetensors',
                     'upscalers/RealESRGAN_x4plus_anime_6B.pth'):
            p = base / 'IMAGE-MODELS' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'x')
        return base

    def test_vae_field_accepts_only_vae_named_files(self):
        with tempfile.TemporaryDirectory() as td:
            base = self._models(td)
            ok = os.path.join('IMAGE-MODELS', 'qwen_image_2.1_vae_bf16.safetensors')
            self.assertTrue(image_ui.model_fits_field(base, 'vae', ok))
            for bad in (os.path.join('IMAGE-MODELS', 'loras', 'NSFW Qwen Lora.safetensors'),
                        os.path.join('IMAGE-MODELS', 'upscalers', 'RealESRGAN_x4plus_anime_6B.pth'),
                        os.path.join('IMAGE-MODELS', 'qwen-image-2.1-Q4_K_M.gguf'),
                        os.path.join('IMAGE-MODELS', 'mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf')):
                self.assertFalse(image_ui.model_fits_field(base, 'vae', bad), bad)

    def test_loras_are_only_flagged_where_they_cannot_go(self):
        lora = os.path.join('IMAGE-MODELS', 'loras', 'NSFW Qwen Lora.safetensors')
        with tempfile.TemporaryDirectory() as td:
            base = self._models(td)
            self.assertFalse(image_ui.model_fits_field(base, 'clip_l', lora))
            self.assertFalse(image_ui.model_fits_field(base, 'clip_g', lora))
            self.assertFalse(image_ui.model_fits_field(base, 't5xxl', lora))
            # a LoRA placed in the diffusion slot is a legitimate (if unusual) choice
            self.assertTrue(image_ui.model_fits_field(base, 'diffusion', lora))

    def test_candidate_labels_are_paths_plus_a_mark_for_mismatches(self):
        with tempfile.TemporaryDirectory() as td:
            base = self._models(td)
            labels = image_ui.model_fieldui_labels(base, 'vae')
            good = str(base / 'IMAGE-MODELS' / 'qwen_image_2.1_vae_bf16.safetensors')
            self.assertIn(good, labels)              # the only usable VAE, unmarked
            self.assertNotIn(good + image_ui.NOT_FOR_FIELD_MARK, labels)
            marked = [l for l in labels if l.endswith(image_ui.NOT_FOR_FIELD_MARK)]
            self.assertTrue(marked, 'mis-candidates must be visibly marked')
            for l in marked:                          # nothing marked may be the real VAE
                self.assertNotIn('qwen_image_2.1_vae_bf16', l)
            self.assertEqual(len(marked), len(labels) - 1)   # every other file is marked

    def test_selected_value_is_never_dropped_from_the_list(self):
        """tk rewrites an unmatched combobox value to '', which would silently unset a field."""
        with tempfile.TemporaryDirectory() as td:
            base = self._models(td)
            outside = base / 'external' / 'my-custom-vae.safetensors'
            outside.parent.mkdir(parents=True, exist_ok=True)
            outside.write_bytes(b'x')
            labels = image_ui.model_fieldui_labels(base, 'vae', extra=[str(outside)])
            self.assertIn(str(outside), labels)


    def test_hires_dir_flag_pasted_into_the_field_falls_back_to_the_conventional_folder(self):
        """The field must hold a directory, never the flag name.

        A pasted '--hires-upscalers-dir' used to be kept verbatim, so every launch refused to
        start over a folder literally named '<base>/--hires-upscalers-dir' — and the window
        saved it straight back, so it never healed.
        """
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            (base / 'IMAGE-MODELS' / 'upscalers').mkdir(parents=True)
            self.assertEqual(image_ui.sanitise_hires_dir('--hires-upscalers-dir', '', base),
                             'IMAGE-MODELS/upscalers')
            self.assertEqual(image_ui.sanitise_hires_dir('  --hires-upscalers-dir  ', '', base),
                             'IMAGE-MODELS/upscalers')

    def test_hires_dir_falls_back_to_empty_when_no_conventional_folder_exists(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(image_ui.sanitise_hires_dir('--hires-upscalers-dir', '', pathlib.Path(td)), '')

    def test_a_real_path_is_never_silently_swapped(self):
        """A typo must still fail loudly at launch (documented), not be replaced by the default."""
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            (base / 'IMAGE-MODELS' / 'upscalers').mkdir(parents=True)
            self.assertEqual(image_ui.sanitise_hires_dir('D:/my/esrgan', '', base), 'D:/my/esrgan')
            # a hand-written flag in 「額外指令」 still wins, as before
            self.assertEqual(image_ui.sanitise_hires_dir('', 'IMAGE-MODELS/upscalers', base),
                             'IMAGE-MODELS/upscalers')
            self.assertEqual(image_ui.sanitise_hires_dir(None, '', base), 'IMAGE-MODELS/upscalers')

    def test_flag_like_value_is_dropped_and_reported_by_name(self):
        """A pasted flag name must not become a joined path, and must not block the launch."""
        with tempfile.TemporaryDirectory() as td:
            cmd = image_ui.build_server_cmd(pathlib.Path(td), 18436,
                                            hires_upscalers_dir='--hires-upscalers-dir')
            self.assertNotIn('--hires-upscalers-dir', cmd)
            message = image_ui.hires_dir_warning('--hires-upscalers-dir', pathlib.Path(td))
            self.assertIn('--hires-upscalers-dir', message)
            self.assertIn('旗標', message)
            self.assertNotIn(str(pathlib.Path(td) / '--hires-upscalers-dir'), message)


    def test_dropdown_marker_never_reaches_a_setting_path(self):
        """The  ✗ 不適用 suffix is display-only; it must not become part of a model path.

        Selecting a marked dropdown entry wrote '...safetensors  ✗ 不適用' into the field, so
        sd-server reported 模型格式不符 for a name that looked correct.
        """
        import tkinter as tk
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            (base / 'IMAGE-MODELS' / 'loras').mkdir(parents=True)
            bad = base / 'IMAGE-MODELS' / 'loras' / 'NSFW Qwen Lora.safetensors'
            bad.write_bytes(b'x')
            settings = base / 'image-settings.json'
            with mock.patch.object(image_ui, 'BASE', base), \
                 mock.patch.object(image_ui, 'SETTINGS', settings), \
                 mock.patch.object(image_ui, 'port_busy', return_value=False), \
                 mock.patch.object(image_ui, 'model_fits_field', return_value=False), \
                 mock.patch.object(image_ui.messagebox, 'showwarning') as warn:
                root = hidden_root()
                try:
                    app = image_ui.ImageApp(root)
                    app.models['vae'].set('NSFW Qwen Lora.safetensors' + image_ui.NOT_FOR_FIELD_MARK)
                    app.check_model_choice('vae')
                    self.assertEqual(app.models['vae'].get(), 'NSFW Qwen Lora.safetensors')
                    warn.assert_called_once()
                finally:
                    app.close()
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass
        # a settings snapshot saved by the buggy build is cleaned on its way into the command
        cmd = image_ui.build_server_cmd(ROOT, 18436,
                                        models={'diffusion': 'IMAGE-MODELS/x.gguf' + image_ui.NOT_FOR_FIELD_MARK})
        self.assertTrue(any(part.endswith('IMAGE-MODELS' + os.sep + 'x.gguf') for part in cmd), cmd)
        self.assertFalse(any(image_ui.NOT_FOR_FIELD_MARK in part for part in cmd))

    def test_hi_res_does_not_refuse_large_outputs_up_front(self):
        """The 32-multiple rule is the only client-side Hi-res shape check left.

        A pixel ceiling used to block the default 512x1024 @2x (=2,097,152 px) before the
        request was ever sent, so the feature looked broken.  If the card cannot take it,
        sd-server fails the job and the task shows the error — that is enough.
        """
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        self.assertNotIn('2_000_000', html)
        self.assertNotIn('200 萬', html)

    def test_viggle_six_step_is_no_longer_blocked_from_img2img_or_hires(self):
        """The block was a stub, not a limit: the server runs the combination fine.

        Live runs confirmed the native path applies the 6-step sigmas to stage 1 and the Hi-res
        block to stage 2.  The page must not refuse it up front — stage 2 just derives its own
        schedule from 第二輪步數+去噪強度, which the help text now says out loud.
        """
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        self.assertNotIn('Viggle 不支援', html)
        self.assertNotIn('不支援去噪 Img2Img 或 Hi-res', html)
        self.assertNotIn('不支援 Hi-res 或一般去噪 Img2Img', html)
        # the native branch still forwards both blocks for the turbo path
        self.assertIn('if(viggleTurbo||(mode===\'img2img\'&&body.hrNative))', html)
        self.assertIn('sample_params.custom_sigmas=viggleSigmas', html)
        self.assertIn('native.hires={enabled:true', html)
        # and the help text now says stage 2 can reuse the first-stage sigmas when opted in
        self.assertIn('第二輪沿用第一輪 turbo sigma', html)
        self.assertNotIn('不會沿用第一輪 turbo sigma', html)

    def test_hi_res_scale_offers_a_same_size_enhance_option(self):
        """1x runs the model upscaler and resamples back, sharpening without enlarging.

        Live: 128x128 with RealESRGAN and target 128x128 loaded the .pth, upscaled internally
        and returned 128x128.  The dropdown must expose it.
        """
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        m = re.search(r'<select id="hr-scale"[^>]*>(.*?)</select>', html, re.S)
        self.assertIsNotNone(m, '找不到 Hi-res 倍率選單')
        self.assertIn('value="1"', m.group(1))
        self.assertIn('同尺寸升畫質', m.group(1))

    def test_denoising_strength_allows_zero_for_pure_hires(self):
        """strength 0 skips stage 1 entirely ('target t_enc is 0 steps') and only upscales.

        Verified live: with strength=0 the log shows 'target t_enc is 0 steps' and then
        'hires fix: upscaling to ...'.  A min of 0.01 made that unreachable from the page.
        """
        html = (ROOT / 'assets/image-web.html').read_text(encoding='utf-8')
        m = re.search(r'<input id="denoising-strength"[^>]*>', html)
        self.assertIsNotNone(m, '找不到改動強度欄位')
        self.assertIn('min="0"', m.group(0))
        self.assertNotIn('min="0.01"', m.group(0))


if __name__ == '__main__':
    unittest.main()
