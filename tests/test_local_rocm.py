import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
SPEC = importlib.util.spec_from_file_location('image_ui_rocm_test', BASE / 'image-ui.py')
ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ui)


class LocalRocmTests(unittest.TestCase):
    names = ('amdhip64_7.dll', 'rocm_kpack.dll', 'amd_comgr.dll',
             'hiprtc0714.dll', 'hiprtc-builtins0714.dll')

    def fixture(self, base):
        rt = base / 'RUNTIMES/stable-diffusion-rocm-master-929-3f8527a'
        core = base / ui.ROCM_PACKAGES / 'packages/_rocm_sdk_core/bin'
        rt.mkdir(parents=True)
        core.mkdir(parents=True)
        for name in self.names:
            (rt / name).write_bytes(name.encode())
            (core / name).write_bytes(name.encode())
        return rt, core

    def test_other_backends_are_unchanged(self):
        self.assertEqual(ui.local_rocm_launch_note(BASE / 'RUNTIMES/example-vulkan/sd-server.exe'), '')
        self.assertEqual(ui.local_rocm_launch_note(BASE / 'RUNTIMES/example-cuda/sd-server.exe'), '')

    def test_matching_bundle_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            rt, _ = self.fixture(base)
            with mock.patch.object(ui, 'BASE', base):
                note = ui.local_rocm_launch_note(rt / 'sd-server.exe')
            self.assertIn(str(rt.resolve() / 'amdhip64_7.dll'), note)
            self.assertIn('7.14', note)

    def test_missing_bundle_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            rt, _ = self.fixture(base)
            (rt / 'rocm_kpack.dll').unlink()
            with mock.patch.object(ui, 'BASE', base), self.assertRaisesRegex(RuntimeError, 'rocm_kpack'):
                ui.local_rocm_launch_note(rt / 'sd-server.exe')

    def test_different_bundle_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            rt, _ = self.fixture(base)
            (rt / 'amdhip64_7.dll').write_bytes(b'other runtime')
            with mock.patch.object(ui, 'BASE', base), self.assertRaisesRegex(RuntimeError, 'amdhip64_7'):
                ui.local_rocm_launch_note(rt / 'sd-server.exe')

    def test_start_and_mode_switch_share_validation(self):
        import inspect
        for method in (ui.ImageApp.start, ui.ImageApp.finish_switch):
            self.assertIn('local_rocm_launch_note(cmd[0])', inspect.getsource(method))

    def test_fixed_tiles_and_reserve_are_forwarded(self):
        config = {'offload': True, 'max_vram_reserve_gib': '2',
                  'vae_tiling': True, 'vae_relative_tile_size': '768x768'}
        cmd = ui.build_cmd_from_settings(BASE, 18436, config)
        self.assertIn('--offload-to-cpu', cmd)
        self.assertEqual(cmd[cmd.index('--max-vram') + 1], '-2')
        self.assertEqual(cmd[cmd.index('--vae-tile-size') + 1], '768x768')
        self.assertNotIn('--vae-relative-tile-size', cmd)

    def test_ratio_keeps_relative_flag(self):
        cmd = ui.build_cmd_from_settings(BASE, 18436,
                                        {'vae_tiling': True, 'vae_relative_tile_size': '0.65'})
        self.assertEqual(cmd[cmd.index('--vae-relative-tile-size') + 1], '0.65')
        self.assertNotIn('--vae-tile-size', cmd)


if __name__ == '__main__':
    unittest.main()
