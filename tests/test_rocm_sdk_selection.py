import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
TMP = None   # 暫存 fixture 放系統暫存資料夾
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location('image_ui_sdk_test', BASE / 'image-ui.py')
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


class RocmSdkSelectionTests(unittest.TestCase):
    def test_new_runtime_uses_only_new_project_sdk(self):
        exe = BASE / 'RUNTIMES' / ui.ROCM_10_RUNTIME / 'sd-server.exe'
        repair = BASE / ui.ROCM_PACKAGES
        old = str(repair / 'packages/_rocm_sdk_core/bin')
        new = str(BASE / ui.ROCM_10_SDK / 'bin')
        original = {'PATH': os.pathsep.join((old, 'C:/Windows/System32')),
                    'ROCBLAS_TENSILE_LIBPATH': str(repair / 'packages/old-data')}
        with mock.patch.dict(os.environ, original, clear=True):
            env = ui.local_rocm_environment(exe)
            self.assertEqual(dict(os.environ), original)
        self.assertIn(new, env['PATH'].split(os.pathsep))
        self.assertNotIn(old, env['PATH'].split(os.pathsep))
        self.assertNotIn('ROCBLAS_TENSILE_LIBPATH', env)

    def test_legacy_runtime_keeps_its_sdk(self):
        exe = BASE / 'RUNTIMES/stable-diffusion-rocm-master-929-3f8527a/sd-server.exe'
        repair = BASE / ui.ROCM_PACKAGES
        with mock.patch.dict(os.environ, {'PATH': str(repair / 'packages-10.1.0/_rocm_sdk_libraries/bin')}, clear=True):
            env = ui.local_rocm_environment(exe)
        self.assertIn(str(repair / 'packages/_rocm_sdk_libraries/bin'), env['PATH'])
        self.assertNotIn('packages-10.1.0', env['PATH'])

    def test_other_backend_preserves_environment(self):
        with mock.patch.dict(os.environ, {'PATH': 'original'}, clear=True):
            self.assertEqual(ui.local_rocm_environment(BASE / 'RUNTIMES/vulkan/sd-server.exe'), dict(os.environ))

    def test_manifest_validates_hashes(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            base = Path(folder)
            sdk = base / ui.ROCM_10_SDK
            library = sdk / 'bin/hipblas.dll'
            core = base / 'RUNTIMES' / ui.ROCM_10_RUNTIME / 'amdhip64_7.dll'
            for file, manifest_name in ((core, 'rocm-runtime.json'), (library, 'sdk-manifest.json')):
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b'test fixture')
                relative = file.relative_to(base).as_posix()
                manifest = {'sdk_version': '10.1.0',
                            'files': {relative: hashlib.sha256(file.read_bytes()).hexdigest()}}
                root = core.parent if file == core else sdk
                (root / manifest_name).write_text(json.dumps(manifest), encoding='utf-8')
            with mock.patch.object(ui, 'BASE', base):
                self.assertIn('10.1.0', ui.local_rocm_launch_note(core.parent / 'sd-server.exe'))
                for file in (core, library):
                    file.write_bytes(b'changed fixture')
                    with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                        ui.local_rocm_launch_note(core.parent / 'sd-server.exe')
                    file.write_bytes(b'test fixture')

    def test_both_launch_paths_use_sdk_environment(self):
        for method in (ui.ImageApp.start, ui.ImageApp.finish_switch):
            self.assertIn('env=local_rocm_environment(cmd[0])', inspect.getsource(method))


if __name__ == '__main__':
    unittest.main()
