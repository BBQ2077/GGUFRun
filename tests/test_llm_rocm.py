import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
TMP = None   # 暫存 fixture 放系統暫存資料夾
sys.path.insert(0, str(BASE))
import llm_rocm


class LlmRocmTests(unittest.TestCase):
    def test_managed_environment_is_child_local_and_excludes_legacy_sdk(self):
        repair = BASE / llm_rocm.LEGACY_SDK_RELATIVE
        original = {'PATH': str(repair / 'packages/_rocm_sdk_core/bin') + os.pathsep + 'C:/Windows/System32',
                    'HIP_PATH': str(repair / 'packages/_rocm_sdk_core')}
        with mock.patch.dict(os.environ, original, clear=True):
            result = llm_rocm.llama_environment(BASE / 'RUNTIMES' / llm_rocm.MANAGED_RUNTIME / 'llama-server.exe', BASE)
            self.assertEqual(dict(os.environ), original)
        self.assertIn('rocm-sdk-10.1.0', result['PATH'])
        self.assertNotIn(str(repair / 'packages/_rocm_sdk_core/bin'), result['PATH'])
        self.assertNotIn('HIP_PATH', result)

    def test_other_runtimes_keep_environment(self):
        with mock.patch.dict(os.environ, {'PATH': 'original'}, clear=True):
            self.assertEqual(llm_rocm.llama_environment(BASE / 'RUNTIMES/vulkan/llama-server.exe', BASE), dict(os.environ))

    def test_explicit_cpu_options_are_supported(self):
        for args in (['-ngl', '0'], ['--gpu-layers=0'], ['--device', 'none']):
            self.assertTrue(llm_rocm.cpu_requested(['llama-server.exe'] + args))
        self.assertFalse(llm_rocm.cpu_requested(['llama-server.exe', '-ngl', 'all']))

    def test_cpu_runtime_does_not_run_a_probe(self):
        with mock.patch.object(llm_rocm.subprocess, 'run') as run:
            _, note = llm_rocm.prepare_llama_launch([str(BASE / 'RUNTIMES/not-installed/llama-server.exe')], BASE)
        run.assert_not_called()
        self.assertEqual(note, '')

    def probe_fixture(self, output, raises=False):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            runtime = Path(folder)
            (runtime / 'ggml-hip.dll').write_bytes(b'fixture')
            cmd = [str(runtime / 'llama-server.exe')]
            response = subprocess.CompletedProcess(cmd, 0, output, '')
            with mock.patch.object(llm_rocm.subprocess, 'run', return_value=response) as run:
                if raises:
                    with self.assertRaisesRegex(RuntimeError, 'ROCm GPU'):
                        llm_rocm.prepare_llama_launch(cmd, BASE)
                else:
                    _, note = llm_rocm.prepare_llama_launch(cmd, BASE)
                    self.assertIn('RX 9070', note)
                self.assertEqual(run.call_args.args[0], [cmd[0], '--list-devices'])

    def test_empty_device_list_blocks_silent_cpu_fallback(self):
        self.probe_fixture('Available devices:\n (none)', raises=True)

    def test_rocm_device_is_reported(self):
        self.probe_fixture('Available devices:\n ROCm0: AMD Radeon RX 9070 (16304 MiB)')

    def make_managed_fixture(self, base):
        runtime = base / 'RUNTIMES' / llm_rocm.MANAGED_RUNTIME
        sdk = base / llm_rocm.SDK_RELATIVE
        inventories = []
        for root, names in ((runtime, llm_rocm.CORE_NAMES), (sdk, llm_rocm.SDK_NAMES)):
            files = {}
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
                files[path.relative_to(base).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            inventories.append(files)
        manifest = {'sdk_version': '10.1.0', 'runtime': runtime.relative_to(base).as_posix(),
                    'sdk_manifest': (sdk / 'sdk-manifest.json').relative_to(base).as_posix(), 'files': inventories[0]}
        sdk_manifest = {'sdk_version': '10.1.0', 'gpu_arch': 'gfx1201',
                        'library_bin': (sdk / 'bin').relative_to(base).as_posix(), 'files': inventories[1]}
        (runtime / 'rocm-runtime.json').write_text(json.dumps(manifest), encoding='utf-8')
        (sdk / 'sdk-manifest.json').write_text(json.dumps(sdk_manifest), encoding='utf-8')
        return runtime, sdk

    def test_manifest_detects_changes(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            base = Path(folder)
            runtime, _ = self.make_managed_fixture(base)
            llm_rocm.verify_managed_runtime(runtime / 'llama-server.exe', base)
            (runtime / 'amdhip64_7.dll').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                llm_rocm.verify_managed_runtime(runtime / 'llama-server.exe', base)

    def test_managed_runtime_works_without_artifact_directories(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            base = Path(folder)
            runtime, _ = self.make_managed_fixture(base)
            llm_rocm.verify_managed_runtime(runtime / 'llama-server.exe', base)
            self.assertFalse((base / llm_rocm.LEGACY_SDK_RELATIVE).exists())

    def test_missing_manifest_reports_permanent_path(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            base = Path(folder)
            runtime, _ = self.make_managed_fixture(base)
            (runtime / 'rocm-runtime.json').unlink()
            with self.assertRaisesRegex(RuntimeError, 'rocm-runtime.json') as caught:
                llm_rocm.verify_managed_runtime(runtime / 'llama-server.exe', base)
            self.assertNotIn('Errno 2', str(caught.exception))

    def test_missing_sdk_library_is_not_ignored(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            base = Path(folder)
            runtime, sdk = self.make_managed_fixture(base)
            (sdk / 'bin/hipblas.dll').unlink()
            with self.assertRaisesRegex(RuntimeError, 'hipblas.dll'):
                llm_rocm.verify_managed_runtime(runtime / 'llama-server.exe', base)

    def test_missing_hip_backend_blocks_cpu_fallback(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            base = Path(folder)
            runtime, _ = self.make_managed_fixture(base)
            (runtime / 'ggml-hip.dll').unlink()
            with mock.patch.object(llm_rocm.subprocess, 'run') as probe:
                with self.assertRaisesRegex(RuntimeError, 'ggml-hip.dll'):
                    llm_rocm.prepare_llama_launch([str(runtime / 'llama-server.exe')], base)
            probe.assert_not_called()

    def test_missing_required_inventory_entry_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            base = Path(folder)
            runtime, sdk = self.make_managed_fixture(base)
            path = sdk / 'sdk-manifest.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            del data['files'][(sdk / 'bin/rocsolver.dll').relative_to(base).as_posix()]
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, '必要元件'):
                llm_rocm.verify_managed_runtime(runtime / 'llama-server.exe', base)

    def test_manifest_cannot_point_outside_runtime(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            base = Path(folder)
            runtime, sdk = self.make_managed_fixture(base)
            path = runtime / 'rocm-runtime.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            data['files'][(sdk / 'bin/hipblas.dll').relative_to(base).as_posix()] = hashlib.sha256(b'fixture').hexdigest()
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, '元件未齊備'):
                llm_rocm.verify_managed_runtime(runtime / 'llama-server.exe', base)

    def test_gui_passes_the_checked_environment_to_server(self):
        source = (BASE / 'gguf-ui.py').read_text(encoding='utf-8')
        self.assertIn('launch_env, gpu_note = prepare_llama_launch(cmd, BASE)', source)
        self.assertIn('cwd=self.server_dir(), env=launch_env', source)


if __name__ == '__main__':
    unittest.main()
