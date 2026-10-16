"""Project-local ROCm environment and GPU preflight for llama.cpp."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess

MANAGED_RUNTIME = 'llama-b11642-rocm-sdk-10.1.0-x64'
SDK_RELATIVE = Path('RUNTIMES/rocm-sdk-10.1.0')
# 舊版專案 SDK 位置：子程序 PATH 只留新版 SDK，把這裡的 packages 排除
LEGACY_SDK_RELATIVE = Path('RUNTIMES/rocm-sdk-packages')
CORE_NAMES = ('llama-server.exe', 'ggml-hip.dll', 'amdhip64_7.dll', 'amd_comgr.dll', 'rocm_kpack.dll')
SDK_NAMES = ('bin/hipblas.dll', 'bin/rocblas.dll', 'bin/libhipblaslt.dll',
             'bin/rocsolver.dll', 'bin/origami.dll',
             '.kpack/blas_lib_gfx1201.kpack', '.kpack/solver_lib_gfx1201.kpack')


def llama_environment(executable, base):
    env = os.environ.copy()
    runtime = Path(executable).resolve().parent
    if runtime.name != MANAGED_RUNTIME:
        return env
    base = Path(base).resolve()
    libraries = base / SDK_RELATIVE / 'bin'
    old_sdk_root = str(base / LEGACY_SDK_RELATIVE).replace(chr(92), '/').casefold() + '/packages'
    new_sdk_root = str(base / SDK_RELATIVE).replace(chr(92), '/').casefold()

    def project_sdk(value):
        normalized = value.strip('"').replace(chr(92), '/').casefold()
        return normalized.startswith(old_sdk_root) or normalized == new_sdk_root or normalized.startswith(new_sdk_root + '/')
    paths = [entry for entry in env.get('PATH', '').split(os.pathsep) if entry and not project_sdk(entry)]
    preferred = [str(runtime), str(libraries)]
    paths = [entry for entry in paths if entry.casefold() not in {p.casefold() for p in preferred}]
    env['PATH'] = os.pathsep.join(preferred + paths)
    for key in ('ROCM_PATH', 'HIP_PATH', 'HIP_DEVICE_LIB_PATH', 'ROCBLAS_TENSILE_LIBPATH',
                'HIPBLASLT_TENSILE_LIBPATH', 'ROCM_KPACK_PATH'):
        if key in env and project_sdk(env[key]):
            del env[key]
    return env


def _load_manifest(path):
    if not path.is_file():
        raise RuntimeError(f'LLM 執行環境缺少驗證清單：{path}；請恢復 RUNTIMES 內的 ROCm 元件與清單。')
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise RuntimeError(f'LLM ROCm 驗證清單無法讀取：{path}') from exc
    if manifest.get('sdk_version') != '10.1.0' or not isinstance(manifest.get('files'), dict) or not manifest['files']:
        raise RuntimeError(f'LLM ROCm 驗證清單內容不符：{path}')
    return manifest


def _verify_files(manifest, base, allowed_root, required):
    paths = set()
    for relative, expected in manifest['files'].items():
        path = (base / relative).resolve()
        if not path.is_relative_to(allowed_root) or not path.is_file():
            raise RuntimeError(f'LLM ROCm 元件未齊備：{relative}')
        paths.add(path)
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != expected:
            raise RuntimeError(f'LLM ROCm 元件 SHA256 不符：{relative}')
    if any((allowed_root / name).resolve() not in paths for name in required):
        raise RuntimeError(f'LLM ROCm 驗證清單未包含必要元件：{allowed_root}')


def verify_managed_runtime(executable, base):
    base = Path(base).resolve()
    runtime = Path(executable).resolve().parent
    if runtime.name != MANAGED_RUNTIME:
        return
    sdk = (base / SDK_RELATIVE).resolve()
    manifest = _load_manifest(runtime / 'rocm-runtime.json')
    if (base / manifest.get('runtime', '')).resolve() != runtime:
        raise RuntimeError('LLM ROCm SDK 設定不符')
    expected_sdk_manifest = sdk / 'sdk-manifest.json'
    if (base / manifest.get('sdk_manifest', '')).resolve() != expected_sdk_manifest:
        raise RuntimeError('LLM ROCm SDK 驗證路徑不符')
    sdk_manifest = _load_manifest(expected_sdk_manifest)
    if (base / sdk_manifest.get('library_bin', '')).resolve() != sdk / 'bin' or sdk_manifest.get('gpu_arch') != 'gfx1201':
        raise RuntimeError('LLM ROCm SDK 裝置或函式庫路徑不符')
    _verify_files(manifest, base, runtime, CORE_NAMES)
    _verify_files(sdk_manifest, base, sdk, SDK_NAMES)


def cpu_requested(command):
    options = {'-ngl', '--gpu-layers', '--n-gpu-layers', '-dev', '--device'}
    for index, token in enumerate(command):
        flag, equal, value = token.partition('=')
        if flag not in options:
            continue
        if not equal:
            value = command[index + 1] if index + 1 < len(command) else ''
        if flag in {'-dev', '--device'} and value.lower() == 'none':
            return True
        if flag not in {'-dev', '--device'} and value == '0':
            return True
    return False


def prepare_llama_launch(command, base):
    executable = command[0]
    runtime = Path(executable).resolve().parent
    env = llama_environment(executable, base)
    verify_managed_runtime(executable, base)
    if not (runtime / 'ggml-hip.dll').is_file() or cpu_requested(command):
        return env, ''
    result = subprocess.run([executable, '--list-devices'], cwd=runtime, env=env,
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    output = result.stdout + result.stderr
    devices = [line.strip() for line in output.splitlines() if re.search(r'ROCm\d+\s*:', line)]
    if result.returncode or not devices:
        raise RuntimeError('ROCm GPU 後端未載入，已停止啟動。請檢查 hipBLAS／rocBLAS 與裝置核心包，'
                           '或選擇 llama-b11642-rocm-sdk-10.1.0-x64/。裝置偵測：' + output[-1500:])
    note = '[GPU 預檢] ' + '; '.join(devices) + '\n'
    if runtime.name == MANAGED_RUNTIME:
        note += '[ROCm] b11642；SDK 10.1.0／gfx1201；專案元件 SHA256 驗證完成\n'
        note += f'[ROCm] BLAS：{Path(base) / SDK_RELATIVE / "bin"}\n'
    return env, note
