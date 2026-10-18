"""Copy a finished build into RUNTIMES/ and write the SHA256 manifest image-ui.py verifies before launch.

usage: package_runtime.py <build/bin> <rocm-sdk venv> [output-runtime-dir]
"""
import hashlib, json, pathlib, shutil, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NAME = 'sd-master-b13a482-moss4-gfx1201-rocm-10.1.0-x64'
BUILD_FILES = ('sd-cli.exe', 'sd-server.exe', 'stable-diffusion.dll')
# HIP core libraries the binaries load directly; the rest come from RUNTIMES/rocm-sdk-10.1.0/bin.
CORE_FILES = ('amd_comgr.dll', 'amdhip64_7.dll', 'rocm_kpack.dll')


def main(build_bin, venv, out=None):
    build_bin, venv = pathlib.Path(build_bin), pathlib.Path(venv)
    out = pathlib.Path(out) if out else ROOT / 'RUNTIMES' / NAME
    core = venv / 'Lib' / 'site-packages' / '_rocm_sdk_core' / 'bin'
    out.mkdir(parents=True, exist_ok=True)
    for name in BUILD_FILES:
        shutil.copy2(build_bin / name, out / name)
    for name in CORE_FILES:
        shutil.copy2(core / name, out / name)
    shutil.copy2(HERE / 'ggufrun-moss-dataspace.patch', out / 'ggufrun-moss-dataspace.patch')
    rel = 'RUNTIMES/' + NAME
    files = {}
    for name in sorted(CORE_FILES + BUILD_FILES):
        files[f'{rel}/{name}'] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    manifest = {
        'sdk_version': '10.1.0',
        'runtime': rel,
        'sdk_manifest': 'RUNTIMES/rocm-sdk-10.1.0/sdk-manifest.json',
        'source': 'local build: leejet/stable-diffusion.cpp master b13a482 + ggufrun-moss-dataspace.patch '
                  '(gfx1201, ROCm 10.1.0 pip SDK)',
        'files': files,
    }
    (out / 'rocm-runtime.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('packaged', out)


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(*sys.argv[1:])
