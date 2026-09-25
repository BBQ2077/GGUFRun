"""Fuse Viggle Qwen-Image-2.1 LoRA MLP gate/proj for fused gate_up GGUF.

Only the two first-layer LoRA modules per transformer block are changed; all
other tensors are copied bit-for-bit. Original input remains untouched.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import numpy as np


def fuse_mlp_pair(gate_a, gate_b, proj_a, proj_b):
    if gate_a.ndim != 2 or gate_b.ndim != 2 or proj_a.ndim != 2 or proj_b.ndim != 2:
        raise ValueError('LoRA tensors must be matrices')
    if gate_a.shape[1] != proj_a.shape[1] or gate_b.shape[0] != proj_b.shape[0] or gate_b.shape[1] != gate_a.shape[0] or proj_b.shape[1] != proj_a.shape[0]:
        raise ValueError('Incompatible LoRA shapes')
    rg, rp = gate_a.shape[0], proj_a.shape[0]
    fused_a = np.concatenate((gate_a, proj_a), axis=0)
    fused_b = np.zeros((gate_b.shape[0] + proj_b.shape[0], rg + rp), dtype=gate_b.dtype)
    fused_b[:gate_b.shape[0], :rg] = gate_b
    fused_b[gate_b.shape[0]:, rg:] = proj_b
    return fused_a, fused_b


def convert(source: Path, target: Path):
    if source.resolve() == target.resolve() or target.exists():
        raise ValueError('Target must be new and different from source')
    with source.open('rb') as stream:
        header_len = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_len))
        origin_data = 8 + header_len
        source_keys = set(header) - {'__metadata__'}
        blocks = sorted({int(k.split('.')[2]) for k in source_keys if k.startswith('transformer.transformer_blocks.') and '.img_mlp.gate_layer.lora_A.weight' in k})
        if blocks != list(range(32)):
            raise ValueError(f'Expected all 32 Qwen-Image-2.1 transformer blocks, found {blocks}')
        extras = {}
        removed = set()
        for i in blocks:
            base = f'transformer.transformer_blocks.{i}.img_mlp.'
            keys = [base + which + '.lora_' + ab + '.weight' for which in ('gate_layer', 'proj') for ab in ('A', 'B')]
            if not all(k in source_keys for k in keys):
                raise ValueError(f'Missing LoRA tensors for block {i}')
            if not all(header[k]['dtype'] == 'BF16' for k in keys):
                raise ValueError('Expected BF16 tensors; refusing to reinterpret dtype')
            def read(k):
                info = header[k]; a, b = info['data_offsets']
                stream.seek(origin_data + a)
                raw = stream.read(b - a)
                if len(raw) != b - a or len(raw) != 2 * np.prod(info['shape']):
                    raise ValueError(f'Invalid tensor bytes for {k}')
                return np.frombuffer(raw, dtype='<u2').reshape(info['shape'])
            a, b = fuse_mlp_pair(*(read(k) for k in keys))
            extras[base + 'gate_up.lora_A.weight'] = a.tobytes(), list(a.shape)
            extras[base + 'gate_up.lora_B.weight'] = b.tobytes(), list(b.shape)
            removed.update(keys)
        if len(removed) != 128 or any(k in source_keys for k in extras):
            raise ValueError('Unexpected LoRA structure')
        retained = sorted(source_keys - removed)
        entries = {}
        offset = 0
        for k in retained:
            inf = header[k]; start, end = inf['data_offsets']
            entries[k] = {**inf, 'data_offsets': [offset, offset + end - start]}
            offset += end - start
        for k, (raw, shape) in extras.items():
            entries[k] = {'dtype': 'BF16', 'shape': shape, 'data_offsets': [offset, offset + len(raw)]}
            offset += len(raw)
        if '__metadata__' in header:
            entries['__metadata__'] = header['__metadata__']
        h = json.dumps(entries, separators=(',', ':')).encode('utf-8')
        tmp = target.with_suffix(target.suffix + '.partial')
        if tmp.exists():
            raise ValueError(f'Partial output exists: {tmp}')
        try:
            with tmp.open('xb') as dst:
                dst.write(struct.pack('<Q', len(h)))
                dst.write(h)
                for k in retained:
                    start, end = header[k]['data_offsets']; stream.seek(origin_data + start)
                    remaining = end - start
                    while remaining:
                        chunk = stream.read(min(4 * 1024 * 1024, remaining))
                        if not chunk: raise ValueError(f'Short read: {k}')
                        dst.write(chunk); remaining -= len(chunk)
                for raw, _ in extras.values(): dst.write(raw)
            if tmp.stat().st_size != 8 + len(h) + offset:
                raise ValueError('Output size mismatch')
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return len(retained), len(extras), target.stat().st_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    print(convert(args.source, args.target), flush=True)
