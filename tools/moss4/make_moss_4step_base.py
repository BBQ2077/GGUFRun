"""Build the MOSS-Image 4-step base for the moss4 runtime.

Copies IMAGE-MODELS/moss_image_sft_int8_convrot.safetensors and swaps in the fp32 time-embedding /
modulation-bias tensors from akhaliq/MOSS-Image-SFT-4-steps (time_embedding.safetensors, put it in
IMAGE-MODELS/moss-4step/). Output: IMAGE-MODELS/moss_image_sft_4step_int8_convrot.safetensors.
The 4-step LoRA stays a separate file and is applied unmerged at runtime by sd.cpp.
"""
import json, os, pathlib, struct, sys

MODELS = pathlib.Path(__file__).resolve().parent.parent.parent / 'IMAGE-MODELS'
BASE = str(MODELS / 'moss_image_sft_int8_convrot.safetensors')
TE = str(MODELS / 'moss-4step' / 'time_embedding.safetensors')
OUT = str(MODELS / 'moss_image_sft_4step_int8_convrot.safetensors')


def read_header(path):
    f = open(path, 'rb')
    n = struct.unpack('<Q', f.read(8))[0]
    return f, json.loads(f.read(n)), 8 + n


bf, bh, bstart = read_header(BASE)
tf, th, tstart = read_header(TE)
repl = {k: v for k, v in th.items() if k != '__metadata__'}
missing = [k for k in repl if k not in bh]
if missing:
    sys.exit(f'base lacks tensors: {missing}')
for k, v in repl.items():
    if [int(x) for x in v['shape']] != [int(x) for x in bh[k]['shape']]:
        sys.exit(f'shape mismatch {k}: {v["shape"]} vs {bh[k]["shape"]}')

names = [k for k in bh if k != '__metadata__']
names.sort(key=lambda k: bh[k]['data_offsets'][0])
new_h = {}
meta = dict(bh.get('__metadata__') or {})
meta['ggufrun_moss_4step'] = 'time_embedding/modulation biases from akhaliq/MOSS-Image-SFT-4-steps (fp32)'
new_h['__metadata__'] = meta
plan = []
off = 0
for k in names:
    if k in repl:
        a, b = repl[k]['data_offsets']
        src = ('te', tstart + a, b - a)
        dtype, shape = repl[k]['dtype'], repl[k]['shape']
    else:
        a, b = bh[k]['data_offsets']
        src = ('base', bstart + a, b - a)
        dtype, shape = bh[k]['dtype'], bh[k]['shape']
    new_h[k] = {'dtype': dtype, 'shape': shape, 'data_offsets': [off, off + src[2]]}
    plan.append(src)
    off += src[2]

hb = json.dumps(new_h, separators=(',', ':')).encode('utf-8')
hb += b' ' * ((8 - len(hb) % 8) % 8)
tmp = OUT + '.part'
with open(tmp, 'wb') as o:
    o.write(struct.pack('<Q', len(hb)))
    o.write(hb)
    done = 0
    for which, pos, size in plan:
        f = bf if which == 'base' else tf
        f.seek(pos)
        left = size
        while left:
            chunk = f.read(min(left, 64 << 20))
            if not chunk:
                sys.exit('short read')
            o.write(chunk)
            left -= len(chunk)
        done += size
print('replaced', sorted(repl))
os.replace(tmp, OUT)
print('wrote', OUT, os.path.getsize(OUT))
