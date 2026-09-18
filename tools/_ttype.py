import struct, sys, os
from collections import Counter

# GGML type id -> name（依 llama.cpp 的 ggml_type 列舉順序）
GGML_TYPES = {
    0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",4:"Q4_2",5:"Q4_3",6:"Q5_0",7:"Q5_1",
    8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",
    15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",
    21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",
    28:"F64",29:"IQ1_M",30:"BF16",31:"Q4_0_4_4",32:"Q4_0_4_8",
    33:"Q4_0_8_8",34:"TQ1_0",35:"TQ2_0",36:"IQ4_NL_4_4",37:"IQ4_NL_4_8",
    38:"IQ4_NL_8_8",39:"MXFP4",40:"COUNT",
}

def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def skip_value(f, t):
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t == 8:
        read_str(f)
    elif t == 9:
        at = struct.unpack("<I", f.read(4))[0]
        an = struct.unpack("<Q", f.read(8))[0]
        if at == 8:
            for _ in range(an): read_str(f)
        elif at in sz:
            f.seek(an * sz[at], 1)
    elif t in sz:
        f.read(sz[t])

path = sys.argv[1]
f = open(path, "rb")
assert f.read(4) == b"GGUF"
ver = struct.unpack("<I", f.read(4))[0]
n_tensors = struct.unpack("<Q", f.read(8))[0]
n_kv = struct.unpack("<Q", f.read(8))[0]
print(f"ver={ver} tensors={n_tensors} kv={n_kv}")

for _ in range(n_kv):
    k = read_str(f)
    t = struct.unpack("<I", f.read(4))[0]
    skip_value(f, t)

cnt = Counter()
unknown = []
for i in range(n_tensors):
    name = read_str(f)
    nd = struct.unpack("<I", f.read(4))[0]
    dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
    ttype = struct.unpack("<I", f.read(4))[0]
    f.read(8)  # offset
    cnt[ttype] += 1
    if ttype not in GGML_TYPES:
        unknown.append((name, ttype))

print("\n=== tensor 型別分布 ===")
for t, c in sorted(cnt.items()):
    print(f"  type {t:3d} {GGML_TYPES.get(t,'★未知★'):12s} x{c}")
if unknown:
    print("\n★ 未知型別前 10 個 tensor：")
    for n, t in unknown[:10]:
        print(f"    {n}  -> type {t}")
print(f"\n未知型別 tensor 總數: {len(unknown)}")
f.close()
