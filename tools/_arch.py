import struct, sys, os

def read_gguf_meta(path):
    """讀 GGUF header 的 KV metadata（只讀字串/數值，跳過大陣列）。"""
    f = open(path, "rb")
    magic = f.read(4)
    if magic != b"GGUF":
        return None
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    out = {"__version": ver, "__n_tensors": n_tensors, "__n_kv": n_kv}

    def rd_str():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    T = {0:("B",1),1:("b",1),2:("H",2),3:("h",2),4:("I",4),5:("i",4),
         6:("f",4),7:("?",1),10:("Q",8),11:("q",8),12:("d",8)}

    for _ in range(min(n_kv, 400)):
        try:
            key = rd_str()
        except Exception:
            break
        try:
            t = struct.unpack("<I", f.read(4))[0]
        except Exception:
            break
        if t == 8:      # string
            out[key] = rd_str()
        elif t == 9:    # array
            at = struct.unpack("<I", f.read(4))[0]
            an = struct.unpack("<Q", f.read(8))[0]
            if at == 8:
                if an > 40:   # tokenizer 之類的大陣列 → 跳過不讀
                    f.seek(0, 2); break
                out[key] = [rd_str() for _ in range(an)]
            elif at in T:
                fmt, sz = T[at]
                f.seek(an * sz, 1)
                out[key] = f"<array[{at}]x{an}>"
            else:
                f.seek(0, 2); break
        elif t in T:
            fmt, sz = T[t]
            out[key] = struct.unpack("<" + fmt, f.read(sz))[0]
        else:
            break
    f.close()
    return out

for p in sys.argv[1:]:
    if not os.path.exists(p):
        print(f"--- {p} 不存在"); continue
    try:
        m = read_gguf_meta(p)
    except Exception as e:
        print(f"--- {os.path.basename(p)}: 讀取失敗 {e}"); continue
    if not m:
        print(f"--- {os.path.basename(p)}: 不是 GGUF"); continue
    arch = m.get("general.architecture", "?")
    print(f"\n=== {os.path.basename(p)} ===")
    print(f"  gguf ver   : {m.get('__version')}  tensors={m.get('__n_tensors')}")
    print(f"  architecture: {arch}")
    for k in sorted(m):
        if k.startswith("__"):
            continue
        if any(x in k for x in ("block_count", "expert_count", "expert_used",
                                 "nextn", "predict_layers", "file_type", "rope",
                                 "context_length", "name", "quant")):
            v = m[k]
            if isinstance(v, str) and len(v) > 60:
                v = v[:60] + "…"
            print(f"    {k} = {v}")
