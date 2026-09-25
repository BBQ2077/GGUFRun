#!/usr/bin/env python3
"""Multi-connection ranged downloader (GitHub/HF edge caps single connections)."""
import sys, os, threading, urllib.request, time

def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"]), r.headers.get("Accept-Ranges","")

def fetch(url, start, end, path, idx):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path,"wb") as f:
        while True:
            chunk = r.read(65536)
            if not chunk: break
            f.write(chunk)

def main(url, out, n=8):
    total, ranges = head_size(url)
    print(f"total={total} ranges={ranges} conns={n}", flush=True)
    part = total // n
    parts_dir = out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    bounds = []
    for i in range(n):
        s = i*part
        e = (total-1) if i==n-1 else (s+part-1)
        bounds.append((s,e))
    # resume: skip completed parts
    done = {}
    for i,(s,e) in enumerate(bounds):
        p = os.path.join(parts_dir, f"{i:03d}.part")
        if os.path.exists(p) and os.path.getsize(p) == (e-s+1):
            done[i]=p
    def worker(i):
        s,e = bounds[i]
        p = os.path.join(parts_dir, f"{i:03d}.part")
        fetch(url, s, e, p, i)
    threads=[]
    for i in range(n):
        if i in done: continue
        t=threading.Thread(target=worker,args=(i,)); t.start(); threads.append(t)
    last=0
    while any(t.is_alive() for t in threads):
        time.sleep(3)
        cur=sum(os.path.getsize(os.path.join(parts_dir,f"{i:03d}.part")) for i in range(n) if os.path.exists(os.path.join(parts_dir,f"{i:03d}.part")))
        sp=(cur-last)/3/1e6
        print(f"  {cur/1e6:.1f}/{total/1e6:.1f} MB ({100*cur/total:.1f}%) {sp:.2f} MB/s", flush=True)
        last=cur
    for t in threads: t.join()
    with open(out,"wb") as f:
        for i in range(n):
            with open(os.path.join(parts_dir,f"{i:03d}.part"),"rb") as pf:
                f.write(pf.read())
    print(f"WROTE {out} {os.path.getsize(out)} bytes", flush=True)

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 8)
