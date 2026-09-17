# -*- coding: utf-8 -*-
"""鍵盤診斷探針：把每個按鍵的 keysym / char / keycode 顯示出來並寫入 key-probe.log。

用法：雙擊同目錄的 run-key-probe.bat，在視窗裡按一下「.」鍵（或數字鍵盤的 .），
      然後把視窗關掉。日誌在同目錄 key-probe.log。
"""
import os
import tkinter as tk

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "key-probe.log")

root = tk.Tk()
root.title("鍵盤診斷 — 請按「.」鍵")
root.geometry("680x340")

tk.Label(root, text="請在下方空白處點一下，然後按「.」鍵（或數字鍵盤的 .），最後關掉視窗。",
         font=("Microsoft JhengHei", 11)).pack(pady=8)
box = tk.Text(root, height=10, font=("Consolas", 11))
box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
box.focus_set()

with open(LOG, "w", encoding="utf-8") as f:
    f.write("=== key-probe start ===\n")


def on_key(e):
    line = f"keysym={e.keysym!r}  char={e.char!r}  keycode={e.keycode}  state={e.state}"
    box.insert("end", line + "\n")
    box.see("end")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


root.bind("<KeyPress>", on_key)

_ms = int(os.environ.get("PROBE_MS", "0") or 0)
if _ms:
    root.after(_ms, root.destroy)

root.mainloop()
