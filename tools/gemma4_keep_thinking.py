r"""從 Gemma 4 GGUF 取出內建 chat template，改成「保留舊輪思考」後存成 .jinja。

Gemma 4 內建模板只保留「最後一個 user 之後、且有 tool call」那幾輪的思考；
其他舊輪思考在下一輪都會被拿掉，前文跟上一輪不同 → llama-server 每輪都要重讀上一輪回答。
改成保留後：多輪對話不用重讀、工具前後的思考也不會斷線；代價是上下文漲得比較快。

用法：
    python tools/gemma4_keep_thinking.py <模型.gguf> [輸出.jinja]
預設輸出到 LLM-MODELS/templates/<模型檔名>-keep-thinking.jinja，
再到 UI「進階取樣設定 → 前文快取／模板 → 聊天模板檔」選這個檔（--chat-template-file）。
"""
import os
import struct
import sys

KEY = b"tokenizer.chat_template"
OLD = "{%- if thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls') -%}"
NEW = "{%- if thinking_text -%}"
GGUF_STRING = 8


def read_chat_template(path, scan_bytes=64 * 1024 * 1024):
    with open(path, "rb") as f:
        head = f.read(scan_bytes)
    if head[:4] != b"GGUF":
        raise ValueError("不是 GGUF 檔")
    i = head.find(KEY)
    while i >= 0:
        p = i + len(KEY)
        if struct.unpack_from("<I", head, p)[0] == GGUF_STRING:
            n = struct.unpack_from("<Q", head, p + 4)[0]
            return head[p + 12:p + 12 + n].decode("utf-8")
        i = head.find(KEY, p)
    raise ValueError("GGUF 裡找不到 tokenizer.chat_template")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = sys.argv[1]
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        base, "LLM-MODELS", "templates",
        os.path.splitext(os.path.basename(src))[0] + "-keep-thinking.jinja")
    tmpl = read_chat_template(src)
    if tmpl.count(OLD) != 1:
        print("這個模板沒有 Gemma 4 的「只保留工具輪思考」判斷式，沒有產生檔案。")
        return 1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(tmpl.replace(OLD, NEW))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
