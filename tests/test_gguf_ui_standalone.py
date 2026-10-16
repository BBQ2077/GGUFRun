"""gguf-ui.py 單獨存在（旁邊沒有 llm_rocm.py／llm_exl3.py）時也能啟動並組出 GGUF 指令。"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

PROBE = r'''
import importlib.util, sys, tkinter as tk
spec = importlib.util.spec_from_file_location("gguf_ui", "gguf-ui.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)
assert "llm_exl3" not in sys.modules and "llm_rocm" not in sys.modules
env, note = ui.prepare_llama_launch(["llama-server.exe"], ".")
assert note == "" and "PATH" in env
ui.save_settings = lambda d: None
root = tk.Tk(); root.withdraw()
app = ui.App(root)
vals = list(app.cb_model.cget("values"))
assert any("demo.gguf" in v for v in vals), vals
app.cb_model.set(next(v for v in vals if "demo.gguf" in v))
app.cb_model.event_generate("<<ComboboxSelected>>")
root.update()
cmd = app.build_cmd()
assert "-m" in cmd and "--vision" not in cmd, cmd
assert not app.is_exl3()
root.destroy()
print("OK", ui.EXL3_RUNTIME)
'''


class StandaloneTests(unittest.TestCase):
    def test_runs_without_optional_modules(self):
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            shutil.copy(ROOT / "gguf-ui.py", base / "gguf-ui.py")
            (base / "LLM-MODELS").mkdir()
            (base / "LLM-MODELS" / "demo.gguf").write_bytes(b"GGUF")
            r = subprocess.run([sys.executable, "-X", "utf8", "-c", PROBE], cwd=base, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=60,
                               env=dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1"))
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-3000:])
            self.assertIn("OK None", r.stdout)


if __name__ == "__main__":
    unittest.main()
