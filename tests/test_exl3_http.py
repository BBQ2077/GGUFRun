"""EXL3 server.py 的 HTTP 層：llama.cpp 網頁（llama-server 內建 UI）的請求是否相容。用假引擎，不載入模型。"""
import argparse
import importlib.util
import io
import json
import pathlib
import os
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
# EXL3_TEST_RUNTIME can point to a separately checked-out runtime repository.
SERVER = (pathlib.Path(os.environ["EXL3_TEST_RUNTIME"]).resolve() / "server.py"
          if os.environ.get("EXL3_TEST_RUNTIME") else
          next((p for p in (ROOT / "RUNTIMES/exl3-gfx1201-b/server.py", ROOT / "RUNTIMES/exl3-gfx1201/server.py")
                if p.is_file()), ROOT / "RUNTIMES/exl3-gfx1201/server.py"))
if SERVER.is_file():
    SPEC = importlib.util.spec_from_file_location("exl3_server", SERVER)
    srv = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(srv)
else:  # GGUFRun 沒裝 EXL3 runtime：整個檔案略過
    srv = None

    class _NoRuntime(unittest.TestCase):
        @unittest.skip("需要 EXL3 runtime（RUNTIMES/exl3-gfx1201）")
        def test_runtime_missing(self):
            pass

    def load_tests(loader, tests, pattern):
        return loader.loadTestsFromTestCase(_NoRuntime)


class FakeEngine(srv.Engine if srv else object):
    def __init__(self):
        self.args = argparse.Namespace(alias="Qwen3.8-27B-EXL3", cache_size=65536, model_dir="M", draft_model_dir=None,
                                       chat_mode="auto", thinking="on", temperature=1.0, top_p=0.95, top_k=20, min_p=0.0)
        super().__init__(self.args)
        self.ready = True
        self.vision = None
        self.lock = threading.Lock()
        self.cancelled = threading.Event()

    def props(self):
        return srv.Engine.props(self)


class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
        cls.httpd.daemon_threads = True
        cls.httpd.engine = FakeEngine()
        cls.port = cls.httpd.server_port
        cls.log = io.StringIO()
        cls.t = threading.Thread(target=cls._serve, daemon=True)
        cls.t.start()

    @classmethod
    def _serve(cls):
        with redirect_stdout(cls.log):
            cls.httpd.serve_forever(poll_interval=0.05)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def req(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data,
                                   headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(r, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_props_with_query(self):
        code, p = self.req("/props?autoload=false")
        self.assertEqual(code, 200)
        self.assertEqual(p["default_generation_settings"]["n_ctx"], 65536)
        self.assertEqual(p["default_generation_settings"]["params"]["top_k"], 20)
        self.assertNotIn("role", p)                     # 單模型模式
        self.assertEqual(p["backend"], "exllamav3")      # GGUFRun 介面靠這欄判斷

    def test_models_slots_tools(self):
        code, m = self.req("/v1/models")
        self.assertEqual((code, m["data"][0]["id"], m["data"][0]["context_length"]), (200, "Qwen3.8-27B-EXL3", 65536))
        code, s = self.req("/slots?model=x")
        self.assertEqual((code, s[0]["is_processing"]), (200, False))
        self.assertEqual(self.req("/tools")[0], 403)
        self.assertEqual(self.req("/nope")[0], 404)

    def test_stream_lookup_and_control(self):
        self.assertEqual(self.req("/v1/streams/lookup", {"conversation_ids": ["a"]}), (200, []))
        code, c = self.req("/v1/chat/completions/control", {"action": "end_reasoning", "id": "x"})
        self.assertEqual((code, c["success"]), (200, False))

    def test_pre_encode_does_not_lock(self):
        code, r = self.req("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                    "n_predict": 0, "stream": False})
        self.assertEqual((code, r["choices"][0]["message"]["content"]), (200, ""))
        self.assertEqual(self.httpd.engine.props()["active_slots"], 0)

    def test_polling_not_logged(self):
        before = self.log.getvalue().count("GET /props")
        for _ in range(3):
            self.req("/props")
        self.req("/nope")
        out = self.log.getvalue()
        self.assertEqual(out.count("GET /props"), before)
        self.assertIn("GET /nope", out)


class Body(unittest.TestCase):
    def test_max_tokens_minus_one(self):
        self.assertEqual(srv.response_budget(100, 65536, -1, 0), 65536 - 100 - 1)
        self.assertEqual(srv.response_budget(100, 65536, 50, 0), 50)
        with self.assertRaises(ValueError):
            srv.response_budget(100, 65536, -2, 0)

    def test_content_parts(self):
        m = srv.text_messages([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                               {"role": "assistant", "content": "x", "reasoning_content": "r"}])
        self.assertEqual(m, [{"role": "user", "content": "a\n\nb"}, {"role": "assistant", "content": "x"}])
        with self.assertRaises(ValueError):
            srv.text_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:"}}]}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
