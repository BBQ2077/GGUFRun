"""EXL3 server.py 的工具呼叫：用模型自己的 chat_template 渲染、解析 Qwen XML 工具呼叫。不載入模型。"""
import importlib.util
import json
import pathlib
import os
import unittest
import unittest.mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
# EXL3_TEST_RUNTIME can point to a separately checked-out runtime repository.
SERVER = (pathlib.Path(os.environ["EXL3_TEST_RUNTIME"]).resolve() / "server.py"
          if os.environ.get("EXL3_TEST_RUNTIME") else
          next((p for p in (ROOT / "RUNTIMES/exl3-gfx1201-b/server.py", ROOT / "RUNTIMES/exl3-gfx1201/server.py")
                if p.is_file()), ROOT / "RUNTIMES/exl3-gfx1201/server.py"))
if SERVER.is_file():
    SPEC = importlib.util.spec_from_file_location("exl3_server_tools", SERVER)
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

MODEL = ROOT / "LLM-MODELS-EXL3/Qwen3.8-27B-EXL3-SC3.0bpw-H4-V4"

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                         "metric": {"type": "boolean"}, "tags": {"type": "array"}, "note": {"type": ["string", "null"]}},
                         "required": ["city"]}}}]

CALL_TEXT = ("I'll check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n台北\n</parameter>\n"
             "<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n"
             "<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n<parameter=note>\nline1\nline2\n</parameter>\n"
             "</function>\n</tool_call>")


def run(parser, text, size):
    out = []
    for i in range(0, len(text), size):
        out += parser.feed(text[i:i + size])
    return out + parser.feed("", final=True)


class Parser(unittest.TestCase):
    def test_every_chunk_size(self):
        for size in (1, 2, 3, 7, 50, 10_000):
            out = run(srv.ToolCallParser(TOOLS), CALL_TEXT, size)
            text = "".join(c for k, c in out if k == "content")
            calls = [c for k, c in out if k == "tool_call"]
            self.assertEqual(text, "I'll check.", size)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["function"]["name"], "get_weather")
            self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                             {"city": "台北", "days": 3, "metric": True, "tags": ["a", "b"], "note": "line1\nline2"})

    def test_plain_text_untouched(self):
        for size in (1, 4, 1000):
            out = run(srv.ToolCallParser(TOOLS), "a < b, x <tool and <tool_c ok\n\nend", size)
            self.assertEqual("".join(c for _, c in out), "a < b, x <tool and <tool_c ok\n\nend")

    def test_two_calls_and_unterminated(self):
        two = CALL_TEXT + "\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>"
        calls = [c for k, c in run(srv.ToolCallParser(TOOLS), two, 5) if k == "tool_call"]
        self.assertEqual([c["index"] for c in calls], [0, 1])
        cut = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOsaka"      # 被 max_tokens 截斷
        calls = [c for k, c in run(srv.ToolCallParser(TOOLS), cut, 4) if k == "tool_call"]
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Osaka"})

    def test_tools_validation(self):
        self.assertIsNone(srv.request_tools(None, None))
        self.assertIsNone(srv.request_tools(TOOLS, "none"))
        self.assertEqual(srv.request_tools(TOOLS, "auto"), TOOLS)
        with self.assertRaises(ValueError):
            srv.request_tools([{"type": "function", "function": {}}], None)


@unittest.skipUnless((MODEL / "chat_template.jinja").is_file(), "model not present")
class Template(unittest.TestCase):
    def setUp(self):
        try:
            self.tpl = srv.HFTemplate(srv.load_hf_template(MODEL), "", "<|im_end|>")
        except ImportError:
            self.skipTest("jinja2 not installed")

    def test_tool_round_trip(self):
        msgs = [{"role": "system", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "台北天氣？"}]},
                {"role": "assistant", "content": "", "reasoning_content": "need weather",
                 "tool_calls": [{"id": "call_1", "type": "function",
                                 "function": {"name": "get_weather", "arguments": "{\"city\": \"台北\", \"days\": 3}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "晴 28°C"}]
        text = self.tpl.render(srv.template_messages(msgs), True, {}, srv.request_tools(TOOLS, "auto"))
        self.assertIn("# Tools", text)
        self.assertIn('"name": "get_weather"', text)
        self.assertIn("be brief", text)
        self.assertIn("<tool_call>\n<function=get_weather>\n<parameter=city>\n台北\n</parameter>\n"
                      "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>", text)
        self.assertIn("<tool_response>\n晴 28°C\n</tool_response>", text)
        self.assertIn("need weather", text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n"))

    def test_no_tools_same_as_before(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertNotIn("# Tools", self.tpl.render(srv.template_messages(msgs), True, {}, None))

    def test_ends_with_assistant_rejected(self):
        with self.assertRaises(ValueError):
            srv.template_messages([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])


class FakeGenerator:
    """Yields the model's raw text in small pieces, like ExLlamaV3's generator (one prefill step first)."""
    def __init__(self, text, size=3):
        self.pieces = [text[i:i + size] for i in range(0, len(text), size)]
        self.max_total_tokens = 65536
        self.prefilled = False

    def enqueue(self, job):
        return 0

    def num_remaining_jobs(self):
        return 1 if self.pieces else 0

    def iterate(self):
        if not self.prefilled:
            self.prefilled = True
            return [{"serial": 0, "stage": "prefill", "eos": False, "curr_progress": 64, "max_progress": 99}]
        piece = self.pieces.pop(0)
        out = [{"serial": 0, "stage": "streaming", "text": piece}]
        if not self.pieces:
            out[0].update(eos=True, eos_reason="stop_token", new_tokens=40, prompt_tokens=100, time_generate=1.0,
                          time_prefill=0.5, cached_tokens=20, accepted_draft_tokens=30, rejected_draft_tokens=10)
        return out

    def cancel(self, job):
        pass


class ServerCase(unittest.TestCase):
    """Real Handler on a random port with a fake engine."""
    def serve(self, raw, fn, ready=True):
        import argparse
        import contextlib
        import io
        import threading
        from http.server import ThreadingHTTPServer

        class Engine(srv.Engine):
            def __init__(self):
                super().__init__(argparse.Namespace(alias="m", cache_size=65536))
                self.ready = True
                self.torch = type("T", (), {"inference_mode": staticmethod(contextlib.nullcontext)})
                self.generator = FakeGenerator(raw)
                self.scheduler = srv.GenerationScheduler(self, self.parallel)

            def prepare(self, b):
                tools = srv.request_tools(b.get("tools"), b.get("tool_choice"))
                return object(), 1000, True, 100, tools

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
        httpd.engine = Engine()
        httpd.engine.ready = ready
        httpd.engine.props = lambda: {"backend": "exllamav3", "status": "ok" if ready else "loading"}
        t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        with contextlib.redirect_stdout(io.StringIO()):
            t.start()
            try:
                return fn(f"http://127.0.0.1:{httpd.server_port}")
            finally:
                httpd.shutdown()
                httpd.engine.scheduler.close()
                httpd.server_close()

    def post(self, raw, body, path="/v1/chat/completions"):
        import urllib.request

        def go(base):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.read().decode()
        return self.serve(raw, go)

    def get(self, path, headers=None, ready=True):
        import urllib.error
        import urllib.request

        def go(base):
            try:
                with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers or {}), timeout=10) as r:
                    return r.status, dict(r.headers), r.read()
            except urllib.error.HTTPError as e:
                return e.code, dict(e.headers), e.read()
        return self.serve("x", go, ready)


class Progress(unittest.TestCase):
    """Periodic progress lines while prefilling / generating, like llama-server's print_timing."""
    def run_log(self, steps):
        import contextlib
        import io
        import types
        now = [1000.0]
        job = types.SimpleNamespace(time_first_prefill=1000.0, time_first_token=None, new_tokens=0,
                                    cached_pages=0, cached_tokens=20, accepted_draft_tokens=None, rejected_draft_tokens=None)
        live = srv.LiveTimings(job, 4116)
        log = srv.ProgressLog(live, clock=lambda: now[0])
        out = io.StringIO()
        with contextlib.redirect_stdout(out), unittest.mock.patch.object(srv.time, "time", lambda: now[0]):
            for t, kind, value in steps:
                now[0] = 1000.0 + t
                if kind == "prefill":
                    log.prefill({"curr_progress": value, "max_progress": 4116})
                else:
                    if job.time_first_token is None:
                        job.time_first_token = now[0]
                    job.new_tokens = value
                    if kind == "draft":
                        job.accepted_draft_tokens, job.rejected_draft_tokens = 30, 10
                    log.generate()
        return out.getvalue().splitlines()

    def test_prefill_lines_every_three_seconds(self):
        lines = self.run_log([(1, "prefill", 500), (3.2, "prefill", 1020), (4, "prefill", 2000), (6.4, "prefill", 2580)])
        self.assertEqual(lines, ["[EXL3] 預填 1000/4096 tok（24%）｜3.2 s｜312.5 tok/s",
                                 "[EXL3] 預填 2560/4096 tok（62%）｜6.4 s｜400.0 tok/s"])

    def test_generation_lines_show_average_and_recent_speed(self):
        lines = self.run_log([(10, "gen", 1), (12, "gen", 25), (13, "gen", 37), (14, "gen", 50), (16, "draft", 100)])
        self.assertEqual(lines, ["[EXL3] 生成 37 tok｜平均 12.3 tok/s｜近 3 秒 12.0 tok/s",
                                 "[EXL3] 生成 100 tok｜平均 16.7 tok/s｜近 3 秒 21.0 tok/s｜草稿接受 30/40（75%）"])

    def test_short_reply_prints_nothing_extra(self):
        self.assertEqual(self.run_log([(0.5, "prefill", 99), (1, "gen", 1), (2, "gen", 12)]), [])


class Stream(ServerCase):
    """Whole request through Handler: SSE deltas merged the way the llama.cpp web UI merges them."""
    def test_stream_tool_call(self):
        raw = "user wants weather\n</think>\n\n" + CALL_TEXT
        data = self.post(raw, {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS, "stream": True})
        events = [json.loads(l[6:]) for l in data.split("\n\n") if l.startswith("data: {")]
        self.assertTrue(data.rstrip().endswith("data: [DONE]"))
        deltas = [e["choices"][0]["delta"] for e in events]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        merged = {}
        for d in deltas:
            for c in d.get("tool_calls", []):                       # web UI: mergeToolCallDeltas by index
                m = merged.setdefault(c["index"], {"name": "", "arguments": ""})
                m["name"] = c["function"].get("name") or m["name"]
                m["arguments"] += c["function"].get("arguments", "")
        self.assertEqual(reasoning.strip(), "user wants weather")
        self.assertEqual(content, "I'll check.")
        self.assertNotIn("<tool_call>", content)
        self.assertEqual(json.loads(merged[0]["arguments"])["city"], "台北")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_non_stream_tool_call(self):
        data = json.loads(self.post("ok\n</think>\n\n" + CALL_TEXT,
                                    {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS}))
        msg = data["choices"][0]["message"]
        self.assertEqual(data["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertNotIn("index", msg["tool_calls"][0])

    def test_no_tools_text_kept(self):
        data = json.loads(self.post("ok\n</think>\n\n" + CALL_TEXT, {"messages": [{"role": "user", "content": "x"}]}))
        self.assertIn("<tool_call>", data["choices"][0]["message"]["content"])
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")

    def test_stream_timings_and_progress(self):
        data = self.post("ok\n</think>\n\nhello world", {"messages": [{"role": "user", "content": "x"}],
                                                          "stream": True, "return_progress": True})
        events = [json.loads(l[6:]) for l in data.split("\n\n") if l.startswith("data: {")]
        prog = [e["prompt_progress"] for e in events if "prompt_progress" in e]
        self.assertEqual(prog[0]["processed"], 64)
        self.assertEqual(prog[0]["total"], 99)
        live = [e["timings"] for e in events[:-1] if "timings" in e]
        self.assertTrue(live, "content chunks carry live timings for the web UI")
        t = events[-1]["timings"]                                   # final chunk: generator's own numbers
        self.assertEqual((t["predicted_n"], t["predicted_ms"], t["predicted_per_second"]), (40, 1000, 40))
        self.assertEqual((t["cache_n"], t["prompt_n"], t["prompt_ms"]), (20, 80, 500))
        self.assertEqual((t["draft_n"], t["draft_n_accepted"]), (40, 30))

    def test_no_progress_unless_asked(self):
        data = self.post("ok\n</think>\n\nhi", {"messages": [{"role": "user", "content": "x"}], "stream": True})
        self.assertNotIn("prompt_progress", data)

    def test_non_stream_timings(self):
        data = json.loads(self.post("ok\n</think>\n\nhi", {"messages": [{"role": "user", "content": "x"}]}))
        self.assertEqual(data["timings"]["predicted_per_second"], 40)


@unittest.skipUnless(srv and srv.UI_FILES, "llama.cpp web UI files not extracted")
class WebUI(ServerCase):
    def test_index_gzip_and_plain(self):
        st, h, body = self.get("/", {"Accept-Encoding": "gzip"})
        self.assertEqual((st, h.get("Content-Encoding")), (200, "gzip"))
        import gzip
        self.assertIn(b"<html", gzip.decompress(body).lower())
        st, h, body = self.get("/")
        self.assertEqual(st, 200)
        self.assertIsNone(h.get("Content-Encoding"))
        self.assertIn(b"<html", body.lower())

    def test_every_manifest_file(self):
        for path, meta in srv.UI_FILES.items():
            st, h, _ = self.get(path + ("?v=1" if path == "/build.json" else ""), {"Accept-Encoding": "gzip"})
            self.assertEqual(st, 200, path)
            self.assertEqual(h.get("Content-Type"), meta["type"], path)

    def test_sw_and_build(self):
        self.assertEqual(self.get("/sw.js")[0], 200)
        self.assertEqual(json.loads(self.get("/build.json")[2])["version"], "b11642")

    def test_props_while_loading(self):
        st, _, body = self.get("/props", {"Sec-Fetch-Mode": "cors"}, ready=False)   # llama.cpp 網頁：等待並重試
        self.assertEqual(st, 503)
        self.assertEqual(json.loads(body)["error"]["type"], "unavailable_error")
        st, _, body = self.get("/props", ready=False)                               # GGUFRun 介面輪詢
        self.assertEqual((st, json.loads(body)["status"]), (200, "loading"))
        self.assertEqual(self.get("/props", {"Sec-Fetch-Mode": "cors"})[0], 200)

    def test_old_page_and_tools(self):
        st, _, body = self.get("/exl3")
        self.assertEqual(st, 200)
        self.assertEqual(body, (srv.ROOT / "web/chat.html").read_bytes())
        self.assertEqual(self.get("/tools")[0], 403)
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post("x", {}, "/tools")
        self.assertEqual(cm.exception.code, 403)
        self.assertEqual(self.get("/nope")[0], 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
