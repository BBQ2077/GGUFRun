"""EXL3 server.py 的看圖：圖片段落 → 視覺 embedding 佔位字串、Job 帶 embeddings、同圖重用。不載入模型。"""
import argparse
import base64
import importlib.util
import io
import pathlib
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER = ROOT / "RUNTIMES/exl3-gfx1201/server.py"
if SERVER.is_file():
    SPEC = importlib.util.spec_from_file_location("exl3_server_vision", SERVER)
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

try:
    from PIL import Image
except ImportError:          # 系統 Python 沒有 Pillow：解碼相關測試略過（runtime 內附 Pillow）
    Image = None
try:
    import jinja2
except ImportError:
    jinja2 = None
# prepare() 只從 exllamav3 拿取樣器；測試用替身，不必載入 torch
FAKE_EXL3 = types.SimpleNamespace(GreedySampler=lambda: "greedy",
                                  model_init=types.SimpleNamespace(get_arg_sampler=lambda a: "sampler"))


def png_url(color=(255, 0, 0), size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img_part(url):
    return {"type": "image_url", "image_url": {"url": url}}


class Content(unittest.TestCase):
    def test_image_without_vision_rejected(self):
        with self.assertRaisesRegex(ValueError, "沒有開視覺"):
            srv.text_content([img_part("data:image/png;base64,AAAA")])

    def test_image_becomes_placeholder_in_order(self):
        seen = []
        text = srv.text_content([{"type": "text", "text": "看這張"}, img_part("U1"), {"type": "image_url", "image_url": "U2"}],
                                lambda u: seen.append(u) or f"<$EMB_{u}$>")
        self.assertEqual(seen, ["U1", "U2"])
        self.assertEqual(text, "看這張\n\n<$EMB_U1$>\n\n<$EMB_U2$>")

    def test_audio_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "不支援音訊"):
            srv.text_content([{"type": "input_audio", "input_audio": {}}], lambda u: "x")

    def test_only_user_images(self):
        msgs = [{"role": "system", "content": [img_part("S")]}, {"role": "user", "content": "q"}]
        with self.assertRaisesRegex(ValueError, "沒有開視覺"):
            srv.template_messages(msgs, lambda u: "x")

    def test_builtin_formatter_needs_template(self):
        with self.assertRaisesRegex(ValueError, "chat_template"):
            srv.text_messages([{"role": "user", "content": [img_part("U")]}])


@unittest.skipIf(Image is None, "需要 Pillow")
class Decode(unittest.TestCase):
    def test_png(self):
        raw, img = srv.decode_image(png_url())
        self.assertEqual(img.size, (8, 6))
        self.assertTrue(raw.startswith(b"\x89PNG"))

    def test_remote_url_rejected(self):
        with self.assertRaisesRegex(ValueError, "不抓外部網址"):
            srv.decode_image("http://example.com/a.png")

    def test_not_an_image(self):
        with self.assertRaisesRegex(ValueError, "無法讀取圖片"):
            srv.decode_image("data:image/png;base64," + base64.b64encode(b"hello").decode())


class FakeEmb:
    def __init__(self, n):
        self.text_alias = f"<$EMB_{n}$>"
        self.embeddings = type("E", (), {"shape": (12, 5120)})()


class FakeVision:
    def __init__(self):
        self.calls = 0

    def get_image_embeddings(self, tokenizer, image):
        self.calls += 1
        return FakeEmb(self.calls)


class FakeTokenizer:
    bos_token, eos_token = "", "<|im_end|>"

    def __init__(self):
        self.last = None

    def encode(self, text, add_bos=False, encode_special_tokens=True, embeddings=None):
        self.last = (text, embeddings)
        return type("Ids", (), {"shape": (1, len(text))})()


@unittest.skipIf(Image is None or jinja2 is None or not (MODEL / "chat_template.jinja").exists(),
                 "需要 Pillow、jinja2 與模型的 chat_template")
class Prepare(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, {"exllamav3": FAKE_EXL3})
        patcher.start()
        self.addCleanup(patcher.stop)

    def engine(self):
        e = srv.Engine(argparse.Namespace(thinking="on", reasoning_effort=None, temperature=0.6, top_p=0.95, top_k=20,
                                          min_p=0.0, vision="ram"))
        e.vision, e.tokenizer = FakeVision(), FakeTokenizer()
        e.hf_template = srv.HFTemplate(srv.load_hf_template(str(MODEL)), "", "<|im_end|>")
        e.formatter, e.stop_ids, e.draft_tokens = None, [1], 7
        e.generator = type("G", (), {"max_total_tokens": 65536})()
        e.Job = lambda **kw: kw
        return e

    def test_job_gets_embeddings_and_same_image_reused(self):
        e, red, blue = self.engine(), png_url(), png_url((0, 0, 255))
        body = {"messages": [{"role": "user", "content": [img_part(red), {"type": "text", "text": "這是什麼"}]},
                             {"role": "assistant", "content": "紅色"},
                             {"role": "user", "content": [img_part(red), img_part(blue), {"type": "text", "text": "比較"}]}],
                "temperature": 0}
        with redirect_stdout(io.StringIO()):
            job, *_ = e.prepare(body)
        text, embs = e.tokenizer.last
        self.assertEqual(e.vision.calls, 2)                       # 紅圖只算一次
        self.assertEqual([x.text_alias for x in job["embeddings"]], ["<$EMB_1$>", "<$EMB_2$>"])
        self.assertIs(embs, job["embeddings"])
        self.assertEqual(text.count("<$EMB_1$>"), 2)
        self.assertEqual(text.count("<$EMB_2$>"), 1)
        with redirect_stdout(io.StringIO()):
            e.prepare(body)                                       # 下一輪：同圖沿用同一個 embedding（前文快取接得上）
        self.assertEqual(e.vision.calls, 2)

    def test_text_only_has_no_embeddings(self):
        e = self.engine()
        job, *_ = e.prepare({"messages": [{"role": "user", "content": "hi"}]})
        self.assertIsNone(job["embeddings"])

    def test_props_reports_vision(self):
        e = self.engine()
        e.args = argparse.Namespace(**vars(e.args), model_dir="M", alias="m", cache_size=65536, draft_model_dir=None,
                                    chat_mode="auto")
        e.ready = True
        self.assertEqual(e.props()["modalities"]["vision"], True)
        self.assertEqual(e.props()["vision"], "ram")
        e.vision = None
        self.assertEqual(e.props()["modalities"]["vision"], False)
        self.assertEqual(e.props()["vision"], "off")


if __name__ == "__main__":
    unittest.main()
