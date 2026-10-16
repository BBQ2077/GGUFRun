import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import llm_exl3  # noqa: E402

SPEC = importlib.util.spec_from_file_location("gguf_ui", ROOT / "gguf-ui.py")
gguf_ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gguf_ui)

GiB = 2 ** 30


class Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_model(root, name, arch="Qwen3_5ForConditionalGeneration", size=GiB, version="0.0.40", draft=False,
               ngram=0):
    d = root / llm_exl3.MODEL_DIR_NAME / name
    cfg = {"architectures": [arch], "quantization_config": {"quant_method": "exl3", "version": version}}
    if draft:
        cfg.update(hidden_size=5120, vocab_size=248320, num_target_layers=64,
                   dflash_config={"block_size": 8})
    else:
        cfg["text_config"] = {"hidden_size": 5120, "vocab_size": 248320, "num_hidden_layers": 64}
    write_json(d / "config.json", cfg)
    (d / "model.safetensors").write_bytes(b"")
    FAKE_SIZES[os.path.normcase(str(d))] = (size, ngram)    # 只記大小，不寫實體大檔
    return str(d)


FAKE_SIZES = {}
REAL_WEIGHT_BYTES = llm_exl3.weight_bytes


def fake_weight_bytes(path):
    return FAKE_SIZES.get(os.path.normcase(str(path)), (0, 0))


def patch_sizes(case):
    patcher = mock.patch.object(llm_exl3, "weight_bytes", fake_weight_bytes)
    patcher.start()
    case.addCleanup(patcher.stop)


def make_runtime(root, archs=("Qwen3_5ForConditionalGeneration", "DFlash2DraftModel"), vram=16 * GiB,
                 version="1.6.2"):
    d = root / "RUNTIMES" / "exl3-test"
    write_json(d / "runtime.json", {"backend": "exllamav3"})
    write_json(d / "supported-architectures.json",
               {"exllamav3_version": version, "architectures": list(archs), "gpu_total_bytes": vram})
    (d / "python").mkdir(parents=True, exist_ok=True)
    (d / "python" / "python.exe").write_bytes(b"")
    return str(d)


class Exl3AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        patch_sizes(self)
        self.rt = make_runtime(self.base)
        self.main = make_model(self.base, "Qwen3.8-27B-EXL3-3.0bpw", size=12 * GiB)
        self.draft = make_model(self.base, "Qwen3.8-27B-DFlash2-EXL3", arch="DFlash2DraftModel", draft=True)
        (self.base / "LLM-MODELS").mkdir()
        (self.base / "LLM-MODELS" / "x.gguf").write_bytes(b"GGUF")

    def tearDown(self):
        self.tmp.cleanup()

    def kw(self, **over):
        kw = dict(port=18440, ctx="32768", ctk="q4_0", ctv="q4_0", think="on", effort="default",
                  speculative=True, drafter="", sampling={"temp": "0.8", "rpen": "", "drym": "0.5"})
        kw.update(over)
        return kw

    def test_discovery_separates_main_draft_and_gguf(self):
        base = str(self.base)
        self.assertEqual(llm_exl3.find_runtime(base), self.rt)
        self.assertEqual([n for n, _p, _s in llm_exl3.list_models(base)], ["Qwen3.8-27B-EXL3-3.0bpw"])
        self.assertEqual([n for n, _p, _s in llm_exl3.list_models(base, draft=True)], ["Qwen3.8-27B-DFlash2-EXL3"])
        self.assertFalse(llm_exl3.is_exl3(str(self.base / "LLM-MODELS")))
        self.assertFalse(llm_exl3.is_exl3(str(self.base / "LLM-MODELS" / "x.gguf")))

    def test_build_cmd_auto_draft_and_kv(self):
        cmd, note = llm_exl3.build_cmd(self.main, self.rt, str(self.base), **self.kw())
        self.assertEqual(cmd[cmd.index("-dm") + 1], self.draft)
        self.assertEqual(cmd[cmd.index("-ndt") + 1], "7")
        self.assertEqual(cmd[cmd.index("-cq") + 1], "4")
        self.assertEqual(cmd[cmd.index("-temp") + 1], "0.8")
        self.assertNotIn("-repp", cmd)
        self.assertNotIn("--reasoning-effort", cmd)
        self.assertIn("投機解碼", note)
        cmd, _ = llm_exl3.build_cmd(self.main, self.rt, str(self.base),
                                    **self.kw(ctk="q8_0", ctv="q4_0", effort="low", speculative=False))
        self.assertEqual(cmd[cmd.index("-cq") + 1], "8,4")
        self.assertEqual(cmd[cmd.index("--reasoning-effort") + 1], "low")
        self.assertNotIn("-dm", cmd)
        cmd, _ = llm_exl3.build_cmd(self.main, self.rt, str(self.base), **self.kw(ctk="f16", ctv="f16", drafter="-"))
        self.assertNotIn("-cq", cmd)
        self.assertNotIn("-dm", cmd)

    def test_vision_flag(self):
        base = str(self.base)
        cmd, _ = llm_exl3.build_cmd(self.main, self.rt, base, **self.kw())
        self.assertNotIn("--vision", cmd)                          # 預設 off
        cmd, note = llm_exl3.build_cmd(self.main, self.rt, base, **self.kw(vision="ram"))
        self.assertNotIn("--vision", cmd)                          # 沒有視覺塔的模型不送
        self.assertIn("沒有視覺塔", note)
        cfg = json.loads((pathlib.Path(self.main) / "config.json").read_text(encoding="utf-8"))
        cfg["vision_config"] = {"depth": 27}
        write_json(pathlib.Path(self.main) / "config.json", cfg)
        write_json(pathlib.Path(self.main) / "preprocessor_config.json", {"patch_size": 16})
        self.assertTrue(llm_exl3.has_vision(self.main))
        for mode in ("gpu", "ram"):
            cmd, note = llm_exl3.build_cmd(self.main, self.rt, base, **self.kw(vision=mode))
            self.assertEqual(cmd[cmd.index("--vision") + 1], mode)
            self.assertIn("[EXL3] 視覺", note)
        cmd, _ = llm_exl3.build_cmd(self.main, self.rt, base, **self.kw(vision=""))   # 舊設定檔：空字串＝off
        self.assertNotIn("--vision", cmd)
        with self.assertRaises(ValueError):
            llm_exl3.build_cmd(self.main, self.rt, base, **self.kw(vision="cpu"))

    def test_recurrent_cache_and_chunk(self):
        base = str(self.base)
        cmd, note = llm_exl3.build_cmd(self.main, self.rt, base, **self.kw(speculative=False))
        self.assertGreaterEqual(float(cmd[cmd.index("-rcs") + 1]), 1.0)
        self.assertEqual(cmd[cmd.index("-chunk_size") + 1], str(llm_exl3.CHUNK_SIZE))
        self.assertIn("多輪快取", note)
        cmd, _ = llm_exl3.build_cmd(self.main, self.rt, base, **self.kw())
        self.assertEqual(cmd[cmd.index("-chunk_size") + 1], str(llm_exl3.CHUNK_SIZE_DRAFT))
        cmd, _ = llm_exl3.build_cmd(self.main, self.rt, base, **self.kw(recurrent_gb=0, chunk=512))
        self.assertEqual(cmd[cmd.index("-rcs") + 1], "0")
        self.assertEqual(cmd[cmd.index("-chunk_size") + 1], "512")
        self.assertNotIn("{chunk}", cmd)

    def test_draft_max(self):
        base = str(self.base)
        for val, want in (("", "7"), ("3", "3"), ("7", "7"), ("12", "7")):
            with self.subTest(val=val):
                cmd, note = llm_exl3.build_cmd(self.main, self.rt, base, **self.kw(draft_max=val))
                self.assertEqual(cmd[cmd.index("-ndt") + 1], want)
                self.assertEqual("超過" in note, val == "12")
        for bad in ("0", "-1", "abc"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                llm_exl3.build_cmd(self.main, self.rt, base, **self.kw(draft_max=bad))
        cmd, _ = llm_exl3.build_cmd(self.main, self.rt, base, **self.kw(draft_max="0", speculative=False))
        self.assertNotIn("-ndt", cmd)

    def test_reason_table_covers_skip_list(self):
        self.assertNotIn("sdnmax", llm_exl3.NA_REASONS)
        self.assertNotIn("ctx", llm_exl3.NA_REASONS)
        for k in llm_exl3.LLAMA_ONLY:
            self.assertIn(k, llm_exl3.NA_REASONS)

    def test_rejects_bad_inputs(self):
        base = str(self.base)
        for over in (dict(ctx="1000"), dict(ctx="abc"), dict(ctk="q5_1"),
                     dict(drafter=str(self.base / "LLM-MODELS" / "x.gguf"))):
            with self.subTest(over=over), self.assertRaises(ValueError):
                llm_exl3.build_cmd(self.main, self.rt, base, **self.kw(**over))
        with self.assertRaises(ValueError):
            llm_exl3.build_cmd(self.draft, self.rt, base, **self.kw())

    def test_weight_bytes_splits_ngram_table(self):
        d = self.base / "small"
        d.mkdir()
        (d / "model-00001.safetensors").write_bytes(b"x" * 300)
        (d / "ngram_embedding_00.safetensors").write_bytes(b"x" * 50)
        (d / "config.json").write_bytes(b"{}")
        self.assertEqual(REAL_WEIGHT_BYTES(str(d)), (300, 50))

    def test_assess_levels(self):
        self.assertEqual(llm_exl3.assess(self.main, self.rt)[0], "ok")
        big = make_model(self.base, "Big", size=int(15.6 * GiB))
        self.assertEqual(llm_exl3.assess(big, self.rt)[0], "error")
        tight = make_model(self.base, "Tight", size=14 * GiB)
        self.assertEqual(llm_exl3.assess(tight, self.rt)[0], "warn")
        ng = make_model(self.base, "Ngram", size=10 * GiB, ngram=40 * GiB)
        level, why = llm_exl3.assess(ng, self.rt)
        self.assertEqual(level, "ok")
        self.assertIn("n-gram", why)
        arch = make_model(self.base, "Unknown", arch="MysteryForCausalLM")
        self.assertIn("不支援", llm_exl3.assess(arch, self.rt)[1])
        newer = make_model(self.base, "Newer", version="1.7.0")
        self.assertIn("更新 Runtime", llm_exl3.assess(newer, self.rt)[1])
        self.assertEqual(llm_exl3.assess(self.main, "")[0], "error")


class Exl3UiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        patch_sizes(self)
        self.rt = make_runtime(self.base)
        self.main = make_model(self.base, "Qwen3.8-27B-EXL3-3.0bpw", size=12 * GiB)
        self.draft = make_model(self.base, "Qwen3.8-27B-DFlash2-EXL3", arch="DFlash2DraftModel", draft=True)
        self.saved = (gguf_ui.BASE, gguf_ui.EXL3_RUNTIME)
        gguf_ui.BASE, gguf_ui.EXL3_RUNTIME = str(self.base), self.rt

    def tearDown(self):
        gguf_ui.BASE, gguf_ui.EXL3_RUNTIME = self.saved
        self.tmp.cleanup()

    def make_app(self, drafter=""):
        app = gguf_ui.App.__new__(gguf_ui.App)
        app.model_path = lambda: self.main
        app.port, app.ctx = Var("18440"), Var("32768")
        app.ctk, app.ctv, app.think, app.rbud = Var("q4_0"), Var("q4_0"), Var("auto"), Var("512")
        app.mode, app.drafter = Var("dspark"), Var("D")
        app._drafter_map = {"D": drafter}
        app.adv = {k: Var("") for k in ("temp", "topp", "rea_effort", "drym", "lora", "sdnmax")}
        app.adv["vision"] = Var("off")
        app.adv["temp"].set("0.7")
        app.adv["drym"].set("0.8")
        return app

    def test_lists_include_exl3_entries(self):
        models = [m for m in gguf_ui.build_model_list() if m[2] == "exl3"]
        self.assertEqual([os.path.basename(m[1]) for m in models], ["Qwen3.8-27B-EXL3-3.0bpw"])
        vals, mp = gguf_ui.build_drafter_list()
        self.assertIn(self.draft, mp.values())

    def test_build_cmd_dispatches_to_exl3(self):
        app = self.make_app()
        cmd = app.build_cmd()
        self.assertTrue(cmd[0].endswith(os.path.join("python", "python.exe")))
        self.assertIn("server.py", cmd)
        self.assertEqual(cmd[cmd.index("--thinking") + 1], "auto")
        self.assertEqual(cmd[cmd.index("-dm") + 1], self.draft)
        self.assertIn("drym", app._spec_note)
        self.assertIn("思考預算", app._spec_note)
        self.assertFalse(app._forced_fa)

    def test_draft_max_from_ui(self):
        app = self.make_app()
        app.adv["sdnmax"].set("4")
        cmd = app.build_cmd()
        self.assertEqual(cmd[cmd.index("-ndt") + 1], "4")

    def test_widgets_grey_out_and_restore(self):
        import tkinter as tk
        from tkinter import ttk
        try:
            root = tk.Tk()
        except tk.TclError as e:
            self.skipTest(f"no Tk display: {e}")
        root.withdraw()
        self.addCleanup(root.destroy)
        app = self.make_app()
        app._engine_widgets = []
        gguf = str(self.base / "x.gguf")
        cur = {"p": self.main}
        app.model_path = lambda: cur["p"]
        ent = ttk.Entry(root)
        combo = ttk.Combobox(root, state="readonly")
        chk = ttk.Checkbutton(root)
        ndt = ttk.Entry(root)
        vis = ttk.Combobox(root, state="readonly")
        tips = {k: app._engine_reg(k, w) for k, w in
                (("drym", ent), ("runtime", combo), ("fa", chk), ("sdnmax", ndt), ("vision", vis))}
        self.assertEqual(str(vis.cget("state")), "readonly")       # 視覺只對 EXL3 有效
        self.assertEqual(str(ent.cget("state")), "disabled")
        self.assertEqual(str(combo.cget("state")), "disabled")
        self.assertEqual(str(chk.cget("state")), "disabled")
        self.assertEqual(str(ndt.cget("state")), "normal")
        self.assertIn("EXL3 不使用", tips["drym"].extra)
        self.assertIn("block_size", tips["sdnmax"].extra)
        cur["p"] = gguf
        app._apply_engine_ui()
        self.assertEqual(str(ent.cget("state")), "normal")
        self.assertEqual(str(combo.cget("state")), "readonly")
        self.assertEqual(str(chk.cget("state")), "normal")
        self.assertEqual(tips["drym"].extra, "")
        self.assertEqual(tips["sdnmax"].extra, "")
        self.assertEqual(str(vis.cget("state")), "disabled")
        self.assertIn("GGUF 不使用", tips["vision"].extra)
        ent.destroy()
        cur["p"] = self.main
        app._apply_engine_ui()
        self.assertEqual(len(app._engine_widgets), 4)
        self.assertEqual(str(vis.cget("state")), "readonly")
        self.assertEqual(str(combo.cget("state")), "disabled")

    def test_status_text(self):
        app = self.make_app()
        st = app._exl3_status(self.main)
        self.assertIn("✅ EXL3", st["text"])
        self.assertIn("自動配對", st["text"])
        app._drafter_map = {"D": "-"}
        self.assertIn("不使用草稿", app._exl3_status(self.main)["text"])
        app.mode.set("normal")
        self.assertIn("Normal", app._exl3_status(self.main)["text"])
        self.assertNotIn("視覺", app._exl3_status(self.main)["text"])     # 沒有視覺塔的模型不顯示
        cfg = json.loads((pathlib.Path(self.main) / "config.json").read_text(encoding="utf-8"))
        write_json(pathlib.Path(self.main) / "config.json", dict(cfg, vision_config={}))
        write_json(pathlib.Path(self.main) / "preprocessor_config.json", {})
        self.assertIn("視覺：關閉", app._exl3_status(self.main)["text"])
        app.adv["vision"].set("ram")
        self.assertIn("視覺：RAM", app._exl3_status(self.main)["text"])
        self.assertEqual(app.build_cmd()[app.build_cmd().index("--vision") + 1], "ram")


if __name__ == "__main__":
    unittest.main()
