import importlib.util
import os
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gguf_ui", ROOT / "gguf-ui.py")
gguf_ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gguf_ui)


class Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class TemplateStateTests(unittest.TestCase):
    def make_app(self, old_path, new_path):
        app = gguf_ui.App.__new__(gguf_ui.App)
        app._loading = False
        app._tpl_busy = False
        app._rt_busy = False
        app._tpl_model_key = os.path.normcase(os.path.abspath(old_path))
        app._path = new_path
        app.model_path = lambda: app._path
        app._presets = {"templates": {}, "bind": {}, "default": "", "runtime_of": {}}
        app.adv = {"lora": Var("old-lora.gguf"), "lora_scale": Var("0.25")}
        app.tpl = Var("Bonsai")
        app.runtime = Var("llama-official/")
        app._lora_applied = ("old-lora.gguf", 0.25)
        app._lora_try_ts = 123
        app._log = lambda _text: None
        app._tpl_default_name = lambda: ""
        app._rt_restore = lambda _path: True
        return app

    def test_switch_to_unbound_model_clears_template_lora(self):
        app = self.make_app(r"D:\\ExampleProject\\Bonsai.gguf", r"D:\\ExampleProject\\MiMo.gguf")

        app._tpl_on_model_change()

        self.assertEqual(app.adv["lora"].get(), "")
        self.assertEqual(app.adv["lora_scale"].get(), "")
        self.assertIsNone(app._lora_applied)
        self.assertEqual(app._lora_try_ts, 0)

    def test_same_model_rescan_keeps_manual_lora(self):
        path = r"D:\\ExampleProject\\MiMo.gguf"
        app = self.make_app(path, path)

        app._tpl_on_model_change()

        self.assertEqual(app.adv["lora"].get(), "old-lora.gguf")
        self.assertEqual(app.adv["lora_scale"].get(), "0.25")

    def test_target_template_restores_its_own_lora(self):
        app = self.make_app(r"D:\\ExampleProject\\Bonsai.gguf", r"D:\\ExampleProject\\Other.gguf")
        key = os.path.normcase(app._path)
        app._presets = {
            "templates": {"Other": {"adv": {"lora": "other.gguf", "lora_scale": "1.5"}}},
            "bind": {key: "Other"}, "default": "", "runtime_of": {},
        }
        app._tpl_apply = lambda tpl: [app.adv[k].set(tpl["adv"].get(k, "")) for k in app.adv]

        app._tpl_on_model_change()

        self.assertEqual(app.adv["lora"].get(), "other.gguf")
        self.assertEqual(app.adv["lora_scale"].get(), "1.5")

    def test_picking_no_template_clears_lora(self):
        path = r"D:\\ExampleProject\\MiMo.gguf"
        app = self.make_app(path, path)
        app._TPL_NONE = "（不使用模板）"
        app.tpl.set(app._TPL_NONE)

        app._tpl_on_pick()

        self.assertEqual(app.adv["lora"].get(), "")
        self.assertEqual(app.adv["lora_scale"].get(), "")


class PortValidationTests(unittest.TestCase):
    def test_valid_port(self):
        self.assertEqual(gguf_ui.parse_port("18435"), 18435)

    def test_rejects_out_of_range_and_non_numeric_ports(self):
        for value in ("", "abc", "0", "-1", "65536"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gguf_ui.parse_port(value)


class AtomicJsonTests(unittest.TestCase):
    def test_replace_failure_preserves_original_file(self):
        with tempfile.TemporaryDirectory() as td:
            target = pathlib.Path(td) / "settings.json"
            target.write_text('{"old": true}\n', encoding="utf-8")
            with mock.patch.object(gguf_ui.os, "replace", side_effect=OSError("disk failure")):
                ok = gguf_ui.atomic_write_json(str(target), {"new": True})

            self.assertFalse(ok)
            self.assertEqual(target.read_text(encoding="utf-8"), '{"old": true}\n')
            self.assertEqual(list(pathlib.Path(td).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
