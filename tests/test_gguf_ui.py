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


class ExtraArgsSpecTests(unittest.TestCase):
    """自訂指令（adv.extra_args）與投機旗標的互動規則（build_cmd 產生的指令）。"""

    NO_DF = "不使用外掛 DF（＝用主模型內建 MTP）"

    def make_app(self, mode="dspark", extra="", builtin_mtp=0):
        app = gguf_ui.App.__new__(gguf_ui.App)
        app._mp = os.path.abspath(__file__)          # 用測試檔本身當「存在的模型檔」
        app.model_path = lambda: app._mp
        app.server_exe = lambda: r"D:\ExampleProject\RUNTIMES\x\llama-server.exe"
        app._drafters = ["自動（依主模型配對）", self.NO_DF]
        app._drafter_map = {app._drafters[0]: "", self.NO_DF: "-"}
        app.drafter = Var(self.NO_DF)
        app.mode = Var(mode)
        app.port = Var("18499")
        app.ctx = Var("65536")
        app.ngl = Var("all")
        app.ngld = Var("all")
        app.ctk = Var("q4_0")
        app.ctv = Var("q4_0")
        app.rbud = Var("-1")
        app.fa = Var(True)
        app.nkvo = Var(False)
        app.think = Var("auto")
        app.jinja = Var(True)
        app.runtime = Var("llama-b11642-rocm-sdk-10.1.0-x64/")
        keys = {"rea_effort", "extra_args", "lora", "lora_scale"}
        for group in (gguf_ui.ADV_CORE, gguf_ui.ADV_OPT, gguf_ui.ADV_MOE,
                      gguf_ui.ADV_LOAD, gguf_ui.ADV_LORA, gguf_ui.ADV_SPEC):
            for row in group:
                keys.add(row[1])
        app.adv = {k: Var("") for k in keys}
        app.adv["rea_effort"].set("medium")
        app.adv["extra_args"].set(extra)
        app._builtin_mtp = builtin_mtp
        return app

    def build(self, extra, builtin_mtp=0, mode="dspark"):
        app = self.make_app(mode=mode, extra=extra, builtin_mtp=builtin_mtp)
        with mock.patch.object(gguf_ui, "gguf_mtp_layers", lambda _p: builtin_mtp):
            cmd = app.build_cmd()
        return cmd, app._spec_note

    @staticmethod
    def spec_values(cmd):
        return [cmd[i + 1] for i, t in enumerate(cmd) if t == "--spec-type"]

    def test_external_mtp_draft_is_kept_on_model_without_builtin_mtp(self):
        draft = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mtp-gemma-4-12B-it.gguf")
        pathlib.Path(draft).write_bytes(b"stub")
        try:
            cmd, note = self.build(f'-md "{draft}" --spec-type draft-mtp')
        finally:
            pathlib.Path(draft).unlink()

        self.assertEqual(self.spec_values(cmd), ["draft-mtp"])   # 沒有被 UI 靜靜關掉
        self.assertIn("-md", cmd)
        self.assertIn("外掛 MTP 草稿", note)
        self.assertNotIn("已移除", note)

    def test_draft_mtp_without_draft_file_is_removed(self):
        cmd, note = self.build("--spec-type draft-mtp")
        self.assertEqual(self.spec_values(cmd), [])
        self.assertIn("已移除", note)

    def test_equals_form_without_draft_file_is_removed(self):
        cmd, note = self.build("--spec-type=draft-mtp")
        self.assertEqual(self.spec_values(cmd), [])
        self.assertIn("已移除", note)

    def test_builtin_mtp_keeps_extra_draft_mtp(self):
        cmd, note = self.build("--spec-type draft-mtp", builtin_mtp=1)
        self.assertEqual(self.spec_values(cmd), ["draft-mtp"])
        self.assertNotIn("已移除", note)

    def test_extra_spec_replaces_ui_own_spec_block(self):
        cmd, note = self.build("-md x.gguf --spec-type draft-dflash", builtin_mtp=1)
        self.assertEqual(self.spec_values(cmd), ["draft-dflash"])   # 只有自訂指令那一份
        self.assertIn("以自訂指令為準", note)

    def test_missing_extra_draft_file_is_reported(self):
        cmd, note = self.build(r"-md D:\nope\missing-mtp.gguf --spec-type draft-mtp")
        self.assertIn("找不到這個草稿檔", note)
        self.assertEqual(self.spec_values(cmd), ["draft-mtp"])


if __name__ == "__main__":
    unittest.main()
