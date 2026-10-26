import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from docsearch.config import load_config

PLAIN_BLOB = {"dpi": 100, "max_side": 1280, "office_max_pages": 4, "visual_k": 100, "rerank_min_chars": 40,
              "rerank_skip_formats": ["slide_image"], "page_reranker": "org/vl", "page_rerank_pages": 10,
              "page_rerank_formats": ["slide_image"], "sheet_cap": "shared"}  # fmt: skip


def write_yaml(d: str, text: str) -> Path:
    p = Path(d) / "docsearch.yaml"
    p.write_text(text)
    return p


class TestSwitches(unittest.TestCase):
    def test_defaults_are_off(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = load_config(Path(t) / "none.yaml", env={})
        self.assertEqual((cfg["enrich"]["llm"], cfg["agent"]["mode"]), ("off", "off"))

    def test_yaml_and_env_values(self):
        with tempfile.TemporaryDirectory() as t:
            y = write_yaml(t, "enrich: {llm: api}\nagent: {mode: on}\n")
            self.assertEqual(load_config(y, env={})["agent"]["mode"], "on")
            cfg = load_config(y, env={"DOCSEARCH_ENRICH_LLM": "local", "DOCSEARCH_AGENT_MODE": "off"})
        self.assertEqual((cfg["enrich"]["llm"], cfg["agent"]["mode"]), ("local", "off"))

    def test_bad_value_names_key_value_and_choices(self):
        with tempfile.TemporaryDirectory() as t:
            y = write_yaml(t, "enrich: {llm: gpt}\n")
            with self.assertRaises(ValueError) as cm:
                load_config(y, env={})
            self.assertEqual(str(cm.exception), "enrich.llm must be one of off, local, api; got 'gpt'")
            with self.assertRaises(ValueError) as cm:
                load_config(Path(t) / "none.yaml", env={"DOCSEARCH_AGENT_MODE": "maybe"})
            self.assertEqual(str(cm.exception), "agent.mode must be one of off, on; got 'maybe'")


class TestEvalMode(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.yaml = write_yaml(self.td.name, "enrich: {llm: api}\nagent: {mode: on}\n")
        self.calls = []

    def tearDown(self):
        self.td.cleanup()

    def eval_cli(self, *args, env=None):
        from docsearch import cli

        def fake_eval(cfg, plan, *a, **k):
            self.calls.append({"enrich.llm": cfg["enrich"]["llm"], "agent.mode": cfg["agent"]["mode"],
                               "mode": k.get("eval_mode")})  # fmt: skip
            return []

        with mock.patch.dict(os.environ, env or {}), mock.patch("docsearch.eval.run_eval", fake_eval), \
                mock.patch("docsearch.report.build_report"), mock.patch("docsearch.cli._device_line", lambda c: ""), \
                mock.patch("docsearch.cli._quiet_hub"), \
                mock.patch("docsearch.cli.timed", lambda *a, **k: contextlib.nullcontext()), \
                contextlib.redirect_stdout(io.StringIO()):  # fmt: skip
            return cli.main(["--config", str(self.yaml), "eval", "--full", *args])

    def test_plain_forces_switches_off(self):
        self.eval_cli("--mode", "plain", env={"DOCSEARCH_ENRICH_LLM": "local"})
        self.eval_cli()
        want = {"enrich.llm": "off", "agent.mode": "off", "mode": "plain"}
        self.assertEqual(self.calls, [want, want])

    def test_unbuilt_modes_stop_before_any_work(self):
        with self.assertRaises(SystemExit) as cm:
            self.eval_cli("--mode", "enrich")
        self.assertEqual(str(cm.exception), "eval --mode enrich: the enrich layer is not built yet (a later phase "
                                            "builds it); nothing was run")  # fmt: skip
        self.assertEqual(self.calls, [])
        self.assertFalse((Path(self.td.name) / "logs").exists())

    def test_unknown_mode(self):
        with self.assertRaises(SystemExit) as cm:
            self.eval_cli("--mode", "fancy")
        self.assertEqual(str(cm.exception), "--mode must be one of plain, enrich, agent; got 'fancy'")

    def test_built_enrich_keeps_the_chosen_llm_and_needs_one(self):
        from docsearch.eval import MODES, Mode, apply_mode

        with mock.patch.dict(MODES, {"enrich": Mode(MODES["enrich"].switches, None)}):
            cfg = load_config(self.yaml, env={})
            apply_mode(cfg, "enrich")
            self.assertEqual((cfg["enrich"]["llm"], cfg["agent"]["mode"]), ("api", "on"))
            cfg = load_config(Path(self.td.name) / "none.yaml", env={})
            with self.assertRaises(SystemExit) as cm:
                apply_mode(cfg, "enrich")
        self.assertEqual(str(cm.exception), "eval --mode enrich needs enrich.llm set to local or api (it is off)")


class TestModeInRunSettings(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        y = write_yaml(self.td.name, "search: {rerank_candidates: 30, rerank_min_chars: 40, rerank_skip_formats: "
                       "[slide_image], page_reranker: org/vl, page_rerank_formats: [slide_image]}\n")  # fmt: skip
        self.cfg = load_config(y, env={})

    def tearDown(self):
        self.td.cleanup()

    def test_plain_blob_is_unchanged(self):
        from docsearch.eval import run_settings, setting_blob
        from docsearch.search import SearchConfig

        self.assertEqual(run_settings(self.cfg)["blob"], PLAIN_BLOB)
        self.assertEqual(setting_blob(self.cfg, SearchConfig.from_cfg(self.cfg), "org/vl", "plain"), PLAIN_BLOB)

    def test_other_modes_are_named_in_the_blob(self):
        from docsearch.eval import setting_blob
        from docsearch.search import SearchConfig

        blob = setting_blob(self.cfg, SearchConfig.from_cfg(self.cfg), "org/vl", "enrich")
        self.assertEqual(blob, {**PLAIN_BLOB, "mode": "enrich"})


if __name__ == "__main__":
    unittest.main()


class TestEvalRunNames(unittest.TestCase):
    """--name names the runs of a plain `--visual --fusion format_aware` eval, on any question set."""

    def names(self, *args):
        from docsearch import cli

        plans = []

        def fake_eval(cfg, plan, *a, **k):
            plans.append([p[0] for p in plan])
            return []

        with tempfile.TemporaryDirectory() as t, mock.patch("docsearch.eval.run_eval", fake_eval), \
                mock.patch("docsearch.report.build_report"), mock.patch("docsearch.cli._device_line", lambda c: ""), \
                mock.patch("docsearch.cli._quiet_hub"), \
                mock.patch("docsearch.cli.timed", lambda *a, **k: contextlib.nullcontext()), \
                contextlib.redirect_stdout(io.StringIO()):  # fmt: skip
            cli.main(["--config", str(write_yaml(t, "")), "eval", "--full", "--visual", "pplx-v2-late",
                      "--fusion", "format_aware", *args])  # fmt: skip
        return plans

    def test_custom_questions_take_the_name(self):
        got = self.names("--questions", "custom", "--reviewed-only", "--name", "j0_custom")
        self.assertEqual(got, [["j0_custom", "j0_custom_norerank"]])

    def test_public_questions_take_the_name(self):
        self.assertEqual(self.names("--name", "x1"), [["x1", "x1_norerank"]])

    def test_no_name_keeps_the_default_names(self):
        got = self.names("--suffix", "_b")
        self.assertEqual(got, [["hybrid_visual_fa_pplx-v2-late_rerank_b", "hybrid_visual_fa_pplx-v2-late_norerank_b"]])
