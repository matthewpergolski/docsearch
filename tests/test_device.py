import logging
import sys
import types
import unittest
from unittest import mock

from docsearch.config import pick_device


def fake_torch(cuda: bool, mps: bool):
    t = types.SimpleNamespace()
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    return t


class TestPickDevice(unittest.TestCase):
    def pick(self, pref="auto", env=None, cuda=False, mps=False):
        with mock.patch.dict(sys.modules, {"torch": fake_torch(cuda, mps)}):
            return pick_device(pref, env=env or {})

    def test_order_cuda_then_mps_then_cpu(self):
        self.assertEqual(self.pick(cuda=True, mps=True), "cuda")
        self.assertEqual(self.pick(mps=True), "mps")
        self.assertEqual(self.pick(), "cpu")

    def test_env_override_wins_over_config(self):
        self.assertEqual(self.pick("cuda", env={"DOCSEARCH_DEVICE": "cpu"}, cuda=True), "cpu")
        self.assertEqual(self.pick("cpu", env={"DOCSEARCH_DEVICE": "auto"}, mps=True), "mps")

    def test_config_pin(self):
        self.assertEqual(self.pick("cpu", cuda=True), "cpu")

    def test_bad_value_raises(self):
        with self.assertRaises(ValueError):
            self.pick("tpu")

    def test_logs_choice(self):
        with self.assertLogs("docsearch.device", level=logging.INFO) as cm:
            self.pick(mps=True)
        self.assertIn("mps", "\n".join(cm.output))


class TestOcrDevice(unittest.TestCase):
    def test_pdf_options_carry_device(self):
        from docsearch.ingest import pdf_options

        self.assertEqual(
            str(
                getattr(
                    pdf_options(device="cpu").accelerator_options.device,
                    "value",
                    pdf_options(device="cpu").accelerator_options.device,
                )
            ),
            "cpu",
        )


class TestOcrChildDevice(unittest.TestCase):
    def cmd_for(self, **kw):
        import json
        from pathlib import Path
        from types import SimpleNamespace

        from docsearch.ingest import parse_pdf_isolated

        seen = []

        def run(cmd, capture_output, timeout):
            seen.append(cmd)
            Path(cmd[-2]).write_text(json.dumps({"status": "success", "errors": [], "doc": {}}))
            return SimpleNamespace(returncode=0)

        parse_pdf_isolated(Path("x.pdf"), run=run, **kw)
        return seen[0]

    def test_resolved_device_is_passed_on_the_command_line(self):
        cmd = self.cmd_for(device="mps")
        i = cmd.index("--parse-worker")
        self.assertEqual(cmd[i + 1 : i + 3], ["--device", "mps"])

    def test_auto_is_resolved_before_the_child_starts(self):
        with mock.patch("docsearch.config.pick_device", return_value="cpu"):
            cmd = self.cmd_for(device="auto")
        self.assertEqual(cmd[cmd.index("--device") + 1], "cpu")

    def test_worker_builds_converter_on_that_device(self):
        import docsearch.ingest as ing

        with (
            mock.patch.object(ing, "build_converter", side_effect=RuntimeError("stop")) as bc,
            mock.patch.object(ing, "quiet_crash"),
            self.assertRaises(RuntimeError),
        ):
            ing.parse_worker("x.pdf", "out.json", "full", "cuda")
        self.assertEqual(bc.call_args.kwargs["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
