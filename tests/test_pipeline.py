import ast
import unittest
from pathlib import Path

import yaml

from docsearch import pipeline
from docsearch.config import DEFAULTS, SWITCHES

SRC = Path(__file__).resolve().parents[1] / "src" / "docsearch"
ROOT = SRC.parents[1]
# sync.Ops methods that keep files and the manifest in order; they change no stage in a drawing.
HOUSEKEEPING = {"register", "unregister", "unindex_text", "unindex_pages", "remove_artifacts", "flush_meta"}


def tree(module: str) -> ast.Module:
    return ast.parse((SRC / f"{module}.py").read_text())


def defined(module: str, name: str) -> bool:
    """name is a top-level def/class, or Class.method, in module (read as text: no model imports)."""
    head, _, attr = name.partition(".")
    for node in tree(module).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == head:
            return not attr or any(getattr(n, "name", None) == attr for n in node.body)
    return False


def calls(module: str, func: str) -> set[str]:
    fn = next(n for n in tree(module).body if isinstance(n, ast.FunctionDef) and n.name == func)
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def stage_keys() -> set[str]:
    return {k for flow in pipeline.FLOWS.values() for s in flow for k in s.keys}


class TestStageList(unittest.TestCase):
    def test_every_stage_names_code_that_exists(self):
        for flow in pipeline.FLOWS.values():
            for s in flow:
                module, name = s.ref.split(":")
                with self.subTest(stage=s.id):
                    self.assertTrue(defined(module, name), s.ref)

    def test_every_sync_step_has_a_stage(self):
        used = {c for f in ("apply", "_apply_work") for c in calls("sync", f)}
        ops = {n.name for n in next(c for c in tree("sync").body if getattr(c, "name", "") == "Ops").body
               if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}  # fmt: skip
        steps = {s.step for s in pipeline.INGEST if s.step}
        self.assertEqual((used & ops) - HOUSEKEEPING - steps, set())

    def test_every_rerank_step_has_a_stage(self):
        local = {n.name for n in tree("search").body if isinstance(n, ast.FunctionDef)}
        steps = calls("search", "rerank_pipeline") & local
        refs = {s.ref.split(":")[1] for s in pipeline.QUERY}
        self.assertEqual(steps - refs, set())

    def test_every_config_key_has_a_stage_or_a_reason(self):
        repo = yaml.safe_load((ROOT / "docsearch.yaml").read_text())
        keys = {f"{sec}.{k}" for sec in pipeline.DRAWN_SECTIONS
                for k in {*DEFAULTS[sec], *(repo.get(sec) or {})}}  # fmt: skip
        self.assertEqual(keys - stage_keys() - pipeline.NOT_DRAWN, set())

    def test_every_llm_switch_is_drawn(self):
        self.assertEqual({f"{s}.{k}" for s, k in SWITCHES} - stage_keys(), set())

    def test_stage_ids_are_unique_and_inputs_exist(self):
        for name, flow in pipeline.FLOWS.items():
            ids = [s.id for s in flow]
            self.assertEqual(len(ids), len(set(ids)), name)
            for s in flow:
                self.assertEqual(set(s.after) - set(ids), set(), s.id)


if __name__ == "__main__":
    unittest.main()
