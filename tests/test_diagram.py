import re
import tempfile
import unittest
from pathlib import Path

from docsearch import diagram
from docsearch.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def cfg_from(text: str):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "docsearch.yaml"
        p.write_text(text)
        return load_config(p, env={})


class TestCommittedDiagrams(unittest.TestCase):
    def test_markdown_diagrams_match_the_code(self):
        cfg = load_config(ROOT / "docsearch.yaml", env={})  # the committed config, never the local env
        for name in diagram.DOCS:
            with self.subTest(file=name):
                self.assertIn("<!-- diagram:", (ROOT / name).read_text(), f"{name} has no diagram markers")
        self.assertEqual(diagram.stale(ROOT, cfg), [], "run `uv run docsearch diagram --write` and commit")


class TestDrawing(unittest.TestCase):
    def test_off_stage_is_dashed(self):
        m = diagram.mermaid(cfg_from("visual: {default: null}\n"), "ingest")
        self.assertIn("ingest_parse -.-> ingest_render", m)
        self.assertRegex(m, r"class [\w,]*\bingest_render\b[\w,]* off")

    def test_on_stage_is_solid(self):
        m = diagram.mermaid(cfg_from("visual: {default: pplx-v2-late}\nenrich: {llm: api}\n"), "ingest")
        self.assertIn("ingest_chunk --> ingest_enrich", m)
        self.assertNotRegex(m, r"class [\w,]*\bingest_enrich\b[\w,]* off")

    def test_off_experiments_share_one_box_and_on_ones_get_their_own(self):
        off = diagram.mermaid(cfg_from(""), "query").replace("<br/>", " ")
        self.assertIn("query_fuse --> query_rerank", off)
        self.assertIn('query_exp["Experiments, off: period demote', off)
        on = diagram.mermaid(cfg_from("search: {period_demote: true}\n"), "query").replace("<br/>", " ")
        self.assertIn("query_fuse --> query_period", on)
        self.assertIn("query_period --> query_rerank", on)
        self.assertNotIn("period demote,", on)

    CONFIGS = ("", 'agent: {mode: "on"}\nenrich: {llm: api}\nvisual: {default: null}\n')

    def drawings(self):
        for text in self.CONFIGS:
            for flow in diagram.FLOWS:
                yield flow, diagram.mermaid(cfg_from(text), flow), diagram.block(cfg_from(text), flow)

    def test_no_label_line_is_longer_than_32_characters(self):
        # GitHub on a phone draws a long line outside its box
        for flow, m, b in self.drawings():
            for label in re.findall(r'\["(.*?)"\]|\[\("(.*?)"\)\]', m + b):
                for line in re.sub(r"</?small>", "", "".join(label)).split("<br/>"):
                    with self.subTest(flow=flow, line=line):
                        self.assertLessEqual(len(line.replace("#quot;", '"')), 32)

    def test_no_class_uses_opacity(self):
        # Safari (WebKit) paints the label of a node with opacity outside its box, top left of the drawing
        for flow, m, _ in self.drawings():
            for line in m.splitlines():
                if line.lstrip().startswith("classDef"):
                    with self.subTest(flow=flow, line=line):
                        self.assertNotIn("opacity", line)

    def test_each_node_has_at_most_one_class(self):
        for flow, m, _ in self.drawings():
            ids = [i for c in re.findall(r"^  class (\S+) \w+$", m, re.M) for i in c.split(",")]
            with self.subTest(flow=flow):
                self.assertEqual(len(ids), len(set(ids)), m)

    def test_off_llm_stage_is_dashed_and_keeps_its_layer_color(self):
        m = diagram.mermaid(cfg_from(""), "ingest")
        self.assertRegex(m, r"class [\w,]*\bingest_enrich\b[\w,]* off_enrich\n")
        self.assertRegex(m, r"classDef off_enrich stroke:#d97706,[^\n]*stroke-dasharray")

    def test_splice_changes_only_the_marked_text(self):
        text = "before\n<!-- diagram:eval -->\nold\n<!-- /diagram -->\nafter\n"
        out = diagram.splice(text, cfg_from(""))
        self.assertTrue(out.startswith("before\n<!-- diagram:eval -->\n```mermaid\nflowchart LR"))
        self.assertTrue(out.endswith("<!-- /diagram -->\nafter\n"))
        self.assertEqual(diagram.splice(out, cfg_from("")), out)

    def test_unknown_diagram_name_is_an_error(self):
        with self.assertRaises(ValueError):
            diagram.splice("<!-- diagram:nope -->\n<!-- /diagram -->", cfg_from(""))


class TestPublicDrawings(unittest.TestCase):
    """The drawings in the docs are public: they name the settings that pick an LLM, never the LLM itself."""

    OWN = 'agent: {mode: "on", provider: acmeprov, model: bigmodel-9}\nenrich: {llm: api}\n'

    def test_ingest_is_drawn_top_to_bottom(self):
        self.assertTrue(diagram.mermaid(cfg_from(""), "ingest").startswith("flowchart TB\n"))

    def test_doc_blocks_name_the_llm_settings_not_the_model(self):
        cfg = cfg_from(self.OWN)
        for flow in diagram.FLOWS:
            with self.subTest(flow=flow):
                text = diagram.block(cfg, flow)
                self.assertNotIn("acmeprov", text)
                self.assertNotIn("bigmodel-9", text)
        self.assertIn("agent.provider, agent.model", diagram.block(cfg, "query").replace("<br/>", " "))
        self.assertIn("enrich.provider, enrich.model", diagram.block(cfg, "ingest").replace("<br/>", " "))

    def test_own_drawing_names_the_configured_model(self):
        self.assertIn("acmeprov bigmodel-9", diagram.mermaid(cfg_from(self.OWN), "query"))

    def test_committed_docs_name_no_llm_model(self):
        cfg = load_config(ROOT / "docsearch.yaml", env={})
        names = {cfg[s].get(k) for s in ("agent", "enrich") for k in ("provider", "model")} - {None, ""}
        for name in diagram.DOCS:
            for m in diagram.MARKER.finditer((ROOT / name).read_text()):
                for word in names:
                    with self.subTest(file=name, flow=m.group(2), word=word):
                        self.assertNotRegex(m.group(3).lower(), rf"\b{word.lower()}\b")


if __name__ == "__main__":
    unittest.main()
