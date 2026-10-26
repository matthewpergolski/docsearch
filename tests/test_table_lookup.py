import json
import sqlite3
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from docsearch.cells import SCHEMA, CellLookup, grid_cells, page_scores, query_values
from docsearch.config import DEFAULTS
from docsearch.search import SearchConfig, rerank_pipeline, table_lookup


def annex(country, young_2022):
    return [[country, "2019", "2022", "2023"],
            ["4 - Activity rate", "78.5", "81.2", "81.6"],
            ["Young (15-24)", "43.5", "44.9", "43.7"],
            ["12 - Unemployment rate", "5.4", "5.6", "6.0"],
            ["Young (15-24)", "11.1", young_2022, "33.4"]]  # fmt: skip


def page_cells(page, grid, table="t"):
    return [{**c, "page": page, "table_id": f"{table}{page}"} for c in grid_cells(grid)]


Q = "What 2023 rate does the annex give for 15-24-year-olds whose 2022 rate was 32.7?"


class TestQueryValues(unittest.TestCase):
    def test_decimals_and_large_integers_not_years_or_small_counts(self):
        self.assertEqual(query_values(Q), {32.7})
        self.assertEqual(query_values("1,320 people in 2019, 14 regions, 4.5% and 250 sites"), {1320.0, 4.5, 250.0})


class TestPageScores(unittest.TestCase):
    def test_value_in_a_labelled_row_under_a_named_year_wins(self):
        cells = page_cells(110, annex("Estonia", "30.1")) + page_cells(115, annex("Latvia", "32.7"))
        # p115: value 32.7 (2) + label 15, 24 (2) + header 2022 (1); p110: label + header only (3)
        self.assertEqual(page_scores(Q, cells), {110: 3, 115: 5})

    def test_entity_named_row_and_header(self):
        cells = page_cells(3, [["Province", "Enrolled 2017", "Wrote 2017"], ["Western Cape", "14 015", "12 000"],
                               ["Gauteng", "20 100", "19 000"]])  # fmt: skip
        self.assertEqual(page_scores("How many candidates enrolled in the Western Cape during 2017?", cells), {3: 4})

    def test_no_label_and_no_value_scores_nothing(self):
        self.assertEqual(page_scores("Who chairs the board?", page_cells(1, annex("Estonia", "32.7"))), {})


def ranked(*items):
    return [{"chunk_id": f"{d}{p}_{i}", "doc_id": d, "format": "pdf", "pages": [p] if p else None}
            for i, (d, p) in enumerate(items)]  # fmt: skip


class Fake:
    def __init__(self, scores):
        self.s, self.calls = scores, []

    def scores(self, query, doc, pages):
        self.calls.append((doc, list(pages)))
        return self.s.get(doc, {})


class TestTableLookup(unittest.TestCase):
    R = ranked(("A", 1), ("B", 7), ("A", 2), ("A", 1), ("B", 8), ("A", 3), ("C", None))

    def ids(self, out):
        return [c["chunk_id"] for c in out]

    def test_best_page_takes_its_files_first_slots_others_keep_theirs(self):
        out = table_lookup(Q, self.R, Fake({"A": {3: 5, 1: 2}}))
        self.assertEqual(self.ids(out), ["A3_5", "B7_1", "A1_0", "A2_2", "B8_4", "A1_3", "CNone_6"])

    def test_ties_and_no_match_keep_the_order(self):
        for s in ({"A": {1: 3, 3: 3}}, {"A": {}}, {"B": {7: 0, 8: 0}}):
            self.assertEqual(self.ids(table_lookup(Q, self.R, Fake(s))), self.ids(self.R))

    def test_files_with_one_page_are_not_looked_up(self):
        fake = Fake({})
        table_lookup(Q, self.R, fake)
        self.assertEqual(fake.calls, [("A", [1, 2, 3]), ("B", [7, 8])])
        self.assertIs(table_lookup(Q, self.R, None), self.R)


class Keep:
    calls = 0

    def score(self, q, texts):
        return [-i for i in range(len(texts))]


class TestPipelineFlag(unittest.TestCase):
    def test_off_by_default_and_on_with_the_key(self):
        self.assertIs(DEFAULTS["search"]["table_lookup"], False)
        cands = [{**c, "text": "x" * 50} for c in TestTableLookup.R]
        fake = Fake({"A": {3: 5}})
        off = SearchConfig(10, 10, 60, 30, 10)
        self.assertEqual([c["chunk_id"] for c in rerank_pipeline(Q, cands, Keep(), off, None, fake)],
                         [c["chunk_id"] for c in cands])  # fmt: skip
        self.assertEqual(fake.calls, [])
        on = SearchConfig(10, 10, 60, 30, 10, table_lookup=True)
        self.assertEqual(rerank_pipeline(Q, cands, Keep(), on, None, fake)[0]["chunk_id"], "A3_5")


class TestCellLookup(unittest.TestCase):
    def test_reads_the_store_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cells.sqlite"
            db = sqlite3.connect(path)
            db.executescript(SCHEMA)
            for page, cc in ((110, "30.1"), (115, "32.7")):
                db.executemany("INSERT INTO cells VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                               [(f"t{page}", "annex", "pdf", page, None, c["row"], c["col"], None, c["text"], c["num"],
                                 c["row_label"], c["col_header"], c["section"], int(c["fused"]))
                                for c in grid_cells(annex("X", cc))])  # fmt: skip
            db.commit()
            db.close()
            lookup = CellLookup(path)
            self.assertEqual(lookup.scores(Q, "annex", [110, 115, 9]), {110: 3, 115: 5})
            with self.assertRaises(sqlite3.OperationalError):
                lookup.db.execute("DELETE FROM cells")

    def test_missing_store_stops_with_the_build_command(self):
        with self.assertRaises(SystemExit) as cm:
            CellLookup(Path("/nonexistent/cells.sqlite"))
        self.assertIn("scripts/build_cells.py", str(cm.exception))


class TestEvalWiring(unittest.TestCase):
    def cfg(self, td, search=""):
        from docsearch.config import load_config

        p = Path(td) / "docsearch.yaml"
        p.write_text("paths: {data_dir: data}\n" + search)
        return load_config(p, env={})

    def test_blob_names_the_lookup_only_when_on(self):
        from docsearch.eval import setting_blob

        with tempfile.TemporaryDirectory() as td:
            off, on = self.cfg(td), self.cfg(td, "search: {table_lookup: true}\n")
        self.assertNotIn("table_lookup", setting_blob(off, SearchConfig.from_cfg(off), ""))
        self.assertIs(setting_blob(on, SearchConfig.from_cfg(on), "")["table_lookup"], True)

    def test_table_questions_load_like_hard_and_a_missing_file_stops(self):
        from docsearch.eval import load_custom

        with tempfile.TemporaryDirectory() as td:
            cfg = self.cfg(td)
            with self.assertRaises(SystemExit) as cm:
                load_custom(cfg, True, "tables")
            self.assertIn("no table questions at", str(cm.exception))
            self.assertIn("questions_tables.jsonl", str(cm.exception))
            (Path(td) / "data").mkdir()
            row = {"qid": "tables:0", "query": "q?", "source": "ocb", "format": "xlsx", "level": "page",
                   "gold": {"ocb/a.xlsx#p2": 1}, "meta": {"type": "table_value", "status": "keep"}}  # fmt: skip
            (Path(td) / "data" / "questions_tables.jsonl").write_text(json.dumps(row) + "\n")
            self.assertEqual([q.qid for q in load_custom(cfg, True, "tables")], ["tables:0"])

    def test_cli_flag_is_a_one_change_override(self):
        from types import SimpleNamespace

        from docsearch.cli import _one_change_plan

        with tempfile.TemporaryDirectory() as td:
            cfg = self.cfg(td, "models: {embedders: {v: {id: org/v}}}\n")
            a = SimpleNamespace(visual="v", fusion="format_aware", table_lookup=True, name="m33", variant="hybrid",
                                full=True, rerank_skip_formats=None, visual_blind_formats=None)  # fmt: skip
            with unittest.mock.patch("docsearch.eval.cached_reference", return_value={"name": "ref", "run_id": "r",
                                                                                      "ndcg@10": "0.8", "n_queries": 1}):  # fmt: skip
                plan, ref, over = _one_change_plan(cfg, a)
        self.assertEqual((plan, ref, over), ([("m33", "hybrid", "hybrid_visual_fa", True)], None,
                                             {("search", "table_lookup"): True}))  # fmt: skip


class TestResearchSearchesKeepTheStats(unittest.TestCase):
    def test_rewrite_searches_do_not_count_in_the_question_stats(self):
        from types import SimpleNamespace

        import numpy as np

        from docsearch.eval import _plain_search
        from docsearch.search import NUMBER_BOOST_STATS, TABLE_LOOKUP_STATS

        cands = [{**c, "text": "x" * 50} for c in TestTableLookup.R]
        sc = SearchConfig(10, 10, 60, 30, 10, table_lookup=True, number_boost=1.0)
        r = SimpleNamespace(sc=sc, candidates=lambda t, qv, mode, vq: list(cands))
        NUMBER_BOOST_STATS.clear()
        TABLE_LOOKUP_STATS.clear()
        NUMBER_BOOST_STATS.update({"queries": 7})
        TABLE_LOOKUP_STATS.update({"queries": 7, "files": 9, "moved": 2})
        with unittest.mock.patch("docsearch.eval.query_vectors", return_value=np.zeros((2, 4), np.float32)):
            out = _plain_search(None, r, "hybrid", None, Keep(), None, Fake({"A": {3: 5}}))(["rw 32.7", "rw two"])
        self.assertEqual([h[0]["chunk_id"] for h in out], ["A3_5", "A3_5"])
        self.assertEqual((dict(NUMBER_BOOST_STATS), dict(TABLE_LOOKUP_STATS)),
                         ({"queries": 7}, {"queries": 7, "files": 9, "moved": 2}))  # fmt: skip


if __name__ == "__main__":
    unittest.main()
