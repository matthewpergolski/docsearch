"""sync: plan (new/changed/deleted/unchanged by hash), apply with a fake Ops, resume, dry run, gate."""

import ast
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from docsearch import sync as S
from docsearch.config import load_config


class FakeOps:
    def __init__(self, fail_parse=(), crash_chunk=()):
        self.calls: list[tuple] = []
        self.fail_parse, self.crash_chunk = set(fail_parse), set(crash_chunk)
        self.version = 1
        self.buffered = 0

    def register(self, docs):
        self.calls.append(("register", sorted(d.doc_id for d in docs)))

    def parse(self, d):
        self.calls.append(("parse", d.doc_id))
        return {"status": "failed", "error": "boom"} if d.doc_id in self.fail_parse else {"status": "ok"}

    def chunk(self, d):
        self.calls.append(("chunk", d.doc_id))
        if d.doc_id in self.crash_chunk:
            self.crash_chunk.discard(d.doc_id)
            raise RuntimeError("crash mid-sync")
        return 2

    def unindex_text(self, ids):
        self.calls.append(("unindex_text", sorted(ids)))

    def unindex_pages(self, ids):
        self.calls.append(("unindex_pages", sorted(ids)))

    def render(self, d):
        self.calls.append(("render", d.doc_id))
        return 1

    def meta(self, d):
        self.calls.append(("meta", d.doc_id))
        self.buffered += 1

    def flush_meta(self):  # as Ops: writes only when it holds rows
        if self.buffered:
            self.calls.append(("flush_meta",))
            self.buffered = 0

    def remove_artifacts(self, ids):
        self.calls.append(("remove", sorted(ids)))

    def index(self):
        self.calls.append(("index",))
        self.version += 1
        return {"hybrid": 1}

    def candidate(self):
        return {"tables": {"chunks_hybrid": self.version}, "visual": None}

    def did(self, op):
        return [c[1] for c in self.calls if c[0] == op]


class Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        (self.root / "docsearch.yaml").write_text("subset: {enabled: false}\nvisual: {default: null}\n")
        self.cfg = load_config(self.root / "docsearch.yaml", env={})
        self.inbox = self.root / "inbox"
        (self.inbox / "sub").mkdir(parents=True)
        self.write("a.csv", "x,y\n1,2\n")
        self.write("sub/b.csv", "p,q\n3,4\n")
        self.write("c.png", "not really a png")
        self.write("notes.md", "ignored extension")

    def tearDown(self):
        self.td.cleanup()

    def write(self, rel, text):
        p = self.inbox / rel
        p.write_text(text)
        return p

    def run_sync(self, ops=None, **kw):
        return S.sync(self.cfg, self.inbox, ops=ops if ops is not None else FakeOps(), **kw)

    def state(self):
        st = S.State(self.cfg.path("sync_manifest"))
        try:
            return {Path(p).name: r for p, r in st.rows().items()}
        finally:
            st.close()


class TestPlanAndApply(Base):
    def test_add_then_unchanged(self):
        ops = FakeOps()
        r = self.run_sync(ops)
        self.assertEqual((r["counts"]["new"], r["counts"]["ignored"]), (3, 1))
        self.assertEqual(sorted(ops.did("parse")), ["inbox/a.csv", "inbox/c.png", "inbox/sub/b.csv"])
        self.assertEqual(sum(c[0] == "index" for c in ops.calls), 1)
        st = self.state()
        self.assertEqual({r["status"] for r in st.values()}, {"ok"})
        a = st["a.csv"]
        self.assertEqual(len(a["sha256"]), 64)
        self.assertTrue(a["parsed_at"] and a["size"] == 8 and a["doc_id"] == "inbox/a.csv")
        ops2 = FakeOps()
        r2 = self.run_sync(ops2)
        self.assertEqual(r2["counts"]["unchanged"], 3)
        self.assertEqual(ops2.calls, [])  # nothing parsed, chunked or indexed

    def test_metadata_row_written_for_each_parsed_file(self):
        ops = FakeOps(fail_parse={"inbox/a.csv"})
        self.run_sync(ops)
        self.assertEqual(sorted(ops.did("meta")), ["inbox/c.png", "inbox/sub/b.csv"])
        self.assertEqual(ops.calls.count(("flush_meta",)), 1)

    def test_a_sync_that_stops_halfway_writes_the_metadata_it_has(self):
        ops = FakeOps(crash_chunk={"inbox/c.png"})
        with self.assertRaises(RuntimeError):
            self.run_sync(ops)
        self.assertEqual([c for c in ops.calls if c[0] in ("meta", "flush_meta")],
                         [("meta", "inbox/a.csv"), ("meta", "inbox/c.png"), ("flush_meta",)])  # fmt: skip

    def test_modified_same_name_is_reparsed_touch_is_not(self):
        self.run_sync()
        p = self.inbox / "a.csv"
        later = time.time() + 100
        os.utime(p, (later, later))  # mtime change only: same hash -> unchanged
        ops = FakeOps()
        r = self.run_sync(ops)
        self.assertEqual((r["counts"]["changed"], r["counts"]["unchanged"]), (0, 3))
        self.write("a.csv", "x,y\n1,999\n")  # same name, new content
        ops = FakeOps()
        r = self.run_sync(ops)
        self.assertEqual((r["counts"]["changed"], r["counts"]["unchanged"]), (1, 2))
        self.assertEqual(ops.did("parse"), ["inbox/a.csv"])
        self.assertIn(("unindex_text", ["inbox/a.csv"]), ops.calls)  # old rows removed before re-chunk
        self.assertIn(("unindex_pages", ["inbox/a.csv"]), ops.calls)
        self.assertEqual(sum(c[0] == "index" for c in ops.calls), 1)

    def test_delete_removes_everything(self):
        self.run_sync()
        (self.inbox / "sub" / "b.csv").unlink()
        ops = FakeOps()
        r = self.run_sync(ops)
        self.assertEqual((r["counts"]["deleted"], r["counts"]["unchanged"]), (1, 2))
        for op in ("unindex_text", "unindex_pages", "remove"):
            self.assertIn((op, ["inbox/sub/b.csv"]), ops.calls)
        self.assertNotIn("b.csv", self.state())
        self.assertEqual(self.run_sync(FakeOps())["counts"]["deleted"], 0)  # idempotent

    def test_dry_run_changes_nothing(self):
        ops = FakeOps()
        r = self.run_sync(ops, dry_run=True)
        self.assertEqual(r["counts"]["new"], 3)
        self.assertEqual(ops.calls, [])
        self.assertEqual(self.state(), {})
        self.assertFalse((self.cfg.path("corpus_dir") / "manifest.csv").exists())

    def test_resume_after_crash_skips_finished_steps(self):
        with self.assertRaises(RuntimeError):
            self.run_sync(FakeOps(crash_chunk={"inbox/c.png"}))
        st = self.state()
        self.assertEqual(st["c.png"]["status"], "parsed")  # parsed, not chunked
        ops = FakeOps()
        r = self.run_sync(ops)
        self.assertGreaterEqual(r["counts"]["resume"], 1)
        self.assertNotIn("inbox/c.png", ops.did("parse"))  # not parsed twice
        self.assertIn("inbox/c.png", ops.did("chunk"))
        self.assertIn("inbox/c.png", ops.did("meta"))
        self.assertEqual({r["status"] for r in self.state().values()}, {"ok"})
        self.assertEqual(self.run_sync(FakeOps())["counts"]["unchanged"], 3)

    def test_failed_parse_not_retried_until_content_changes(self):
        ops = FakeOps(fail_parse={"inbox/a.csv"})
        self.run_sync(ops)
        self.assertEqual(self.state()["a.csv"]["status"], "failed")
        ops = FakeOps()
        self.assertEqual(self.run_sync(ops)["counts"]["unchanged"], 3)
        self.assertEqual(ops.did("parse"), [])
        self.write("a.csv", "fixed\n")
        ops = FakeOps()
        self.run_sync(ops)
        self.assertEqual(ops.did("parse"), ["inbox/a.csv"])

    def test_adopts_already_parsed_corpus_without_reparse(self):
        from docsearch.ingest import _write_log, parsed_path
        from docsearch.manifest import Doc, write_manifest

        corpus = self.cfg.path("corpus_dir")
        (corpus / "x").mkdir(parents=True)
        (corpus / "x" / "old.csv").write_text("a\n1\n")
        write_manifest(corpus / "manifest.csv", [Doc("x/old", "x/old.csv", "csv", "x", "CC0")])
        parsed_path(self.cfg, "x/old").parent.mkdir(parents=True, exist_ok=True)
        parsed_path(self.cfg, "x/old").write_text("{}")
        _write_log(self.cfg.path("parsed_dir") / "parse_log.csv", {"x/old": {"doc_id": "x/old", "status": "ok"}})
        ops = FakeOps()
        r = S.sync(self.cfg, None, ops=ops)  # default dir = corpus_dir
        self.assertEqual((r["counts"]["adopted"], r["counts"]["new"]), (1, 0))
        self.assertEqual(ops.did("parse"), [])
        self.assertEqual(self.state()["old.csv"]["doc_id"], "x/old")


class TestSkippedFiles(Base):
    """Every file sync does not parse is named: ignored by extension, 0-byte files, .doc without a converter."""

    def say(self, **kw):
        lines: list[str] = []
        with mock.patch.object(S.ui, "say", lines.append):
            r = self.run_sync(**kw)
        return r, lines

    def no_converter(self):
        self.cfg["ingest"]["doc_converter"] = "off"

    def test_ignored_files_are_counted_by_extension(self):
        for name in ("x.py", "y.py", "z.txt", "Makefile"):
            self.write(name, "text")
        r, lines = self.say(dry_run=True)
        self.assertEqual(r["counts"]["ignored"], 5)
        self.assertIn("ignored 5: .py 2, (no extension) 1, .md 1, .txt 1", lines)

    def test_empty_files_are_skipped_and_named_not_parsed(self):
        self.write("sub/empty.csv", "")
        ops = FakeOps()
        r, lines = self.say(ops=ops)
        self.assertEqual((r["counts"]["new"], r["counts"]["empty"]), (3, 1))
        self.assertNotIn("inbox/sub/empty.csv", ops.did("parse"))
        self.assertNotIn("empty.csv", self.state())
        self.assertEqual(lines[2:4], ["empty (0 bytes), skipped: 1", "  empty    sub/empty.csv"])
        self.write("sub/empty.csv", "a\n1\n")  # it gets content: now a new file
        ops = FakeOps()
        self.assertEqual(self.say(ops=ops)[0]["counts"]["new"], 1)
        self.assertEqual(ops.did("parse"), ["inbox/sub/empty.csv"])

    def test_old_word_file_is_a_docx_doc_that_keeps_its_name(self):
        self.write("old.doc", "old word bytes")
        with mock.patch("docsearch.docconv.shutil.which", lambda n: f"/bin/{n}"):
            plan = S.make_plan(self.cfg, S.State(self.cfg.path("sync_manifest")), self.inbox)
        d = next(fi.doc for fi in plan.new if fi.doc.doc_id.endswith(".doc"))
        self.assertEqual((d.doc_id, d.format, Path(d.path).name), ("inbox/old.doc", "docx", "old.doc"))

    def test_doc_without_converter_is_skipped_named_and_picked_up_later(self):
        self.write("old.doc", "old word bytes")
        self.no_converter()
        ops = FakeOps()
        r, lines = self.say(ops=ops)
        self.assertEqual((r["counts"]["new"], r["counts"]["skipped"]), (3, 1))
        self.assertNotIn("inbox/old.doc", ops.did("parse"))
        self.assertNotIn("old.doc", self.state())
        self.assertEqual(lines[2:4], ["1 .doc file skipped: ingest.doc_converter is off", "  skipped  old.doc"])
        self.cfg["ingest"]["doc_converter"] = "auto"
        ops = FakeOps()
        with mock.patch("docsearch.docconv.shutil.which", lambda n: f"/bin/{n}" if n == "textutil" else None):
            self.assertEqual(self.run_sync(ops)["counts"]["new"], 1)
        self.assertEqual(ops.did("parse"), ["inbox/old.doc"])

    def test_an_indexed_doc_stays_when_the_converter_goes(self):
        self.write("old.doc", "old word bytes")
        self.cfg["ingest"]["doc_converter"] = "textutil"
        with mock.patch("docsearch.docconv.shutil.which", lambda n: f"/bin/{n}"):
            self.run_sync()
        self.no_converter()
        ops = FakeOps()
        r = self.run_sync(ops)
        self.assertEqual((r["counts"]["unchanged"], r["counts"]["deleted"], r["counts"]["skipped"]), (4, 0, 0))
        self.write("old.doc", "edited old word bytes")
        ops = FakeOps()
        r = self.run_sync(ops)
        self.assertEqual((r["counts"]["changed"], r["counts"]["deleted"], r["counts"]["skipped"]), (0, 0, 1))
        self.assertEqual(ops.did("unindex_text"), [])  # the indexed version stays until it can be converted

    def test_an_indexed_file_that_reads_0_bytes_keeps_its_indexed_version(self):
        # CR2: a cloud placeholder or a file mid-rewrite reads 0 bytes for a while; that must not unindex it.
        self.run_sync()
        self.write("a.csv", "")
        ops = FakeOps()
        r = self.run_sync(ops)
        self.assertEqual((r["counts"]["deleted"], r["counts"]["empty"]), (0, 1))
        self.assertEqual((ops.did("unindex_text"), ops.did("remove")), ([], []))
        self.assertEqual(self.state()["a.csv"]["status"], "ok")

    def test_a_skipped_file_outside_the_folder_through_a_link_is_named_not_a_crash(self):
        # CR2: scan resolves links, so a skipped path can sit outside the synced folder.
        outside = self.root / "elsewhere"
        outside.mkdir()
        (outside / "old.doc").write_text("")
        (self.inbox / "link.doc").symlink_to(outside / "old.doc")
        _, lines = self.say(dry_run=True)
        self.assertIn(f"  empty    {(outside / 'old.doc').resolve()}", lines)

    def test_a_parsed_doc_finishes_without_a_converter(self):
        # CR2: a .doc parsed before a crash needs chunk and index only, not the converter.
        self.write("old.doc", "old word bytes")
        self.cfg["ingest"]["doc_converter"] = "textutil"
        with mock.patch("docsearch.docconv.shutil.which", lambda n: f"/bin/{n}"), self.assertRaises(RuntimeError):
            self.run_sync(FakeOps(crash_chunk={"inbox/old.doc"}))
        self.assertEqual(self.state()["old.doc"]["status"], "parsed")
        self.no_converter()
        ops = FakeOps()
        r = self.run_sync(ops)
        self.assertEqual(r["counts"]["skipped"], 0)
        self.assertNotIn("inbox/old.doc", ops.did("parse"))
        self.assertIn("inbox/old.doc", ops.did("chunk"))
        self.assertEqual(self.state()["old.doc"]["status"], "ok")

    def test_a_doc_that_failed_for_want_of_a_converter_is_tried_again_when_one_is_found(self):
        # CR2: a failed state counts as unchanged, so the failure reason decides a retry.
        self.write("old.doc", "old word bytes")
        ops = FakeOps(fail_parse={"inbox/old.doc"})
        ops.parse = lambda d: {"status": "failed", "error": "old.doc: no converter (install LibreOffice or set "
                                                            "ingest.doc_converter)"}  # fmt: skip
        self.cfg["ingest"]["doc_converter"] = "textutil"
        with mock.patch("docsearch.docconv.shutil.which", lambda n: f"/bin/{n}"):
            self.run_sync(ops)
        self.assertEqual(self.state()["old.doc"]["status"], "failed")
        ops = FakeOps()
        with mock.patch("docsearch.docconv.shutil.which", lambda n: f"/bin/{n}"):
            self.assertEqual(self.run_sync(ops)["counts"]["changed"], 1)
        self.assertEqual(ops.did("parse"), ["inbox/old.doc"])
        st = FakeOps(fail_parse={"inbox/old.doc"})  # a failure for another reason is not retried
        self.write("old.doc", "other bytes")
        with mock.patch("docsearch.docconv.shutil.which", lambda n: f"/bin/{n}"):
            self.run_sync(st)
            ops = FakeOps()
            self.assertEqual(self.run_sync(ops)["counts"]["changed"], 0)
        self.assertEqual(ops.did("parse"), [])

    def test_a_doc_read_by_textutil_is_read_again_when_libreoffice_comes(self):
        # CR2: textutil drops tables; once soffice is on PATH, the .doc goes through it.
        self.write("old.doc", "old word bytes")
        self.cfg["ingest"]["doc_converter"] = "auto"
        sha = S.file_sha256(self.inbox / "old.doc")
        cache = self.cfg.path("cache_dir") / "doc_docx"
        with mock.patch("docsearch.docconv.shutil.which", lambda n: f"/bin/{n}" if n == "textutil" else None):
            self.run_sync()
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{sha}-textutil.docx").write_bytes(b"PK")
        ops = FakeOps()
        with mock.patch("docsearch.docconv.shutil.which", lambda n: f"/bin/{n}"):
            r = self.run_sync(ops)
        self.assertEqual(r["counts"]["changed"], 1)
        self.assertEqual(ops.did("parse"), ["inbox/old.doc"])
        ops = FakeOps()
        (cache / f"{sha}-soffice.docx").write_bytes(b"PK")
        with mock.patch("docsearch.docconv.shutil.which", lambda n: f"/bin/{n}"):
            self.assertEqual(self.run_sync(ops)["counts"]["changed"], 0)

    def test_the_converted_docx_of_an_old_version_is_removed(self):
        # CR2: the doc_docx cache kept one .docx per version of every .doc, forever.
        self.write("old.doc", "v1")
        self.cfg["ingest"]["doc_converter"] = "textutil"
        cache = self.cfg.path("cache_dir") / "doc_docx"
        cache.mkdir(parents=True, exist_ok=True)
        with mock.patch("docsearch.docconv.shutil.which", lambda n: f"/bin/{n}"):
            self.run_sync()
            v1 = cache / f"{S.file_sha256(self.inbox / 'old.doc')}-textutil.docx"
            v1.write_bytes(b"PK")
            self.write("old.doc", "v2")
            self.run_sync()
            self.assertFalse(v1.exists())
            v2 = cache / f"{S.file_sha256(self.inbox / 'old.doc')}-textutil.docx"
            v2.write_bytes(b"PK")
            (self.inbox / "old.doc").unlink()
            self.run_sync()
        self.assertEqual(list(cache.iterdir()), [])

    def test_failed_files_are_named_with_their_error(self):
        _, lines = self.say(ops=FakeOps(fail_parse={"inbox/a.csv"}))
        self.assertEqual(lines[-1], "  failed   inbox/a.csv: boom")


class TestGate(Base):
    def test_gate_eval_runs_text_only_when_the_page_store_is_empty(self):
        # CR2: FIX1 covered `search`; the gate's eval still stopped on an empty page store.
        self.cfg["visual"]["default"] = "pplx-v2-late"
        seen = {}

        def fake_run_eval(cfg, plan, **kw):
            seen.update(mode=plan[0][2], visual=kw.get("visual"))
            return [{"ndcg@10": 0.5}]

        with mock.patch("docsearch.eval.run_eval", fake_run_eval), \
             mock.patch("docsearch.altembed.has_vectors", lambda d: False):  # fmt: skip
            self.assertEqual(S.eval_ndcg(self.cfg, None, False), 0.5)
        self.assertEqual(seen, {"mode": "hybrid", "visual": None})

    def test_gate_pass_fail_and_tolerance(self):
        self.cfg["sync"]["gate_tolerance"] = 0.01
        r = self.run_sync(FakeOps(), use_gate=True, eval_fn=lambda live: 0.80)
        self.assertTrue(r["gate"]["promoted"])  # first gated version becomes live
        live = S.read_live(self.cfg)
        self.assertEqual(live["ndcg"], 0.80)
        v1 = live["tables"]

        self.write("a.csv", "worse\n")
        ops = FakeOps()
        ops.version = 10
        r = self.run_sync(ops, use_gate=True, eval_fn=lambda live: 0.75)
        self.assertFalse(r["gate"]["promoted"])
        self.assertEqual(S.read_live(self.cfg)["tables"], v1)  # previous version stays live

        self.write("a.csv", "ok\n")
        ops = FakeOps()
        ops.version = 20
        r = self.run_sync(ops, use_gate=True, eval_fn=lambda live: 0.795)  # within tolerance
        self.assertTrue(r["gate"]["promoted"])
        self.assertEqual(S.read_live(self.cfg)["tables"], {"chunks_hybrid": 21})

    def test_gate_evaluates_live_when_its_score_is_unknown(self):
        S.promote(self.cfg, {"tables": {"chunks_hybrid": 1}, "visual": None}, None, "ungated")
        seen = []

        def fake_eval(live):
            seen.append(live is not None)
            return 0.70 if live is None else 0.80

        ok, info = S.gate(self.cfg, {"tables": {"chunks_hybrid": 2}, "visual": None}, fake_eval, 0.005)
        self.assertFalse(ok)
        self.assertEqual(sorted(seen), [False, True])
        self.assertEqual(info["live_ndcg"], 0.80)


class TestStoreHelpers(unittest.TestCase):
    def test_drop_ids_and_shard_numbering(self):
        from docsearch.altembed import VectorStore, _done_ids, _save_shard, drop_ids

        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            v = lambda n: np.ones((n, 4), np.float16)  # noqa: E731
            _save_shard(d, ["a#p1", "b#p1"], [v(2), v(3)])
            _save_shard(d, ["c#p1"], [v(1)])
            snap = Path(t) / "snap"
            snap.mkdir()
            for f in d.glob("shard_*"):
                os.link(f, snap / f.name)
            self.assertEqual(drop_ids(d, {"a#p1", "c#p1"}), 2)
            self.assertEqual(_done_ids(d), {"b#p1"})
            _save_shard(d, ["e#p1"], [v(1)])  # must not reuse a live shard number
            self.assertEqual(_done_ids(d), {"b#p1", "e#p1"})
            self.assertEqual(_done_ids(snap), {"a#p1", "b#p1", "c#p1"})  # hard-linked snapshot untouched
            cfg = load_config(Path(t) / "none.yaml", env={})
            cfg.raw["models"].setdefault("embedders", {})["x"] = {"kind": "visual"}
            store = VectorStore(cfg, "x", "pages", d)
            self.assertEqual(sorted(store.ids), ["b#p1", "e#p1"])
            self.assertEqual(int(store.lens[store.ids.index("b#p1")]), 3)

    def test_drop_from_single_shard_keeps_the_rest(self):
        from docsearch.altembed import VectorStore, _done_ids, _save_shard, drop_ids

        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _save_shard(d, ["a#p1", "b#p1", "c#p1"], [np.full((2, 4), i, np.float16) for i in range(3)])
            drop_ids(d, {"b#p1"})
            self.assertEqual(_done_ids(d), {"a#p1", "c#p1"})
            cfg = load_config(Path(t) / "none.yaml", env={})
            cfg.raw["models"].setdefault("embedders", {})["x"] = {"kind": "visual"}
            store = VectorStore(cfg, "x", "pages", d)  # loads: every committed json has its npy
            self.assertEqual(store.mat.shape, (4, 4))
            self.assertEqual(sorted(set(store.mat[:, 0].tolist())), [0.0, 2.0])

    def test_replace_doc_chunks(self):
        from docsearch.chunk import Chunk, replace_doc_chunks

        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "c.jsonl"
            p.write_text("".join(json.dumps({"doc_id": x, "chunk_id": x + "#0"}) + "\n" for x in "abc"))
            new = Chunk(chunk_id="b#0", doc_id="b", format="csv", source="s", pages=[], text="new", n_tokens=1)
            self.assertEqual(replace_doc_chunks(p, {"b"}, [new]), 1)
            rows = [json.loads(x) for x in p.open()]
            self.assertEqual([r["doc_id"] for r in rows], ["a", "c", "b"])
            self.assertEqual(rows[-1]["text"], "new")


class TestIngestChangeDetection(unittest.TestCase):
    def test_changed_same_name_reparsed_and_legacy_rows_backfilled(self):
        from docsearch.ingest import file_sha256, is_current

        with tempfile.TemporaryDirectory() as t:
            src, out = Path(t) / "a.csv", Path(t) / "a.json"
            src.write_text("v1")
            out.write_text("{}")
            row = {"status": "ok", "sha256": file_sha256(src)}
            self.assertTrue(is_current(src, out, row))
            src.write_text("v2")
            self.assertFalse(is_current(src, out, row))
            legacy = {"status": "ok"}
            self.assertTrue(is_current(src, out, legacy))  # trusted once ...
            self.assertEqual(legacy["sha256"], file_sha256(src))  # ... and backfilled
            src.write_text("v3")
            self.assertFalse(is_current(src, out, legacy))

    def test_parse_log_is_written_when_its_folder_does_not_exist_yet(self):
        # The first file of a fresh sync can fail before its parsed JSON creates the folder (DOC1: a .doc with
        # no converter); the failure must still reach the log, not crash the sync.
        import csv

        from docsearch.ingest import _write_log

        with tempfile.TemporaryDirectory() as t:
            log = Path(t) / "work" / "parsed" / "parse_log.csv"
            _write_log(log, {"inbox/a.doc": {"doc_id": "inbox/a.doc", "status": "failed", "error": "no converter"}})
            rows = list(csv.DictReader(log.open()))
            self.assertEqual([(r["doc_id"], r["status"], r["error"]) for r in rows],
                             [("inbox/a.doc", "failed", "no converter")])  # fmt: skip


class TestNoLibraryProgressBars(unittest.TestCase):
    """sentence-transformers calls must pass show_progress_bar=False; our rich bar is driven by the batch loop."""

    def test_every_encode_and_predict_disables_tqdm(self):
        names = {"encode", "encode_document", "encode_query", "predict"}
        missing = []
        for f in Path(__file__).resolve().parents[1].joinpath("src", "docsearch").glob("*.py"):
            for node in ast.walk(ast.parse(f.read_text())):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in names
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "model"
                    and not any(k.arg == "show_progress_bar" for k in node.keywords)
                ):
                    missing.append(f"{f.name}:{node.lineno}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
