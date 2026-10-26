"""The web API: the same search as `docsearch search --json`, through Django + Ninja, with a fake search."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HAVE_DJANGO = importlib.util.find_spec("django") is not None and importlib.util.find_spec("ninja") is not None


def results(query="q", confident=True, answer=None):
    from docsearch import display

    page = display.PageHit(3, 1, "Wages", "wages by region", 0.9, "rerank")
    group = display.ResultGroup(1, "src/report.pdf", "pdf", "Annual report", "2024-01-31", "", (page,))
    return display.Results(query, confident, answer, (group,), ("c1",))


@unittest.skipUnless(HAVE_DJANGO, "the web extra is not installed (uv sync --extra ui)")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from docsearch import web

        web.setup()

    def setUp(self):
        from ninja.testing import TestClient

        from docsearch.config import load_config
        from docsearch.web import api

        self.cfg = load_config(env={})
        api.CFG = self.cfg
        self.client = TestClient(api.api)
        self.calls = []

    def fake_run(self, res=None, notes=(), refuse=None):
        def run(cfg, query, variant="hybrid", rerank=True, k=None, flt=None, agent=False, reranker=None, progress=None):
            from docsearch import display

            self.calls.append({"query": query, "k": k, "flt": flt, "agent": agent})
            if refuse:
                raise display.SearchRefused(refuse)
            return res or results(query), list(notes)

        return mock.patch("docsearch.display.run", run)

    def test_search_returns_the_cli_json_shape_and_notes(self):
        with self.fake_run(notes=["2 files have no date"]):
            r = self.client.get("/search?q=wages")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["notes"], ["2 files have no date"])
        self.assertEqual(body["results"]["groups"][0]["pages"][0],
                         {"page": 3, "rank": 1, "title": "Wages", "snippet": "wages by region", "score": 0.9,
                          "scorer": "rerank"})  # fmt: skip
        self.assertEqual((body["results"]["query"], body["results"]["confident"]), ("wages", True))

    def test_search_names_the_query_terms_the_snippets_were_cut_around(self):
        with self.fake_run():
            body = self.client.get("/search?q=What were the wages in 2023, by region?").json()
        self.assertEqual(body["terms"], ["2023", "region", "wages"])

    def test_filters_reach_the_search(self):
        with self.fake_run():
            self.client.get("/search?q=x&k=5&formats=pdf&formats=docx&after=2024-01-01&folder=hr&undated=true")
        f = self.calls[0]["flt"]
        self.assertEqual((self.calls[0]["k"], f.formats, f.after, f.before, f.folder, f.undated),
                         (5, ("pdf", "docx"), "2024-01-01", None, "hr", True))  # fmt: skip

    def race(self, old_tab, new_tab):
        """'old' runs until something stops it (or 0.5 s pass); 'new' arrives while it runs."""
        import threading
        import time

        from docsearch import models

        started, old = threading.Event(), {}

        def run(cfg, query, **kw):
            if query == "old":
                started.set()
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    if models._STOP is not None and models._STOP():  # what the page reranker checks between pages
                        raise models.Superseded
                    time.sleep(0.01)
            return results(query), []

        def get(q, tab):
            return self.client.get(f"/search?q={q}", headers={"X-Docsearch-Tab": tab} if tab else {})

        with mock.patch("docsearch.display.run", run):
            t = threading.Thread(target=lambda: old.update(r=get("old", old_tab)))
            t.start()
            started.wait(5)
            new = get("new", new_tab)
            t.join(5)
        self.assertIsNone(models._STOP)
        return old["r"].status_code, new.status_code

    def test_a_newer_search_from_the_same_tab_stops_the_one_running_which_answers_409(self):
        self.assertEqual(self.race("tab1", "tab1"), (409, 200))

    def test_searches_from_other_tabs_or_without_a_tab_wait_their_turn_and_are_not_stopped(self):
        self.assertEqual(self.race("tab1", "tab2"), (200, 200))
        self.assertEqual(self.race(None, None), (200, 200))

    def test_bad_input_is_a_400_with_the_reason(self):
        for url, why in (("/search?q=%20", "empty"), ("/search?q=x&formats=exe", "exe"),
                         ("/search?q=x&after=2024-13-01", "YYYY-MM-DD")):  # fmt: skip
            r = self.client.get(url)
            self.assertEqual(r.status_code, 400, url)
            self.assertIn(why, r.json()["detail"])

    def test_a_refused_search_is_a_400_with_the_reason(self):
        with self.fake_run(refuse="search filters need file metadata"):
            r = self.client.get("/search?q=x&formats=pdf")
        self.assertEqual((r.status_code, r.json()["detail"]), (400, "search filters need file metadata"))

    def test_the_agent_runs_only_when_agent_mode_is_on(self):
        with self.fake_run():
            self.assertEqual(self.client.get("/search?q=x&agent=true").status_code, 400)
            self.cfg["agent"]["mode"] = "on"
            self.assertEqual(self.client.get("/config").json()["agent"], True)
            self.client.get("/search?q=x&agent=true")
            self.client.get("/search?q=x")
        self.assertEqual([c["agent"] for c in self.calls], [True, False])

    def test_config_names_the_formats_and_says_whether_the_agent_is_on(self):
        body = self.client.get("/config").json()
        self.assertEqual(body["agent"], False)
        self.assertIn("pdf", body["formats"])
        self.assertIn("slide_image", body["formats"])
        self.cfg["web"]["examples"] = ["Q3 freight cost by region"]
        self.assertEqual(self.client.get("/config").json()["examples"], ["Q3 freight cost by region"])

    def test_config_says_whether_filters_can_run(self):
        # filters need file metadata (docsearch meta); without it the page says so instead of failing
        with mock.patch("docsearch.meta.read_meta", return_value={}):
            self.assertEqual(self.client.get("/config").json()["filters"], False)
        with mock.patch("docsearch.meta.read_meta", return_value={"a.pdf": object()}):
            self.assertEqual(self.client.get("/config").json()["filters"], True)

    def test_warm_loads_each_search_model_once_and_then_config_says_ready(self):
        from docsearch.web import api

        loaded = []
        self.addCleanup(api.READY.clear)
        api.READY.clear()
        self.assertEqual(self.client.get("/config").json()["ready"], False)
        with mock.patch("docsearch.models.load_embedder", lambda cfg: loaded.append("embedder")), \
             mock.patch("docsearch.models.load_reranker", lambda cfg: loaded.append("reranker")), \
             mock.patch("docsearch.altembed.query_encoder", lambda cfg, alias: loaded.append(alias)), \
             mock.patch("docsearch.models.load_page_reranker",
                        lambda cfg, name: loaded.append(name) or mock.Mock()), \
             mock.patch("docsearch.display.run", lambda cfg, q, **kw: loaded.append(f"search {q}")):  # fmt: skip
            self.cfg["web"]["examples"] = ["Q3 freight cost by region"]
            api.warm(self.cfg)
        vis, page = self.cfg["visual"]["default"], self.cfg["search"]["page_reranker"]
        # one real search pays the GPU's one-time costs at start (the first search took 8 s, then 1 s)
        self.assertEqual(loaded, ["embedder", "reranker", vis, page, "search Q3 freight cost by region"])
        self.assertEqual(self.client.get("/config").json()["ready"], True)

    def test_a_failed_warm_up_still_lets_searches_run(self):
        from docsearch.web import api

        self.addCleanup(api.READY.clear)
        api.READY.clear()
        with mock.patch("docsearch.models.load_embedder", mock.Mock(side_effect=OSError("no model"))):
            api.warm(self.cfg)
        self.assertTrue(api.READY.is_set())


@unittest.skipUnless(HAVE_DJANGO, "the web extra is not installed (uv sync --extra ui)")
class TestStream(unittest.TestCase):
    """GET /api/search/stream: the same search, as lines of JSON. When slides wait for the page reranker, the
    other results come first, then each slide read, then the final results (as /api/search gives them)."""

    def setUp(self):
        from django.test import Client

        from docsearch import web
        from docsearch.config import load_config
        from docsearch.web import api

        web.setup()
        api.CFG = load_config(env={})
        self.client = Client()

    def events(self, run):
        import json

        with mock.patch("docsearch.display.run", run):
            r = self.client.get("/api/search/stream?q=wages+by+region", HTTP_X_DOCSEARCH_TAB="t")
            body = b"".join(r.streaming_content).decode()
        self.assertEqual(r["Content-Type"], "application/x-ndjson")
        return [json.loads(line) for line in body.splitlines()]

    def test_other_results_first_then_each_slide_then_the_final_results(self):
        from docsearch import models

        def hit(i, doc, fmt, page=1):
            return {"chunk_id": str(i), "doc_id": doc, "format": fmt, "pages": [page], "text": "wages by region"}

        def run(cfg, query, k=None, flt=None, agent=False, progress=None):
            progress(
                "ranked", [hit(0, "r.pdf", "pdf"), hit(1, "s/deck_page_2", "slide_image"), hit(2, "w.docx", "docx")]
            )
            models._ON_PAGE(["s/deck_page_2#p1"], [])
            models._ON_PAGE(["s/deck_page_2#p1"], ["s/deck_page_2#p1"])
            return results(query), []

        ev = self.events(run)
        self.assertEqual([e["event"] for e in ev], ["stage", "early", "pages", "pages", "done"])
        self.assertEqual([g["file"] for g in ev[1]["results"]["groups"]], ["r.pdf", "w.docx"])
        self.assertEqual(ev[1]["terms"], ["region", "wages"])
        self.assertEqual(ev[2]["pages"], [{"unit": "s/deck_page_2#p1", "file": "s/deck_page_2", "page": 1}])
        self.assertEqual((ev[2]["done"], ev[3]["done"]), ([], ["s/deck_page_2#p1"]))
        self.assertEqual(ev[4]["results"]["groups"][0]["file"], "src/report.pdf")
        self.assertEqual(set(ev[4]), {"event", "results", "notes", "terms"})
        self.assertIsNone(models._ON_PAGE)

    def test_no_early_results_when_a_later_step_could_move_them(self):
        from docsearch import models
        from docsearch.web import api

        api.CFG["search"]["number_boost"] = 2.0

        def run(cfg, query, progress=None, **kw):
            progress("ranked", [{"chunk_id": "0", "doc_id": "r.pdf", "format": "pdf", "pages": [1], "text": "t"}])
            models._ON_PAGE(["s/deck_page_2#p1"], [])
            return results(query), []

        self.assertEqual([e["event"] for e in self.events(run)], ["stage", "pages", "done"])

    def test_the_stages_say_what_runs_and_how_many_candidates_of_each_type(self):
        def hit(i, doc, fmt):
            return {"chunk_id": str(i), "doc_id": doc, "format": fmt, "pages": [1], "text": "t"}

        def run(cfg, query, progress=None, **kw):
            progress("candidates", [hit(0, "a.pdf", "pdf"), hit(1, "a.pdf", "pdf"), hit(2, "w.docx", "docx"),
                                    hit(3, "s/deck_page_1", "slide_image"), hit(4, "s/deck_page_2", "slide_image")])  # fmt: skip
            return results(query), []

        ev = self.events(run)
        self.assertEqual([e["event"] for e in ev], ["stage", "stage", "done"])
        self.assertEqual(ev[0], {"event": "stage", "stage": "looking"})
        self.assertEqual(
            ev[1], {"event": "stage", "stage": "ranking", "files": {"pdf": 1, "docx": 1, "slide_image": 1}}
        )

    def test_a_search_that_waits_for_another_says_so_first(self):
        from docsearch.web import api

        api.READY.set()
        self.addCleanup(api.READY.clear)
        api._ONE_AT_A_TIME.acquire()
        threading_timer = __import__("threading").Timer(0.3, api._ONE_AT_A_TIME.release)
        threading_timer.start()
        ev = self.events(lambda cfg, query, **kw: (results(query), []))
        threading_timer.join()
        self.assertEqual(ev[0], {"event": "waiting", "why": "another search"})
        self.assertEqual(ev[-1]["event"], "done")

    def test_a_search_with_no_slides_to_read_gives_only_the_final_results(self):
        ev = self.events(lambda cfg, query, progress=None, **kw: (progress("ranked", []), (results(query), []))[1])
        self.assertEqual([e["event"] for e in ev], ["stage", "done"])

    def test_a_refused_or_replaced_search_ends_with_its_reason(self):
        from docsearch import display, models

        def refuse(exc):
            def run(cfg, query, **kw):
                raise exc

            return run

        self.assertEqual(self.events(refuse(display.SearchRefused("filters need metadata")))[-1],
                         {"event": "error", "status": 400, "detail": "filters need metadata"})  # fmt: skip
        self.assertEqual(self.events(refuse(models.Superseded()))[-1]["status"], 409)


@unittest.skipUnless(HAVE_DJANGO, "the web extra is not installed (uv sync --extra ui)")
class TestPageText(unittest.TestCase):
    """GET /api/text: a page's text from the index, for a preview where there is no page picture."""

    ROWS = [
        {"chunk_id": "hybrid:f/t.xlsx:1", "doc_id": "f/t.xlsx", "format": "xlsx", "pages": [2],
         "text": "File: t.xlsx\nTable 9\nBarista | £50,000\nCost inflation | 0.05"},
        {"chunk_id": "hybrid:f/t.xlsx:0", "doc_id": "f/t.xlsx", "format": "xlsx", "pages": [1],
         "text": "File: t.xlsx\nTable 1\nYear | Revenue"},
        {"chunk_id": "hybrid:s/deck_page_3:0", "doc_id": "s/deck_page_3", "format": "slide_image", "pages": [],
         "text": "File: deck_page_3.png\nOverview\n3Q18 revenue"},
        {"chunk_id": "hybrid:w/a.docx:1", "doc_id": "w/a.docx", "format": "docx", "pages": [],
         "text": "File: a.docx\n**Terms** apply\nto all"},
        {"chunk_id": "hybrid:w/a.docx:0", "doc_id": "w/a.docx", "format": "docx", "pages": [], "text": "File: a.docx\nIntro"},
    ]  # fmt: skip

    def setUp(self):
        from ninja.testing import TestClient

        from docsearch import web
        from docsearch.config import load_config
        from docsearch.web import api

        web.setup()
        api.CFG = load_config(env={})
        self.client = TestClient(api.api)
        rows = mock.patch("docsearch.search.page_rows",
                          lambda cfg, variant, docs: [r for r in self.ROWS if r["doc_id"] in docs])  # fmt: skip
        rows.start()
        self.addCleanup(rows.stop)

    def test_a_sheet_gives_its_tables_in_order(self):
        body = self.client.get("/text?file=f/t.xlsx&page=2").json()
        self.assertEqual(
            body,
            {"format": "xlsx", "parts": [{"title": "Table 9", "body": "Barista | £50,000 / Cost inflation | 0.05"}]},
        )

    def test_a_slide_of_a_deck_and_a_file_without_pages_give_their_text_in_order(self):
        self.assertEqual(self.client.get("/text?file=s/deck&page=3").json()["parts"],
                         [{"title": "Overview", "body": "3Q18 revenue"}])  # fmt: skip
        self.assertEqual(self.client.get("/text?file=w/a.docx").json()["parts"],
                         [{"title": "", "body": "Intro"}, {"title": "", "body": "Terms apply to all"}])  # fmt: skip

    def test_a_file_not_in_the_index_is_a_404(self):
        self.assertEqual(self.client.get("/text?file=nope.pdf&page=1").status_code, 404)


@unittest.skipUnless(HAVE_DJANGO, "the web extra is not installed (uv sync --extra ui)")
class TestFile(unittest.TestCase):
    """GET /api/file: a result's file, for the browser to show. Only files in corpus/manifest.csv."""

    def setUp(self):
        from django.test import Client

        from docsearch import web
        from docsearch.config import load_config
        from docsearch.manifest import Doc, write_manifest
        from docsearch.web import api

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp)
        corpus, outside = self.tmp / "corpus", self.tmp / "elsewhere"
        for p, data in (("reports/a.pdf", b"%PDF-1.4"), ("hr/policy.docx", b"PK docx"), ("hr/plan.docx", b"PK plan"),
                        ("slides/deck_page_2.png", b"\x89PNG"), ("data/t.csv", b"a,b\n1,2\n")):  # fmt: skip
            (corpus / p).parent.mkdir(parents=True, exist_ok=True)
            (corpus / p).write_bytes(data)
        outside.mkdir()
        (outside / "notes.pdf").write_bytes(b"%PDF-1.7 outside")
        docs = [Doc("src/a.pdf", "reports/a.pdf", "pdf", "src", ""), Doc("src/policy.docx", "hr/policy.docx", "docx", "src", ""),
                Doc("src/plan.docx", "hr/plan.docx", "docx", "src", ""),
                Doc("slides/deck_page_2", "slides/deck_page_2.png", "slide_image", "slides", ""),
                Doc("src/t.csv", "data/t.csv", "csv", "src", ""),
                Doc("mine/notes.pdf", str(outside / "notes.pdf"), "pdf", "mine", "")]  # fmt: skip
        write_manifest(corpus / "manifest.csv", docs)
        self.cfg = load_config(env={})
        self.cfg.raw["paths"]["corpus_dir"] = str(corpus)
        self.cfg.raw["paths"]["index_dir"] = str(self.tmp / "work" / "index")
        rendition = self.tmp / "work" / "pages" / "_office_pdf" / "src__policy.docx" / "policy.pdf"
        rendition.parent.mkdir(parents=True)
        rendition.write_bytes(b"%PDF-1.5 rendition")
        web.setup()
        api.CFG = self.cfg
        self.client = Client()

    def get(self, query):
        r = self.client.get(f"/api/file?{query}")
        body = b"".join(r.streaming_content) if r.status_code == 200 else r.content
        return r, body

    def test_a_pdf_is_shown_inline_as_a_pdf(self):
        r, body = self.get("file=src/a.pdf&page=3")
        self.assertEqual((r.status_code, body, r["Content-Type"]), (200, b"%PDF-1.4", "application/pdf"))
        self.assertTrue(r["Content-Disposition"].startswith("inline"))
        self.assertEqual(r["X-Content-Type-Options"], "nosniff")
        self.assertEqual(self.get("file=mine/notes.pdf")[1], b"%PDF-1.7 outside")

    def test_a_slide_deck_page_is_that_slide_image(self):
        r, body = self.get("file=slides/deck&page=2")
        self.assertEqual((r.status_code, body, r["Content-Type"]), (200, b"\x89PNG", "image/png"))
        self.assertEqual(self.get("file=slides/deck&page=9")[0].status_code, 404)

    def test_an_office_file_shows_its_pdf_copy_else_downloads(self):
        r, body = self.get("file=src/policy.docx&page=1")
        self.assertEqual((body, r["Content-Type"]), (b"%PDF-1.5 rendition", "application/pdf"))
        r, body = self.get("file=src/plan.docx")
        self.assertEqual(body, b"PK plan")
        self.assertTrue(r["Content-Disposition"].startswith("attachment"))

    def test_a_csv_shows_as_text(self):
        r, body = self.get("file=src/t.csv")
        self.assertEqual((body, r["Content-Type"]), (b"a,b\n1,2\n", "text/plain; charset=utf-8"))

    def test_only_files_in_the_manifest(self):
        for q in ("file=src/missing.pdf", "file=../corpus/reports/a.pdf", "file=reports/a.pdf"):
            self.assertEqual(self.get(q)[0].status_code, 404, q)


@unittest.skipUnless(HAVE_DJANGO, "the web extra is not installed (uv sync --extra ui)")
class TestPagePicture(unittest.TestCase):
    """GET /api/page: the picture `docsearch render` made of a result's page. Only pages in pages.jsonl."""

    def setUp(self):
        import json

        from django.test import Client

        from docsearch import web
        from docsearch.config import load_config
        from docsearch.web import api

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp)
        pages = self.tmp / "work" / "pages"
        rows = []
        for unit, name, data in (("src/a.pdf#p3", "a/p3.png", b"\x89PNG a3"),
                                 ("slides/deck_page_2#p1", "deck_page_2.png", b"\x89PNG slide")):  # fmt: skip
            (pages / name).parent.mkdir(parents=True, exist_ok=True)
            (pages / name).write_bytes(data)
            doc, page = unit.split("#p")
            rows.append({"unit": unit, "doc_id": doc, "format": "pdf", "page": int(page), "path": str(pages / name)})
        (pages / "pages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        cfg = load_config(env={})
        cfg.raw["paths"]["index_dir"] = str(self.tmp / "work" / "index")
        web.setup()
        api.CFG = cfg
        self.client = Client()

    def get(self, query):
        r = self.client.get(f"/api/page?{query}")
        return r, b"".join(r.streaming_content) if r.status_code == 200 else r.content

    def test_a_rendered_page_is_its_picture(self):
        r, body = self.get("file=src/a.pdf&page=3")
        self.assertEqual((r.status_code, body, r["Content-Type"]), (200, b"\x89PNG a3", "image/png"))
        self.assertEqual(r["X-Content-Type-Options"], "nosniff")

    def test_a_slide_deck_page_is_that_slide(self):
        self.assertEqual(self.get("file=slides/deck&page=2")[1], b"\x89PNG slide")

    def test_a_page_with_no_picture_is_a_404(self):
        for q in ("file=src/a.pdf&page=4", "file=src/a.pdf", "file=src/other.pdf&page=3"):
            self.assertEqual(self.get(q)[0].status_code, 404, q)


@unittest.skipUnless(HAVE_DJANGO, "the web extra is not installed (uv sync --extra ui)")
class TestPage(unittest.TestCase):
    def test_the_built_page_and_its_assets_are_served_and_a_missing_build_is_named(self):
        from django.test import Client

        from docsearch import web

        web.setup()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "assets").mkdir()
            (Path(d) / "index.html").write_text("<!doctype html><title>docsearch</title>")
            (Path(d) / "assets" / "app.js").write_text("console.log(1)")
            with mock.patch.dict(os.environ, {"DOCSEARCH_UI_DIST": d}):
                c = Client()
                self.assertIn(b"<title>docsearch</title>", c.get("/").content)
                self.assertEqual(c.get("/assets/app.js").status_code, 200)
            with mock.patch.dict(os.environ, {"DOCSEARCH_UI_DIST": str(Path(d) / "none")}):
                r = Client().get("/")
                self.assertEqual(r.status_code, 503)
                self.assertIn(b"bun run build", r.content)


if __name__ == "__main__":
    unittest.main()
