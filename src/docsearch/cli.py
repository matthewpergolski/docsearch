"""`uv run docsearch <command>` entry point."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import ui
from .config import load_config

PIPELINE = {"sync", "ingest", "chunk", "index", "render", "eval", "all", "search", "answer"}


@contextmanager
def timed(cfg, step: str):
    t0 = time.time()
    yield
    trial = cfg["trial"]["name"]
    p = cfg.path("results_dir") / (f"timings_{trial}.json" if trial else "timings.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tm = json.loads(p.read_text()) if p.exists() else {}
    tm[step] = round(time.time() - t0, 1)
    p.write_text(json.dumps(tm, indent=2))
    ui.info(f"[timing] {step}: {tm[step]:.0f}s")


def _store_stage_time(cfg, key: str, seconds: float, min_s: float = 30.0) -> None:
    """Remember how long an `all` stage took (weights for the overall line next time). Skipped under min_s:
    that is a resumed stage with nothing to do, not its real cost."""
    if seconds < min_s:
        return
    p = cfg.path("results_dir") / "stage_timings.json"  # not timings.json: the report expects flat numbers there
    tm = json.loads(p.read_text()) if p.exists() else {}
    tm[key] = round(seconds, 1)
    p.write_text(json.dumps(tm, indent=2))


def _iso_day(s: str) -> str:
    from datetime import date

    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a date as YYYY-MM-DD, got {s!r}") from None


def _csv_list(xs) -> list[str]:
    """`--x a,b` and `--x a b` both give [a, b]."""
    return [p for x in xs or [] for p in str(x).split(",") if p]


def _one_change_plan(cfg, a) -> tuple[list, tuple | None, dict]:
    """A one-change run against the current default: the single challenger run, the reference run only when
    runs.csv has no cached run with the default settings, and the config overrides for the challenger."""
    from .eval import cached_reference

    if not (a.visual and a.fusion == "format_aware"):
        raise SystemExit("--rerank-skip-formats / --visual-blind-formats need --visual ALIAS --fusion format_aware")  # noqa: E501
    over: dict = {}
    tags = []
    if getattr(a, "reranker", None):
        over[("models", "reranker")] = a.reranker  # resolved (alias -> HF id) after the reference lookup
        tags.append(a.reranker.replace("/", "_"))
    if a.rerank_skip_formats:
        over[("search", "rerank_skip_formats")] = _csv_list(a.rerank_skip_formats)
        tags.append("rrskip_" + "_".join(over[("search", "rerank_skip_formats")]))
    if getattr(a, "distinct_boost", None):
        over[("search", "distinct_boost")] = float(a.distinct_boost)
        tags.append(f"distinct{a.distinct_boost:g}")
    if getattr(a, "late_visual_formats", None):
        over[("search", "late_visual_formats")] = _csv_list(a.late_visual_formats)
        tags.append("latevis_" + "_".join(over[("search", "late_visual_formats")]))
    if getattr(a, "number_boost", None):
        over[("search", "number_boost")] = float(a.number_boost)
        if getattr(a, "number_boost_formats", None):
            over[("search", "number_boost_formats")] = _csv_list(a.number_boost_formats)
        tags.append(f"numboost{a.number_boost:g}")
    if getattr(a, "rerank_context", False):
        over[("search", "rerank_context")] = True
        tags.append("rrctx")
    if getattr(a, "rerank_unit_chunks", None):
        over[("search", "rerank_unit_chunks")] = int(a.rerank_unit_chunks)
        tags.append(f"unitcap{a.rerank_unit_chunks}")
    if getattr(a, "page_reranker", None):
        over[("search", "page_reranker")] = a.page_reranker
        tags.append("pagerr_" + a.page_reranker.replace("/", "_"))
    # The page reranker is on in the default (M16b), so these two act on their own, not only with --page-reranker.
    if getattr(a, "page_rerank_pages", None):
        over[("search", "page_rerank_pages")] = int(a.page_rerank_pages)
        tags.append(f"pagerr{a.page_rerank_pages}")
    if getattr(a, "page_rerank_formats", None):
        over[("search", "page_rerank_formats")] = _csv_list(a.page_rerank_formats)
        tags.append("pagerrfmt_" + "_".join(over[("search", "page_rerank_formats")]))
    if getattr(a, "visual_weights", None):
        over[("search", "visual_weights")] = {f: float(w) for f, w in (x.split("=", 1) for x in a.visual_weights)}
        tags.append("visw_" + "_".join(x.replace("=", "") for x in a.visual_weights))
    if getattr(a, "period_demote", False):
        over[("search", "period_demote")] = True
        tags.append("period")
    if getattr(a, "table_lookup", False):
        over[("search", "table_lookup")] = True
        tags.append("tablelookup")
    if getattr(a, "mode", "plain") == "agent":
        tags.append("agent")
        if getattr(a, "agent_research", False):
            tags.append("research")
        if getattr(a, "agent_deck", False):
            tags.append("deck")
    if getattr(a, "visual_text_rich", None):
        over[("visual", "text_rich_chars")] = int(a.visual_text_rich)
        tags.append(f"textrich{a.visual_text_rich}")
    if getattr(a, "office_shared_pages", None):
        over[("visual", "office_shared_pages")] = int(a.office_shared_pages)
        tags.append(f"officeshared{a.office_shared_pages}")
        if getattr(a, "office_own_formats", None):
            over[("visual", "office_own_formats")] = _csv_list(a.office_own_formats)
            tags.append("own_" + "_".join(over[("visual", "office_own_formats")]))
    if a.visual_blind_formats:
        over[("visual", "blind_formats")] = _csv_list(a.visual_blind_formats)
        tags.append("blind_" + "_".join(over[("visual", "blind_formats")]))
    name = a.name or f"hybrid_visual_fa_{a.visual}_rerank_{'_'.join(tags)}"
    plan = [(name, a.variant, "hybrid_visual_fa", True)]
    vid = cfg["models"]["embedders"][a.visual]["id"]
    from .chunk import base_variant

    ref_variant = base_variant(a.variant) or a.variant  # a chunk-variant challenger compares to its base
    if a.variant != ref_variant:
        tags.append(a.variant)
        name = a.name or f"hybrid_visual_fa_{a.visual}_rerank_{'_'.join(tags)}"
        plan = [(name, a.variant, "hybrid_visual_fa", True)]
    hit = cached_reference(cfg, a.full, ref_variant, vid)  # current (default) settings, before the overrides
    if hit:
        ui.say(f"reference: cached run {hit['name']} ({hit['run_id']}, nDCG@10 {float(hit['ndcg@10']):.3f}, "
               f"n={hit['n_queries']}); not re-run")  # fmt: skip
        return plan, None, over
    ref = f"reference_hybrid_visual_fa_{a.visual}_rerank"
    ui.say(f"reference: no cached run with the default settings for this scope; running {ref} first")
    return plan, (ref, ref_variant, "hybrid_visual_fa", True), over


def _device_line(cfg) -> str:
    from .config import pick_device, pick_dtype

    m = cfg["models"]
    if m["backend"] == "api":
        return f"models: api (embedder {m['embed_api_url']}, reranker {m['rerank_api_url']}); text-only"
    dev = pick_device(m["device"])
    emb, rr = (
        str(pick_dtype(dev, m[k], r)).replace("torch.", "")
        for k, r in (("embed_dtype", "embed"), ("rerank_dtype", "rerank"))
    )
    return f"device: {dev} (embed {emb}, rerank {rr})"


def _quiet_hub(cfg, a) -> None:
    """Warn (naming them) only for models this command will load that are not fully cached; otherwise hide
    the Hugging Face download bars."""
    from .hfcache import missing, models_for

    gone = missing(models_for(cfg, a.cmd, a))
    if gone:
        ui.say("not cached yet, will download: " + ", ".join(f"{r} ({r.label})" for r in gone))
    else:
        ui.disable_hub_bars()


def _run_all(cfg, a) -> int:
    """The adopted pipeline end to end. Every step skips finished work, so re-running resumes."""
    from .chunk import build_chunks
    from .eval import run_eval
    from .index import build_index
    from .report import build_report
    from .visual import build_visual_store, render

    vis = a.visual or cfg["visual"].get("default")
    steps = []
    if a.with_ingest:
        from .ingest import ingest
        from .prepare import prepare

        steps += [("prepare", lambda: prepare(cfg, None)), ("ingest", lambda: ingest(cfg, None, False, None))]
    steps += [
        ("chunk hybrid", lambda: build_chunks(cfg, "hybrid", a.full)),
        ("index hybrid", lambda: build_index(cfg, "hybrid")),
    ]
    if vis:
        steps += [
            ("render pages", lambda: render(cfg, a.full)),
            (f"index pages ({vis})", lambda: build_visual_store(cfg, vis)),
        ]
    out: dict = {}
    plan = (
        [(f"hybrid_visual_fa_{vis}_rerank", "hybrid", "hybrid_visual_fa", True),
         (f"hybrid_visual_fa_{vis}_norerank", "hybrid", "hybrid_visual_fa", False)]
        if vis else [("baseline_hybrid_rerank", "hybrid", "hybrid", True), ("baseline_hybrid_norerank", "hybrid", "hybrid", False)]
    )  # fmt: skip
    note = "docsearch all" + (" --full" if a.full else "")
    steps += [
        ("eval", lambda: out.__setitem__("runs", run_eval(cfg, plan, None, None, note, a.full, None, vis))),
        ("report", lambda: out.__setitem__("report", build_report(cfg))),
    ]
    tp = cfg.path("results_dir") / "stage_timings.json"
    stored = json.loads(tp.read_text()) if tp.exists() else {}
    overall = ui.Overall([n for n, _ in steps], stored)
    ui.set_overall(overall)
    try:
        for i, (name, fn) in enumerate(steps, 1):
            overall.at(i - 1)
            ui.say(overall.line())
            t0 = time.time()
            with ui.stage(i, len(steps), name), timed(cfg, name.replace(" ", "_")):
                fn()
            _store_stage_time(cfg, ui.stage_key(name), time.time() - t0)
    finally:
        ui.set_overall(None)
    from .sync import follow_working

    follow_working(cfg)
    ui.summary_table(out.get("runs", []), out.get("report"))
    return 0


def _llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["api", "cli"], help="api = LiteLLM keys, cli = subscription CLIs")
    p.add_argument("--provider", choices=["claude", "openai", "grok", "ollama"])
    p.add_argument("--model", help="override the configured model id")


def _judge(cfg, a):
    mode = getattr(a, "abstain", "threshold")
    if mode == "threshold":
        return None
    if mode == "gated" and a.gate is None:
        raise SystemExit("--abstain gated needs --gate SCORE (see scripts/abstain_sweep.py)")
    from .judge import Judge
    from .llm import LLMClient, check_available, settings

    over = {**QUESTION_DEFAULTS, **{k: getattr(a, k) for k in ("backend", "provider", "model") if getattr(a, k, None)}}
    client = LLMClient(settings(cfg, "judge", **over))
    if not a.judge_cache_only:
        check_available(client.s, client.env, client.which)
    ui.say(f"[eval] abstention judge: {client.s.backend} {client.s.provider} {client.s.model or '(CLI default)'}")
    return Judge(cfg, client, cache_only=a.judge_cache_only, gate=a.gate if mode == "gated" else None)


QUESTION_DEFAULTS = {"backend": "cli", "provider": "grok"}  # a Grok subscription via the grok CLI


def _questions(cfg, a) -> int:
    from pathlib import Path

    from . import questions as Q

    if a.qcmd == "lint":
        path = Path(a.path) if a.path else Q.custom_path(cfg)
        if a.drop_stitched:
            Q.lint_drop(path, undo=a.undo)
        else:
            Q.lint(path)
        return 0
    if a.qcmd == "paraphrase":
        from .llm import LLMClient, LLMError, LLMUnavailable, settings

        over = {**QUESTION_DEFAULTS, **{k: getattr(a, k) for k in ("backend", "provider", "model") if getattr(a, k)}}
        try:
            Q.paraphrase(cfg, LLMClient(settings(cfg, "questions", **over)), a.max_ngram, a.max_overlap, a.attempts)
        except LLMUnavailable as e:
            print(f"[skip] LLM backend unavailable: {e}", file=sys.stderr)
            return 2
        except LLMError as e:
            print(f"[error] {e} (re-run the same command to resume)", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            return 130
        return 0
    if a.qcmd == "apply-suggestions":
        Q.apply_suggestions(Path(a.path) if a.path else Q.custom_path(cfg),
                            Path(a.suggestions) if a.suggestions else Q.suggestions_path(cfg),
                            set(a.only) if a.only else None, a.reviewer, a.undo, a.force)  # fmt: skip
        return 0
    if a.qcmd == "review":
        sp = None
        if a.suggestions is not None:
            sp = Path(a.suggestions) if a.suggestions else Q.suggestions_path(cfg)
        Q.review(cfg, Path(a.path) if a.path else None, include_reviewed=a.all, suggestions=sp,
                 reviewer=a.reviewer)  # fmt: skip
        return 0
    from .llm import LLMClient, LLMError, LLMUnavailable, settings

    over = {**QUESTION_DEFAULTS, **{k: getattr(a, k) for k in ("backend", "provider", "model") if getattr(a, k)}}
    try:
        client = LLMClient(settings(cfg, "questions", **over))
        print(f"[questions] backend={client.s.backend} provider={client.s.provider} model={client.s.model or '(CLI default)'}"
              f" images={'yes' if client.supports_images() and not a.no_images else 'no (text only)'}")  # fmt: skip
        Q.draft(cfg, client, a.n, Q.parse_mix(a.mix) if a.mix else None, Path(a.out) if a.out else None,
                not a.no_closed_book_filter, not a.no_images, a.seed, hard=a.hard, hard_from=a.hard_from)  # fmt: skip
    except LLMUnavailable as e:
        print(f"[skip] LLM backend unavailable: {e}\n       Try --provider claude or --provider openai "
              "(codex). Check with: uv run docsearch llm-status", file=sys.stderr)  # fmt: skip
        return 2
    except LLMError as e:
        print(f"[error] {e} (re-run the same command to resume)", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="docsearch", description="Mixed-format search + RAG test bench")
    ap.add_argument("--config", help="path to docsearch.yaml (default: search upward from cwd)")
    ap.add_argument("-v", "--verbose", action="store_true", help="show warnings and library logs live")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="download datasets, build corpus/ + manifest + questions")
    p.add_argument("--only", nargs="*", choices=["vidore", "finslides", "ocb", "images"])
    p = sub.add_parser("ingest", help="parse corpus with Docling")
    p.add_argument("--source")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("subset", help="(re)build the fixed-seed box subset in data/subset.json")
    p = sub.add_parser("chunk", help="build chunks for a variant")
    p.add_argument("--full", action="store_true", help="ignore the box subset; use the whole corpus")
    p.add_argument(
        "--variant",
        default="all",
        choices=[
            "all",
            "hybrid",
            "recursive",
            "hybrid_tbl",
            "hybrid_sheets",
            "hybrid_pdf",
            "hybrid_pdftable",
            "hybrid_scan",
            "hybrid_scan_noocr",
            "hybrid_mix",
            "hybrid_mix_noocr",
            "hybrid_pdfocr",
        ],  # fmt: skip
    )
    p.add_argument(
        "--pdf-tables",
        action="store_true",
        help="M15: same as --variant hybrid_pdftable (Docling tables kept whole in PDFs)",
    )
    p = sub.add_parser("index", help="embed chunks into LanceDB + full-text index")
    p.add_argument(
        "--variant",
        default="all",
        choices=[
            "all",
            "hybrid",
            "recursive",
            "hybrid_tbl",
            "hybrid_sheets",
            "hybrid_pdf",
            "hybrid_pdftable",
            "hybrid_scan",
            "hybrid_scan_noocr",
            "hybrid_mix",
            "hybrid_mix_noocr",
            "hybrid_pdfocr",
            "hybrid_ctx",
            "hybrid_ctx_slides",
        ],  # fmt: skip
    )
    p.add_argument(
        "--pdf-tables",
        action="store_true",
        help="M15: same as --variant hybrid_pdftable (Docling tables kept whole in PDFs)",
    )
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--embedder", help="challenger embedder alias (models.embedders): build its vector store")
    p.add_argument("--visual", help="visual embedder alias: embed rendered pages (run `render` first)")
    p = sub.add_parser("render", help="render pages/slides/images for visual retrieval (LibreOffice for docx/pptx)")
    p.add_argument("--full", action="store_true", help="ignore the box subset")
    p = sub.add_parser("sync", help="incremental update from a folder: parse/chunk/embed only new or changed files")
    p.add_argument("--dir", help="folder to sync (default: paths.corpus_dir)")
    p.add_argument("--dry-run", action="store_true", help="only print new/changed/deleted/unchanged counts")
    p.add_argument("--gate", action="store_true", help="eval after sync; promote to live only if nDCG holds")
    p.add_argument("--watch", action="store_true", help="repeat in the foreground every --interval seconds")
    p.add_argument("--interval", type=int, help="seconds between --watch runs (default: sync.interval)")
    p.add_argument(
        "--full",
        action="store_true",
        help="ignore the box subset (use on a GPU or Apple-silicon machine, like `all --full`)",
    )
    p = sub.add_parser("eval", help="run retrieval eval and append to results/runs.csv")
    p.add_argument("--runs", nargs="*", help="run names from the default plan (default: all)")
    p.add_argument("--full", action="store_true", help="ignore the box subset; use every question")
    p.add_argument("--sources", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--notes", default="")
    p.add_argument("--reranker", help="reranker alias (models.rerankers) or HF id; runs one hybrid+rerank challenger")
    p.add_argument("--name", help="run name for the --reranker challenger (default rerank_<alias>), a one-change "
                   "run, or a single --runs run")  # fmt: skip
    p.add_argument(
        "--abstain",
        choices=["threshold", "judge", "gated"],
        default="threshold",
        help="judge = ask an LLM (grok CLI by default) whether the top 5 answer each question; gated = trust a "
        "top rerank score >= --gate and ask the judge only below it",
    )
    p.add_argument("--gate", type=float, help="gate for --abstain gated (pick it with scripts/abstain_sweep.py)")
    p.add_argument("--judge-cache-only", action="store_true",
                   help="never call the LLM; use cached verdicts only (missing ones are reported)")  # fmt: skip
    p.add_argument("--provider", choices=["claude", "openai", "grok"], help="judge provider (default grok)")
    p.add_argument("--backend", choices=["api", "cli"], help="judge backend (default cli)")
    p.add_argument("--model", help="judge model id (default: the CLI's)")
    p.add_argument("--mode", default="plain",
                   help="plain (default: enrich.llm and agent.mode off for the run), enrich or agent; the --abstain judge is a separate LLM opt-in")  # fmt: skip
    p.add_argument("--agent-provider", choices=["openai", "grok", "claude"],
                   help="--mode agent: the LLM CLI that picks the page (default agent.provider)")  # fmt: skip
    p.add_argument("--agent-model", help="--mode agent: model for --agent-provider (default: that CLI's own)")
    p.add_argument("--agent-research", action="store_true",
                   help="--mode agent, one-change run (decisions.tsv D1): when the reader names no page, two LLM rewrites "
                        "of the question are searched, merged with the plain list (RRF) and read once more")  # fmt: skip
    p.add_argument("--agent-deck", action="store_true",
                   help="--mode agent, one-change run (decisions.tsv D2): the gate counts a slide deck as one file; when "
                        "only that opens it, the reader reads the top 20 pages (or agent.pages when larger)")  # fmt: skip
    p.add_argument("--questions", choices=["labeled", "custom", "custom_para", "hard", "tables", "all"],
                   default="labeled",
                   help="custom = data/questions_custom.jsonl (by type + abstention), tables = "
                        "data/questions_tables.jsonl (Phase C3); runs the M3 default unless --visual/--fusion are given")  # fmt: skip
    p.add_argument("--reviewed-only", action="store_true", help="custom questions marked keep/edited only")
    p.add_argument(
        "--embedder", help="challenger embedder alias (models.embedders); runs hybrid with and without rerank"
    )
    p.add_argument("--visual", help="visual embedder alias: run visual-only and hybrid+visual fusion evals")
    p.add_argument("--variant", default="hybrid", help="chunk variant for --visual runs (e.g. hybrid_tbl)")
    p.add_argument(
        "--pdf-tables",
        action="store_true",
        help="one-change run (M15): search the chunks_hybrid_pdftable table (= --variant hybrid_pdftable)",
    )
    p.add_argument(
        "--fusion",
        default="rrf",
        choices=["rrf", "format_aware"],
        help="with --visual: plain RRF, or format-aware RRF (unseen formats keep their text rank)",
    )
    p.add_argument("--suffix", default="", help="appended to run names (e.g. for flag variants)")
    p.add_argument(
        "--rerank-skip-formats",
        nargs="+",
        metavar="FMT",
        help="one-change run: these formats keep their fused rank (no rerank). With --visual --fusion "
        "format_aware --name N: runs only N, plus a default-settings reference if none is cached",
    )
    p.add_argument(
        "--visual-blind-formats",
        nargs="+",
        metavar="FMT",
        help="one-change run: format-aware fusion treats these formats (e.g. pptx,csv) as unseen by the visual "
        "model, so they keep their text rank; same reference rule as --rerank-skip-formats",
    )
    p.add_argument("--distinct-boost", type=float, metavar="POS",
                   help="one-change run (M11): after rerank, move a top candidate up POS positions per query term "
                        "that few of the top candidates contain (tells near-duplicate pages apart)")  # fmt: skip
    p.add_argument("--rerank-context", action="store_true",
                   help="one-change run (M12): prepend document name + page to reranker input and rerank "
                        "slide formats too (overrides rerank_skip_formats)")  # fmt: skip
    p.add_argument("--late-visual-formats", nargs="+", metavar="FMT",
                   help="one-change run (M13): after rerank, re-fuse these formats (e.g. pdf) with the visual page "
                        "rank (RRF, k=search.late_visual_k)")  # fmt: skip
    p.add_argument(
        "--number-boost",
        type=float,
        metavar="POS",
        help="one-change run (M14): move a candidate up POS positions per exact query number on it",
    )
    p.add_argument("--number-boost-formats", nargs="+", metavar="FMT", help="formats for --number-boost (default pdf)")
    p.add_argument("--rerank-unit-chunks", type=int, metavar="N",
                   help="one-change run (M17): a page/file puts at most N chunks in a row in the fused list, so "
                        "the rerank window holds more distinct pages/files; its other chunks follow at the end")  # fmt: skip
    p.add_argument("--page-reranker", metavar="ALIAS",
                   help="one-change run (M16): after the text rerank, a vision-language reranker "
                        "(models.page_rerankers alias or HF id) reorders the top pages by their images")  # fmt: skip
    p.add_argument("--page-rerank-pages", type=int, metavar="N", help="pages per query it scores (default 10)")
    p.add_argument("--page-rerank-formats", nargs="+", metavar="FMT",
                   help="formats it reorders (default pdf slide_image)")  # fmt: skip
    p.add_argument("--visual-weights", nargs="+", metavar="FMT=W",
                   help="one-change run (M20): weight of a format's visual rank in the fusion, e.g. slide_image=2")  # fmt: skip
    p.add_argument("--visual-text-rich", type=int, metavar="CHARS",
                   help="one-change run (M25): a pdf/docx/pptx page or file with at least CHARS characters of text "
                        "keeps its text rank; page pictures vote only for pages with little text (scans, charts)")  # fmt: skip
    p.add_argument("--office-shared-pages", type=int, metavar="N",
                   help="one-change run (M24): with every Office page rendered, only the first N pages of a "
                        "docx/pptx compete with other formats; the rest order Office files among themselves")  # fmt: skip
    p.add_argument("--office-own-formats", nargs="+", metavar="FMT",
                   help="with --office-shared-pages: the Office formats whose later pages are used (default docx pptx)")  # fmt: skip
    p.add_argument("--period-demote", action="store_true",
                   help="one-change run (M21): files whose name carries another quarter than the query's "
                        "(e.g. ibm-3q13 for a 4Q 2013 question) go behind the other candidates")  # fmt: skip
    p.add_argument("--table-lookup", action="store_true",
                   help="one-change run (M33, Phase C2): a file's page whose table row and column the query names "
                        "takes the file's first slots (needs work/cells.sqlite from scripts/build_cells.py)")  # fmt: skip
    sub.add_parser("report", help="write results/report.html")
    p = sub.add_parser("diagram", help="draw the ingest, query and eval flows from the code and this config (Mermaid)")
    p.add_argument("flow", nargs="?", choices=["ingest", "query", "eval", "all"], default="all")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true", help="put the drawings between the diagram markers in "
                   "README.md, REPORT.md and DEPLOYMENT.md (the committed docsearch.yaml, no DOCSEARCH_* env)")  # fmt: skip
    g.add_argument("--check", action="store_true", help="write nothing; exit 1 when a Markdown drawing is stale")
    g.add_argument("--html", metavar="PATH", help="write a page with the drawings (light or dark)")
    p = sub.add_parser("serve", help="the local search page and its API at http://HOST:PORT (needs the ui extra: "
                       "uv sync --extra ui; the page: cd ui && bun install && bun run build)")  # fmt: skip
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p = sub.add_parser("search", help="search the corpus")
    p.add_argument("query")
    p.add_argument("--variant", help="chunk variant (default: hybrid; hybrid_ctx_slides with enrich.llm on)")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--reranker", help="reranker alias (models.rerankers) or HF id")
    p.add_argument("-k", type=int, default=10)
    from .manifest import EXT_FORMAT

    formats = sorted({*EXT_FORMAT.values(), "slide_image"})
    p.add_argument("--format", nargs="+", action="extend", metavar="FMT", choices=formats,
                   help="only files of these formats: " + ", ".join(formats))  # fmt: skip
    p.add_argument("--after", type=_iso_day, metavar="YYYY-MM-DD",
                   help="only files dated on or after this day. A file's date comes from inside the file (Office "
                        "properties, PDF metadata): its last change, else its creation. A file with no date is left "
                        "out")  # fmt: skip
    p.add_argument("--before", type=_iso_day, metavar="YYYY-MM-DD",
                   help="only files dated on or before this day; a file with no date is left out")  # fmt: skip
    p.add_argument("--undated", action="store_true",
                   help="with --after/--before: keep files that have no date of their own (CSV, images, many "
                        "scans); without it, search says how many it left out")  # fmt: skip
    p.add_argument("--folder", metavar="PREFIX",
                   help="only files in this folder (relative to the corpus folder) or its subfolders")  # fmt: skip
    p.add_argument("--scores", action="store_true",
                   help="show each page's score and its scorer (rerank, page or fused: different scales)")  # fmt: skip
    p.add_argument("--json", action="store_true",
                   help="print the grouped results as JSON on stdout (status lines go to stderr)")  # fmt: skip
    p.add_argument("--agent", action="store_true",
                   help="also run the +agent step (agent.mode on does the same): an LLM reads the top pages and "
                        "names the answer and its page, shown first; the result list stays the plain one")  # fmt: skip
    p = sub.add_parser("meta", help="read title, author and dates from inside each file in corpus/manifest.csv into "
                       "meta.jsonl next to the index (no models; `sync` keeps it current)")  # fmt: skip
    p = sub.add_parser("answer", help="retrieve then answer with citations (needs an LLM backend; no search filters)")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=8)
    _llm_args(p)
    q = sub.add_parser("questions", help="harder custom questions: draft with an LLM, then hand-review")
    qs = q.add_subparsers(dest="qcmd", required=True)
    p = qs.add_parser("draft", help="draft multi-doc / no-answer / image questions into data/questions_custom.jsonl")
    p.add_argument("--n", type=int, default=200, help="total questions (resumable; existing ones count)")
    p.add_argument("--mix", help="type weights, default multi_doc=0.4,no_answer=0.25,image=0.35")
    p.add_argument("--no-closed-book-filter", action="store_true")
    p.add_argument("--no-images", action="store_true", help="text only even if the backend takes images")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", help="output path (default data/questions_custom.jsonl, or questions_hard.jsonl)")
    p.add_argument(
        "--hard",
        action="store_true",
        help="hard-page set -> data/questions_hard.jsonl: near-duplicate "
        "pages, dense tables, pages missed in the M3 public eval, look-alike doc pairs",
    )
    p.add_argument("--hard-from", help="public-set run whose per-question results give the missed pages "
                   "(default m3_skip_slide_rerank)")  # fmt: skip
    _llm_args(p)
    p = qs.add_parser("lint", help="read-only: flag stitched questions (too many '?', 'and', or words)")
    p.add_argument("--path", help="questions file (default data/questions_custom.jsonl)")
    p.add_argument("--drop-stitched", action="store_true", help="mark flagged ones dropped (backup first)")
    p.add_argument("--undo", action="store_true", help="with --drop-stitched: restore the lint-dropped ones")
    p = qs.add_parser("paraphrase", help="rewrite kept questions in a user's own words -> questions_custom_para.jsonl")
    p.add_argument("--max-ngram", type=int, default=4, help="reject rewrites copying a word n-gram this long or longer "
                   "from the gold chunks (default 4)")  # fmt: skip
    p.add_argument("--max-overlap", type=float, default=0.25, help="reject if more than this share of the rewrite's "
                   "word 3-grams appear in the gold chunks (default 0.25)")  # fmt: skip
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--reviewed-only", action="store_true", default=True, help=argparse.SUPPRESS)
    _llm_args(p)
    p = qs.add_parser("apply-suggestions", help="write suggested keep/drop/edit decisions (backup first; --undo)")
    p.add_argument("--only", nargs="*", choices=["keep", "drop", "edit"], help="only these decisions (default all)")
    p.add_argument("--reviewer", default="reviewer", help="recorded as meta.reviewer (default reviewer)")
    p.add_argument("--suggestions", help="suggestions file (default data/questions_custom.suggestions.jsonl)")
    p.add_argument("--path", help="questions file (default data/questions_custom.jsonl)")
    p.add_argument("--undo", action="store_true", help="restore every question this reviewer applied")
    p.add_argument("--force", action="store_true", help="also overwrite questions already reviewed by someone")
    p = qs.add_parser("review", help="terminal loop: keep / drop / edit each drafted question")
    p.add_argument("--all", action="store_true", help="also revisit already reviewed questions")
    p.add_argument("--suggestions", nargs="?", const="", help="show suggestions (default "
                   "data/questions_custom.suggestions.jsonl); Enter accepts")  # fmt: skip
    p.add_argument("--reviewer", default="human", help="name recorded on decisions (meta.reviewer)")
    p.add_argument("--path", help="questions file (default data/questions_custom.jsonl)")
    p = sub.add_parser("draft-questions", help="LLM-draft multi-doc + no-answer questions (unreviewed)")
    p.add_argument("--n-multi", type=int, default=20)
    p.add_argument("--n-none", type=int, default=10)
    p.add_argument("--no-closed-book-filter", action="store_true")
    _llm_args(p)
    sub.add_parser("llm-status", help="show which LLM backends are usable here (no keys printed)")
    p = sub.add_parser(
        "all", help="chunk, index, render, index --visual, eval (format-aware fusion), report; resumable"
    )
    p.add_argument("--full", action="store_true", help="whole corpus (recommended on a GPU or Apple-silicon machine)")
    p.add_argument("--with-ingest", action="store_true", help="run prepare + ingest first (both resumable)")
    p.add_argument("--visual", help="visual embedder alias (default: visual.default in the config)")
    p.add_argument("--skip-prepare", action="store_true", help=argparse.SUPPRESS)  # kept for old scripts

    a = ap.parse_args(argv)
    from pathlib import Path

    cfg = load_config(Path(a.config) if a.config else None)
    if a.cmd == "eval":
        from .eval import apply_mode

        apply_mode(cfg, a.mode)
        if a.mode == "agent" and not (a.visual and a.fusion == "format_aware"):
            raise SystemExit("eval --mode agent runs one challenger on the default pipeline: give --visual ALIAS "
                             "--fusion format_aware (and --name)")  # fmt: skip
        if a.agent_provider and a.agent_provider != cfg["agent"]["provider"]:
            cfg["agent"]["provider"], cfg["agent"]["model"] = a.agent_provider, ""  # "" = that CLI's default model
        if a.agent_model is not None:
            cfg["agent"]["model"] = a.agent_model
        if a.agent_research:
            if a.mode != "agent":
                raise SystemExit("--agent-research needs --mode agent; nothing was run")
            cfg["agent"]["research"] = "on"
        if a.agent_deck:
            if a.mode != "agent":
                raise SystemExit("--agent-deck needs --mode agent; nothing was run")
            cfg["agent"]["deck_gate"] = "on"
    if getattr(a, "pdf_tables", False):
        a.variant = "hybrid_pdftable"
    variants = ["hybrid", "recursive"] if getattr(a, "variant", "all") == "all" else [a.variant]
    if cfg["models"]["backend"] == "api" and (getattr(a, "visual", None) or getattr(a, "embedder", None)):
        raise SystemExit("models.backend api runs text-only: --visual and --embedder load local models; drop them")

    a.out = sys.stdout  # search --json: the JSON goes here, every status line to stderr
    with contextlib.redirect_stdout(sys.stderr) if getattr(a, "json", False) else contextlib.nullcontext():
        log_path = ui.setup(cfg.root / cfg["logging"]["dir"], a.verbose, int(cfg["logging"]["keep_runs"]))
        if a.cmd in PIPELINE:
            ui.say(_device_line(cfg))
            _quiet_hub(cfg, a)
        try:
            if a.cmd == "all":
                return _run_all(cfg, a)
            rc = _dispatch(cfg, a, variants)
            if a.cmd == "index":
                from .sync import follow_working

                follow_working(cfg)
            return rc
        finally:
            del log_path
            if a.cmd in PIPELINE:
                ui.finish()
            ui.close()


def _dispatch(cfg, a, variants) -> int:
    if a.cmd == "prepare":
        from .prepare import prepare

        with timed(cfg, "prepare"):
            log = prepare(cfg, getattr(a, "only", None))
        print(json.dumps(log.get("_merged", {}), indent=2))
    if a.cmd == "ingest":
        from .ingest import ingest

        with timed(cfg, "ingest"):
            ingest(cfg, getattr(a, "source", None), getattr(a, "force", False), getattr(a, "limit", None))
    if a.cmd == "chunk":
        from .chunk import DERIVED, build_chunks, build_derived

        for v in variants:
            with timed(cfg, f"chunk_{v}"):
                if v in DERIVED:
                    build_derived(cfg, v, getattr(a, "full", False))
                else:
                    build_chunks(cfg, v, getattr(a, "full", False))
    if a.cmd == "sync":
        from .sync import sync, watch

        root = Path(a.dir).expanduser() if a.dir else None
        if a.watch:
            if a.dry_run:
                raise SystemExit("--watch and --dry-run don't combine")
            return watch(cfg, root, int(a.interval or cfg["sync"]["interval"]), a.gate, a.full)
        res = sync(cfg, root, dry_run=a.dry_run, use_gate=a.gate, full=a.full)
        return 1 if res.get("gate", {}).get("promoted") is False else 0
    if a.cmd == "render":
        from .visual import render

        with timed(cfg, "render"):
            render(cfg, a.full)
    if a.cmd == "index":
        from .index import build_index

        if getattr(a, "visual", None):
            from .visual import build_visual_store

            with timed(cfg, f"index_visual_{a.visual}"):
                build_visual_store(cfg, a.visual, a.rebuild)
            variants = []
        elif getattr(a, "embedder", None):
            from .altembed import build_store

            for v in variants if a.variant != "all" else ["hybrid"]:
                with timed(cfg, f"index_{v}_{a.embedder}"):
                    build_store(cfg, a.embedder, v, a.rebuild)
            variants = []
        for v in variants:
            with timed(cfg, f"index_{v}"):
                build_index(cfg, v, getattr(a, "rebuild", False))
    if a.cmd == "eval":
        from .eval import DEFAULT_PLAN, run_eval

        plan = DEFAULT_PLAN
        for sub, parent in (("number_boost_formats", "number_boost"), ("office_own_formats", "office_shared_pages")):
            if getattr(a, sub, None) and not getattr(a, parent, None):  # a sub-option alone would be ignored
                raise SystemExit(f"--{sub.replace('_', '-')} needs --{parent.replace('_', '-')}")
        from .chunk import base_variant

        one_change = (
            getattr(a, "visual", None)
            and a.fusion == "format_aware"
            and any(
                getattr(a, k, None)
                for k in (
                    "reranker",
                    "rerank_skip_formats",
                    "visual_blind_formats",
                    "distinct_boost",
                    "rerank_context",
                    "late_visual_formats",
                    "number_boost",
                    "page_reranker",
                    "page_rerank_pages",
                    "page_rerank_formats",
                    "rerank_unit_chunks",
                    "visual_weights",
                    "period_demote",
                    "table_lookup",
                    "office_shared_pages",
                    "visual_text_rich",
                )
            )  # fmt: skip
            or getattr(a, "visual", None)
            and a.fusion == "format_aware"
            and a.mode == "agent"
            or getattr(a, "visual", None)
            and a.fusion == "format_aware"
            and base_variant(a.variant) is not None  # a derived or enriched chunk variant against its base
        )
        if getattr(a, "reranker", None) and not one_change:
            from .config import resolve_reranker

            alias = resolve_reranker(cfg, a.reranker)
            plan = [(a.name or f"rerank_{alias}", "hybrid", "hybrid", True)]
        qset = getattr(a, "questions", "labeled")
        if qset != "labeled" and not getattr(a, "visual", None):  # M3 default settings
            a.visual, a.fusion = cfg["visual"]["default"] or None, "format_aware"
            tag = {"judge": "_judge", "gated": "_gated"}.get(getattr(a, "abstain", "threshold"), "")
            mode = "hybrid_visual_fa" if a.visual else "hybrid"  # no visual.default (models.backend api): text-only
            plan = [(a.name or f"m3_questions_{qset}{tag}", a.variant, mode, True)]
        elif getattr(a, "visual", None):
            v = a.visual
            if a.fusion == "format_aware":
                rr, norr = (a.name, f"{a.name}_norerank") if a.name else (f"hybrid_visual_fa_{v}_rerank",
                                                                            f"hybrid_visual_fa_{v}_norerank")  # fmt: skip
                plan = [(rr + a.suffix, a.variant, "hybrid_visual_fa", True),
                        (norr + a.suffix, a.variant, "hybrid_visual_fa", False)]  # fmt: skip
            else:
                plan = [(f"visual_{v}_only", "hybrid", "visual", False),
                    (f"hybrid_visual_{v}_rerank", "hybrid", "hybrid_visual", True),
                    (f"hybrid_visual_{v}_norerank", "hybrid", "hybrid_visual", False)]  # fmt: skip
        elif getattr(a, "embedder", None):
            plan = [
                (f"embed_{a.embedder}_{tag}", "hybrid", "hybrid", rr)
                for tag, rr in (("rerank", True), ("norerank", False))
            ]
        elif getattr(a, "runs", None):
            plan = [r for r in DEFAULT_PLAN if r[0] in a.runs]
            if getattr(a, "name", None) and len(plan) == 1:
                plan = [(a.name, *plan[0][1:])]
        results = []
        if one_change:
            plan, ref, over = _one_change_plan(cfg, a)
            if ref:  # reference first, with the default settings
                with timed(cfg, "eval"):
                    results += run_eval(cfg, [ref], a.sources, a.limit, a.notes, a.full, None, a.visual)
            for (sec, key), val in over.items():
                if (sec, key) == ("models", "reranker"):
                    from .config import resolve_reranker

                    resolve_reranker(cfg, val)
                else:
                    cfg[sec][key] = val
        with timed(cfg, "eval"):
            results += run_eval(cfg, plan, getattr(a, "sources", None), getattr(a, "limit", None), getattr(a, "notes", ""),
                     getattr(a, "full", False), getattr(a, "embedder", None), getattr(a, "visual", None),
                     questions=qset, reviewed_only=getattr(a, "reviewed_only", False), judge=_judge(cfg, a),
                     eval_mode=a.mode)  # fmt: skip
    if a.cmd in ("report", "eval"):
        from .report import build_report

        report = build_report(cfg)
        if a.cmd == "eval":
            ui.summary_table(results, report)
        else:
            ui.say(f"Report: {report}")
    if a.cmd == "subset":
        from .subset import build_subset

        build_subset(cfg)
    if a.cmd == "diagram":
        from . import diagram

        if a.write or a.check:  # the docs show the public defaults, so the env of this shell is left out
            doc_cfg = load_config(cfg.root / "docsearch.yaml", env={})
            names = diagram.write(cfg.root, doc_cfg) if a.write else diagram.stale(cfg.root, doc_cfg)
            print(f"diagram: {'wrote' if a.write else 'stale'}: {', '.join(names) or 'none'}")
            return 1 if a.check and names else 0
        flows = list(diagram.FLOWS) if a.flow == "all" else [a.flow]
        if a.html:
            Path(a.html).write_text(diagram.html(cfg, flows))
            print(f"diagram: wrote {a.html}")
            return 0
        print("\n".join(diagram.mermaid(cfg, f) for f in flows), end="")
        return 0
    if a.cmd == "meta":
        from .meta import build_meta

        st = build_meta(cfg)
        print(f"meta: {st['files']} files, {st['with_title']} with a title, {st['with_date']} with a date "
              f"-> {st['path']}")  # fmt: skip
    if a.cmd == "serve":
        from .web import serve

        return serve(cfg, a.host, a.port)
    if a.cmd == "search":
        from . import display
        from .meta import Filter

        flt = Filter(tuple(a.format or ()), a.after, a.before, a.folder, a.undated)
        try:
            res, notes = display.run(cfg, a.query, a.variant, not a.no_rerank, a.k, flt, a.agent or None, a.reranker)
        except display.SearchRefused as e:
            raise SystemExit(str(e)) from None
        for note in notes:
            print(note)
        if a.json:
            print(json.dumps(display.as_json(res), ensure_ascii=False, indent=2), file=a.out)
        elif res.chunk_ids:
            print(display.render(res, a.scores))
    if a.cmd == "questions":
        return _questions(cfg, a)
    if a.cmd in ("answer", "draft-questions"):
        from .llm import LLMError, LLMUnavailable

        over = {k: getattr(a, k) for k in ("backend", "provider", "model") if getattr(a, k)}
        try:
            if a.cmd == "answer":
                from .answer import answer

                res = answer(cfg, a.question, k=a.k, **over)
                print(res["answer"])
                if res["citations"]:
                    print("\nSources:")
                    for c in res["citations"]:
                        print(f"  [{c['n']}] {c['label']}")
            else:
                from .draft_questions import draft
                from .llm import LLMClient, settings

                draft(cfg, a.n_multi, a.n_none, not a.no_closed_book_filter,
                      client=LLMClient(settings(cfg, "draft", **over)))  # fmt: skip
        except LLMUnavailable as e:
            print(f"[skip] LLM backend unavailable: {e}\n       Search and eval do not need an LLM. "
                  "See README 'LLM backends'. Check with: uv run docsearch llm-status", file=sys.stderr)  # fmt: skip
            return 2
        except LLMError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 1
    if a.cmd == "llm-status":
        from .llm import status

        for b, p, s in status(cfg):
            if b == "api" and p == "ollama" and s == "ready":
                s = "configured (server not probed; run `ollama serve`)"
            print(f"{b:4s} {p:7s} {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
