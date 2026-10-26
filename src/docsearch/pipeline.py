"""The pipeline stages in run order, for `docsearch diagram` and the README, REPORT and DEPLOYMENT diagrams.

Each stage names the function that does the work and the docsearch.yaml keys that change it. The stage list
does not run anything: sync.apply, search.search, search.rerank_pipeline, display.run and eval.run_eval do.
tests/test_pipeline.py fails when one of those calls a step that is not here, or when a config key has no stage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .config import Config

PLAIN, ENRICH, AGENT = "plain", "+enrich", "+agent"  # the three layers (eval.MODES)


@dataclass(frozen=True)
class Stage:
    id: str  # Mermaid node id
    label: str
    ref: str  # "module:function" that does the work (tests check that it exists)
    layer: str = PLAIN
    keys: tuple[str, ...] = ()  # "section.key" in docsearch.yaml that change this stage
    on: Callable[[Config], bool] = lambda cfg: True
    detail: Callable[[Config], str] = lambda cfg: ""
    after: tuple[str, ...] = ()  # stage ids that feed this one; () = the stage before it in the list
    experiment: bool = False  # an off experiment step is drawn in one shared "experiments (off)" box
    store: bool = False  # a data store (cylinder)
    step: str = ""  # the sync.Ops method that runs this stage (ingest only)
    llm: str = ""  # the docsearch.yaml section whose provider and model this stage calls ("agent", "enrich")


def _list(v) -> str:
    return ", ".join(v) if v else "none"


def _short(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _visual(cfg: Config) -> str | None:
    return cfg["visual"].get("default") or None


def _embedder(cfg: Config) -> str:
    m = cfg["models"]
    if m["backend"] == "api":
        return f"API: {m['embed_api_model'] or _short(m['embedder'])}"
    return _short(m["embedder"])


def _reranker(cfg: Config) -> str:
    m = cfg["models"]
    name = _short(m["rerank_api_model"] or m["reranker"])
    return f"API: {name}" if m["backend"] == "api" else name


def _page_reranker(cfg: Config) -> str:
    alias = cfg["search"].get("page_reranker") or ""
    return _short(cfg["models"].get("page_rerankers", {}).get(alias, alias))


def _llm(cfg: Config, section: str, generic: bool) -> str:
    """The LLM a stage calls. generic (the drawings in the docs): the settings that pick it, not a vendor's model."""
    s = cfg[section]
    named = f"{s.get('provider') or ''} {s.get('model') or ''}".strip()
    return named if named and not generic else f"LLM from {section}.provider, {section}.model"


INGEST: tuple[Stage, ...] = (
    Stage(
        "scan",
        "Scan folder: new, changed, deleted files",
        "sync:make_plan",
        keys=("sync.source",),
        detail=lambda c: "PDF, DOCX, PPTX, XLSX, CSV, images, DOC",
    ),
    Stage(
        "docconv",
        "Convert .doc to .docx",
        "docconv:docx_for",
        keys=("ingest.doc_converter",),
        on=lambda c: c["ingest"]["doc_converter"] != "off",
        detail=lambda c: f"converter: {c['ingest']['doc_converter']} (soffice, then textutil)",
    ),
    Stage(
        "parse",
        "Parse with Docling",
        "ingest:_parse_one",
        step="parse",
        keys=("ingest.pdf_ocr",),
        after=("scan", "docconv"),
        detail=lambda c: "PDF OCR on" if c["ingest"].get("pdf_ocr") else "PDF text layer; OCR for images",
    ),
    Stage(
        "meta",
        "File metadata: title, author, dates",
        "meta:extract",
        step="meta",
        after=("parse",),
        detail=lambda c: "for the type and date filters",
    ),
    Stage(
        "chunk",
        "Chunk",
        "chunk:DocChunker",
        step="chunk",
        keys=("sync.variants", "chunking.max_tokens", "chunking.max_chunks_per_doc", "chunking.max_parsed_mb"),
        after=("parse",),
        detail=lambda c: (
            f"{_list(c['sync']['variants'])}, {c['chunking']['max_tokens']} tokens, "
            f"max {c['chunking']['max_chunks_per_doc']} per spreadsheet"
        ),
    ),
    Stage(
        "enrich",
        "LLM context line per slide image",
        "enrich:Enricher",
        layer=ENRICH,
        keys=("enrich.llm", "enrich.backend", "enrich.provider", "enrich.model"),
        on=lambda c: c["enrich"]["llm"] != "off",
        llm="enrich",
        detail=lambda c: "writes the hybrid_ctx_slides chunks",
    ),
    Stage(
        "embed",
        "Embed text + full-text index",
        "index:build_index",
        step="index",
        keys=("models.embedder", "models.backend"),
        after=("chunk", "enrich"),
        detail=_embedder,
    ),
    Stage(
        "render",
        "Render page images",
        "visual:render_doc",
        step="render",
        keys=("visual.dpi", "visual.max_side", "visual.office_max_pages"),
        on=lambda c: bool(_visual(c)),
        after=("parse",),
        detail=lambda c: (
            f"{c['visual']['dpi']} dpi, {c['visual']['office_max_pages']} pages per DOCX/PPTX (LibreOffice)"
        ),
    ),
    Stage(
        "pagevec",
        "Embed page images",
        "visual:build_visual_store",
        step="index",
        keys=("visual.default",),
        on=lambda c: bool(_visual(c)),
        detail=lambda c: _visual(c) or "visual.default is empty",
    ),
    Stage(
        "db",
        "LanceDB tables + page store + meta.jsonl",
        "index:connect",
        store=True,
        after=("embed", "pagevec", "meta"),
    ),
    Stage(
        "gate",
        "Gate: promote to live only if nDCG@10 holds",
        "sync:gate",
        keys=("sync.gate_tolerance",),
        on=lambda c: False,
        detail=lambda c: f"sync --gate, tolerance {c['sync']['gate_tolerance']}",
    ),
)

QUERY: tuple[Stage, ...] = (
    Stage("q", "Query + optional type and date filter", "meta:Filter"),
    Stage(
        "bm25",
        "Full-text search",
        "search:Retriever.bm25",
        keys=("search.bm25_k",),
        after=("q",),
        detail=lambda c: f"top {c['search']['bm25_k']}",
    ),
    Stage(
        "vec",
        "Text vector search",
        "search:Retriever.vector",
        keys=("search.vector_k",),
        after=("q",),
        detail=lambda c: f"top {c['search']['vector_k']}, {_embedder(c)}",
    ),
    Stage(
        "vis",
        "Page-image search",
        "search:Retriever._visual_hits",
        keys=("visual.visual_k",),
        on=lambda c: bool(_visual(c)),
        after=("q",),
        detail=lambda c: f"top {c['visual']['visual_k']}, {_visual(c)}",
    ),
    Stage(
        "fuse",
        "Fusion (RRF)",
        "search:Retriever._fuse_format_aware",
        keys=(
            "search.rrf_k",
            "visual.blind_formats",
            "visual.text_rich_chars",
            "visual.office_shared_pages",
            "search.visual_weights",
            "search.rerank_unit_chunks",
        ),
        after=("bm25", "vec", "vis"),
        detail=lambda c: ("format-aware: " if _visual(c) else "text only: ") + f"k {c['search']['rrf_k']}",
    ),
    Stage(
        "period",
        "Period demote",
        "search:period_demote",
        keys=("search.period_demote",),
        on=lambda c: bool(c["search"]["period_demote"]),
        experiment=True,
    ),
    Stage(
        "rerank",
        "Text rerank",
        "search:rerank",
        keys=(
            "models.reranker",
            "search.rerank_candidates",
            "search.rerank_skip_formats",
            "search.rerank_min_chars",
            "search.rerank_context",
        ),
        detail=lambda c: (
            f"{_reranker(c)}, top {c['search']['rerank_candidates']}; skips {_list(c['search']['rerank_skip_formats'])}"
        ),
    ),
    Stage(
        "pagerr",
        "Page-image rerank",
        "search:page_rerank",
        keys=("search.page_reranker", "search.page_rerank_pages", "search.page_rerank_formats"),
        on=lambda c: bool(_visual(c) and c["search"]["page_reranker"]),
        detail=lambda c: (
            f"{_page_reranker(c)}, top {c['search']['page_rerank_pages']} pages; "
            f"{_list(c['search']['page_rerank_formats'])} only"
        ),
    ),
    Stage(
        "table",
        "Table cell lookup",
        "search:table_lookup",
        keys=("search.table_lookup",),
        on=lambda c: bool(c["search"]["table_lookup"]),
        experiment=True,
    ),
    Stage(
        "late",
        "Late visual fusion",
        "search:late_visual",
        keys=("search.late_visual_formats", "search.late_visual_k"),
        on=lambda c: bool(c["search"]["late_visual_formats"]),
        experiment=True,
    ),
    Stage(
        "num",
        "Number boost",
        "search:number_boost",
        keys=("search.number_boost", "search.number_boost_formats"),
        on=lambda c: float(c["search"]["number_boost"]) > 0,
        experiment=True,
    ),
    Stage(
        "distinct",
        "Near-duplicate tie-break",
        "search:distinct_tiebreak",
        keys=("search.distinct_boost",),
        on=lambda c: float(c["search"]["distinct_boost"]) > 0,
        experiment=True,
    ),
    Stage(
        "topk",
        "Top results",
        "display:build",
        keys=("search.top_k", "search.abstain_threshold"),
        detail=lambda c: (
            f"top {c['search']['top_k']}; 'No confident match' below {c['search'].get('abstain_threshold')}"
        ),
    ),
    Stage(
        "agate",
        "Gate: 2+ pages of one file in the top pages?",
        "agent:gate",
        layer=AGENT,
        keys=("agent.mode", "agent.gate_k", "agent.deck_gate"),
        on=lambda c: c["agent"]["mode"] == "on",
        detail=lambda c: f"top {c['agent']['gate_k']}",
    ),
    Stage(
        "agent",
        "LLM reads the top pages, names the page",
        "agent:Agent",
        layer=AGENT,
        keys=("agent.backend", "agent.provider", "agent.model", "agent.pages", "agent.page_chars", "agent.research"),
        on=lambda c: c["agent"]["mode"] == "on",
        llm="agent",
        detail=lambda c: f"{c['agent']['pages']} pages",
    ),
)

EVAL: tuple[Stage, ...] = (
    Stage("qs", "Question sets", "eval:load_custom", detail=lambda c: "labeled, custom, hard, tables"),
    Stage("mode", "Mode: plain, +enrich or +agent", "eval:apply_mode"),
    Stage("run", "Same query pipeline per question", "eval:run_eval"),
    Stage("metrics", "recall@10, nDCG@10, MRR@10, hit@10", "metrics:score_query"),
    Stage("abst", "Abstention on no-answer questions", "eval:abstention", keys=("search.abstain_threshold",)),
    Stage("runs", "results/runs.csv + per-question JSONL", "eval:_write_run", store=True, after=("metrics", "abst")),
    Stage(
        "ans",
        "Answer checks: correct, cites the page, says not found",
        "answer:answer",
        on=lambda c: False,
        after=("run",),
        detail=lambda c: "not built yet",
    ),
    Stage("report", "HTML report", "report:build_report", after=("runs",)),
)

FLOWS: dict[str, tuple[Stage, ...]] = {"ingest": INGEST, "query": QUERY, "eval": EVAL}

# Config keys that change no stage in a drawing (paths, batch sizes, logs, data download). A new key must go into a
# stage's keys or here; tests/test_pipeline.py checks it.
NOT_DRAWN: frozenset[str] = frozenset({"visual.shard_pages", "visual.text_rich_formats", "visual.office_own_formats",
    "chunking.pdf_max_tokens", "chunking.pdf_overlap_tokens", "agent.workers", "sync.interval"})  # fmt: skip
DRAWN_SECTIONS = ("ingest", "chunking", "search", "visual", "enrich", "agent", "sync")


@dataclass
class Resolved:
    stage: Stage
    on: bool
    detail: str
    after: tuple[str, ...] = field(default=())


def resolve(cfg: Config, flow: str, generic: bool = False) -> list[Resolved]:
    """The stages of one flow with on/off and labels from this config. generic: name no LLM model (the docs)."""
    out, prev = [], None
    for s in FLOWS[flow]:
        after = s.after or ((prev,) if prev else ())
        detail = "; ".join(x for x in (_llm(cfg, s.llm, generic) if s.llm else "", s.detail(cfg)) if x)
        out.append(Resolved(s, bool(s.on(cfg)), detail, after))
        prev = s.id
    return out
