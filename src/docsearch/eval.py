"""Evaluation harness: page/file-level recall@10, nDCG@10, MRR@10 by format and source."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import ui
from .config import Config, pick_device
from .manifest import Question, ranked_units, read_questions
from .metrics import mean_metrics, score_query
from .search import Retriever, rerank_pipeline

K = 10
RUN_FIELDS = [
    "run_id", "timestamp", "name", "chunking", "retrieval", "rerank", "embedder", "reranker",
    "rerank_candidates", "rerank_max_length", "bm25_k", "vector_k", "rrf_k", "device",
    "n_queries", "recall@10", "ndcg@10", "mrr@10", "hit@10", "macro_recall@10_by_format",
    "by_format", "by_source", "seconds", "rerank_pairs_scored", "notes",
]  # fmt: skip

# Default phase-1 run plan: (name, chunking variant, retrieval mode, rerank)
DEFAULT_PLAN = [
    ("baseline_hybrid_rerank", "hybrid", "hybrid", True),
    ("baseline_hybrid_norerank", "hybrid", "hybrid", False),
    ("ablation_bm25_only", "hybrid", "bm25", False),
    ("ablation_vector_only", "hybrid", "vector", False),
    ("control_recursive_rerank", "recursive", "hybrid", True),
    ("control_recursive_norerank", "recursive", "hybrid", False),
]


class Mode(NamedTuple):
    switches: dict[tuple[str, str], tuple[str, ...]]  # the switch values this mode allows; one value is forced
    todo: str | None  # the phase that builds this layer; None = built


# The three modes every phase reports. A phase turns a mode on by setting its todo to None.
MODES: dict[str, Mode] = {
    "plain": Mode({("enrich", "llm"): ("off",), ("agent", "mode"): ("off",)}, None),
    "enrich": Mode({("enrich", "llm"): ("local", "api")}, "a later phase"),
    "agent": Mode({("agent", "mode"): ("on",)}, None),
}


def apply_mode(cfg: Config, name: str) -> None:
    """Set the LLM switches for an eval mode (plain: both off); stop on an unbuilt mode. The --abstain judge is a
    separate, explicit LLM opt-in that no mode turns on."""
    if name not in MODES:
        raise SystemExit(f"--mode must be one of {', '.join(MODES)}; got {name!r}")
    mode = MODES[name]
    if mode.todo:
        raise SystemExit(f"eval --mode {name}: the {name} layer is not built yet ({mode.todo} builds it); "
                         "nothing was run")  # fmt: skip
    for (section, key), allowed in mode.switches.items():
        if cfg[section][key] in allowed:
            continue
        if len(allowed) > 1:
            raise SystemExit(f"eval --mode {name} needs {section}.{key} set to {' or '.join(allowed)} "
                             f"(it is {cfg[section][key]})")  # fmt: skip
        cfg[section][key] = allowed[0]


def aggregate(per_query: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in per_query:
        groups[r[key]].append(r["metrics"])
    return {
        g: {"n": len(rows), **{m: round(v, 4) for m, v in mean_metrics(rows).items()}}
        for g, rows in sorted(groups.items())
    }


def query_vectors(cfg: Config, qs: list[Question]) -> np.ndarray:
    """Embed questions once; cached on disk by embedder (and server, for api) + question text hash."""
    from .models import embed_cache_tag, load_embedder

    h = hashlib.sha1((embed_cache_tag(cfg) + "\n".join(q.query for q in qs)).encode()).hexdigest()[:16]
    p = cfg.path("cache_dir") / f"qvec_{h}.npy"
    if p.exists():
        return np.load(p)
    emb = load_embedder(cfg)
    v = emb.queries([q.query for q in qs])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, v)
    del emb
    gc.collect()
    return v


def skip_missing(plan: list[tuple[str, str, str, bool]], tables: set[str]) -> list[tuple[str, str, str, bool]]:
    """Drop runs whose chunk variant has no LanceDB table (e.g. no recursive control on the Mac), one line each."""
    from .index import table_name

    keep = []
    for run in plan:
        if table_name(run[1]) in tables:
            keep.append(run)
        else:
            ui.say(f"skipped {run[0]}: no {run[1]} index (run chunk --variant {run[1]} + index --variant {run[1]})")
    return keep


def run_eval(
    cfg: Config,
    plan: list[tuple[str, str, str, bool]] | None = None,
    sources: list[str] | None = None,
    limit: int | None = None,
    notes: str = "",
    full: bool = False,
    embedder: str | None = None,
    visual: str | None = None,
    live: dict | None = None,
    questions: str = "labeled",
    reviewed_only: bool = False,
    judge=None,
    eval_mode: str = "plain",
) -> list[dict]:
    """judge: a judge.Judge; with it, each reranked query also gets an LLM ANSWERABLE/NOT verdict on its top 5."""
    """live: evaluate the version pinned by a sync pointer instead of the working index (used by sync --gate)."""
    plan = plan or DEFAULT_PLAN
    qs = []
    if questions in ("labeled", "all"):
        qs = read_questions(cfg.path("data_dir") / "questions.jsonl")
        qs = [q for q in qs if q.gold]  # labeled sets: retrieval questions only
    custom = (load_custom(cfg, reviewed_only, questions)
              if questions in ("custom", "all", "custom_para", "hard", "tables") else [])  # fmt: skip
    from .subset import load_subset

    sub = load_subset(cfg, full)
    if sub and qs:
        keep_q = set(sub["questions"])
        qs = [q for q in qs if q.qid in keep_q]
        notes = (notes + "; box subset").strip("; ")
    # Questions whose every gold doc was excluded by a sizing cap (not a parse failure) are dropped
    # and counted; parse failures stay in and count as misses.
    from .chunk import chunks_path
    from .manifest import unit_doc

    excluded: set[str] = set()
    for v in {variant for _, variant, _, _ in plan}:
        sp = chunks_path(cfg, v).with_suffix(".stats.json")
        if sp.exists():
            excluded |= set(json.loads(sp.read_text()).get("excluded_oversize", []))
    n0 = len(qs)
    qs = [q for q in qs if not all(unit_doc(u) in excluded for u in q.gold)]
    if n0 != len(qs):
        ui.info(f"[eval] dropped {n0 - len(qs)} questions whose gold docs were excluded as oversize", flush=True)
        notes = (notes + f"; dropped {n0 - len(qs)} q (oversize gold docs)").strip("; ")
    if custom:  # not part of the box subset; keep those whose gold docs are in the index (no-answer: all)
        ok_docs = _indexed_docs(cfg, {v for _, v, _, _ in plan})
        if visual:  # a document with no text chunk (a scanned PDF read without OCR) is still indexed by its pages
            from .visual import read_pages

            ok_docs |= {p["doc_id"] for p in read_pages(cfg)}
        kept = [q for q in custom if all(unit_doc(u) in ok_docs for u in q.gold)]
        if len(kept) != len(custom):
            ui.info(f"[eval] {len(custom) - len(kept)} custom questions skipped: gold docs not indexed")
        qs += kept
        prov = {}
        for q in kept:  # the question file names the reviewer (an assistant's name); the run notes do not
            unreviewed = q.meta.get("status") == "unreviewed"
            who = "reviewer" if q.meta.get("reviewer") else "unreviewed" if unreviewed else "human"
            prov[who] = prov.get(who, 0) + 1
        notes = (notes + f"; questions={questions}" + ("(reviewed)" if reviewed_only else "")
                 + f"; reviewed_by={json.dumps(prov, sort_keys=True)}").strip("; ")  # fmt: skip
        ui.say("[eval] custom questions by reviewer: " + ", ".join(f"{k} {v}" for k, v in sorted(prov.items())))
    if sources:
        qs = [q for q in qs if q.source in sources]
    if limit:
        qs = qs[:limit]
    t0 = time.time()
    if embedder:
        from .altembed import query_embeddings

        qvecs = query_embeddings(cfg, embedder, [q.query for q in qs])
    else:
        qvecs = query_vectors(cfg, qs)
    vqs = None
    if visual:
        from .altembed import query_embeddings

        vqs = query_embeddings(cfg, visual, [q.query for q in qs])
    ui.info(f"[eval] {len(qs)} queries embedded/loaded in {time.time() - t0:.0f}s", flush=True)
    reranker = page_rr_model = None
    cand_cache: dict[tuple[str, str], list[list[dict]]] = {}
    retrievers: dict[tuple[str, str], Retriever] = {}
    results = []
    from .index import connect

    plan = skip_missing(plan, set(connect(cfg).table_names()))
    for name, variant, mode, use_rr in plan:
        t1 = time.time()
        key = (variant, mode)
        if key not in cand_cache:
            r = retrievers[key] = Retriever(cfg, variant, embedder, visual if "visual" in mode else None, live)
            cand_cache[key] = [
                r.candidates(q.query, qvecs[i] if mode != "bm25" else None, mode, vqs[i] if vqs else None)
                for i, q in enumerate(qs)
            ]
        sc = Retriever(cfg, variant).sc
        pairs_before = 0
        if use_rr and reranker is None:
            from .models import load_reranker

            reranker = load_reranker(cfg)
        if use_rr:
            pairs_before = reranker.calls
        page_rr = None
        if use_rr and sc.page_reranker and "visual" in mode:  # it needs the rendered pages, as in search()
            if page_rr_model is None:
                from .models import PageReranker

                page_rr_model = PageReranker(cfg, sc.page_reranker)
            page_rr = page_rr_model
        cells = None
        if use_rr and sc.table_lookup:
            from .cells import CellLookup, cells_path

            cells = CellLookup(cells_path(cfg))
        per_query = []
        from .search import NUMBER_BOOST_STATS, TABLE_LOOKUP_STATS

        NUMBER_BOOST_STATS.clear()
        TABLE_LOOKUP_STATS.clear()
        pb = ui.progress(len(qs), name, "q ").start() if use_rr else None
        ranked = []
        for i, q in enumerate(qs):
            if pb:
                pb.advance()
            hits = cand_cache[key][i]
            if use_rr:
                hits = rerank_pipeline(q.query, hits, reranker, sc, page_rr, cells)
                if (i + 1) % 25 == 0:
                    el = time.time() - t1
                    ui.info(f"[eval] {name}: {i + 1}/{len(qs)} reranked ({el:.0f}s)", flush=True)
            ranked.append(hits)
        if pb:
            pb.close()
        steps = agent_stats = None
        if eval_mode == "agent" and use_rr:
            search_many = (_plain_search(cfg, retrievers[key], mode, visual, reranker, page_rr, cells)
                           if cfg["agent"]["research"] == "on" else None)  # fmt: skip
            ranked, steps, agent_stats = _agent_step(cfg, retrievers[key], [q.query for q in qs], ranked, search_many)
        for i, q in enumerate(qs):
            hits = ranked[i]
            units = ranked_units(hits[:200], q.level)
            per_query.append(
                {
                    "qid": q.qid,
                    "source": q.source,
                    "format": q.format,
                    "level": q.level,
                    "type": q.meta.get("type", "labeled"),
                    "metrics": score_query(units, q.gold, K) if q.gold else {},
                    "top": units[:K],
                    "gold": list(q.gold),
                    "top_score": _top_score(hits, use_rr),
                    **({"agent": steps[i].row()} if steps else {}),
                }
            )
            gate = getattr(judge, "gate", None)
            ts = per_query[-1]["top_score"]
            if judge is not None and use_rr and (gate is None or ts is None or ts < gate):
                j = judge(q.query, hits)
                by_cid = {h["chunk_id"]: h for h in hits[:10]}
                per_query[-1]["judge"] = j["answerable"]
                per_query[-1]["judge_units"] = [u for c in j["chunk_ids"] if c in by_cid
                                                for u in ranked_units([by_cid[c]], q.level)]  # fmt: skip
        answerable = [r for r in per_query if r["gold"]]
        res = _summarise(cfg, name, variant, mode, use_rr, answerable, time.time() - t1, notes, embedder)
        if any(r["type"] != "labeled" for r in per_query):
            extra = {"by_type": aggregate(answerable, "type"),
                     "abstention": abstention(per_query, cfg["search"].get("abstain_threshold"))}  # fmt: skip
            if judge is not None and use_rr:
                from .judge import judge_abstention

                if judge.gate is None:
                    extra["abstention_judge"] = judge_abstention(per_query)
                else:
                    from .judge import gated

                    g = gated(per_query, judge.gate)
                    extra["abstention_judge"] = {
                        **g,
                        "failed_calls": g["missing_verdicts"],
                        "answerable_cites_gold": None,
                    }
                extra["abstention_judge"]["model"] = (f"{judge.client.s.provider}:{judge.client.s.model or 'default'}"
                                                      + (f" gated>={judge.gate}" if judge.gate is not None else ""))  # fmt: skip
            res["by_type"], res["abstention"] = extra["by_type"], extra["abstention"]
            res["notes"] = (res["notes"] + "; " + json.dumps(extra)).strip("; ")
            ui.say(format_custom(name, extra))
        if "visual" in mode:
            vid = cfg["models"]["embedders"][visual]["id"]
            res["embedder"] = vid if mode == "visual" else f"{res['embedder']} + {vid}"
            blob = {
                **setting_blob(cfg, sc, page_rr.name if page_rr else "", eval_mode),
                **({"number_boost_stats": dict(NUMBER_BOOST_STATS)} if sc.number_boost else {}),
                **({"table_lookup_stats": dict(TABLE_LOOKUP_STATS)} if cells is not None else {}),
                **({"page_pairs_scored": page_rr.calls} if page_rr else {}),
                **({"agent_stats": agent_stats} if agent_stats else {}),
            }
            res["notes"] = (res["notes"] + "; " + json.dumps(blob)).strip("; ")
        res["rerank_pairs_scored"] = (reranker.calls - pairs_before) if use_rr else 0
        _write_run(cfg, res, per_query)
        results.append(res)
        ui.info(
            f"[eval] {name}: recall@10={res['recall@10']:.3f} ndcg@10={res['ndcg@10']:.3f} "
            f"mrr@10={res['mrr@10']:.3f} ({res['seconds']}s)",
            flush=True,
        )
    return results


def _plain_search(cfg: Config, r: Retriever, mode: str, visual: str | None, reranker, page_rr, cells):
    """search_many for agent.research (D1): this run's plain pipeline for new query texts, in one batch; query
    vectors are cached by text, as for the questions. The rewrites leave the run's number-boost and table-lookup
    counts as they were, so those counts cover the questions only."""
    from .search import NUMBER_BOOST_STATS, TABLE_LOOKUP_STATS

    def search_many(texts: list[str]) -> list[list[dict]]:
        qv = query_vectors(cfg, [Question(f"rewrite:{i}", t, "", "", "page", {}) for i, t in enumerate(texts)])
        vq = None
        if visual and "visual" in mode:
            from .altembed import query_embeddings

            vq = query_embeddings(cfg, visual, texts)
        kept = [(c, dict(c)) for c in (NUMBER_BOOST_STATS, TABLE_LOOKUP_STATS)]
        try:
            return [rerank_pipeline(t, r.candidates(t, qv[i], mode, vq[i] if vq is not None else None), reranker,
                                    r.sc, page_rr, cells) for i, t in enumerate(texts)]  # fmt: skip
        finally:
            for c, before in kept:
                c.clear()
                c.update(before)

    return search_many


def _agent_step(cfg: Config, r: Retriever, queries: list[str], ranked: list[list[dict]], search_many=None):
    """+agent after the plain pipeline: (new ranked lists, one AgentStep per query, their counts). search_many:
    agent.research (D1) on, the plain search for the rewrites."""
    from . import agent
    from .llm import LLMUnavailable

    rows = getattr(r, "_rows", None) or {x["chunk_id"]: x for x in r.tbl.search().limit(r.tbl.count_rows()).to_list()}
    ag = agent.from_cfg(cfg, rows.values())
    t0 = time.time()
    try:
        out = ag.run_all(queries, ranked, int(cfg["agent"]["workers"]), lambda m: ui.info(m, flush=True))
        if search_many is not None:
            out = ag.research_all(queries, ranked, out, search_many, int(cfg["agent"]["workers"]),
                                  lambda m: ui.info(m, flush=True))  # fmt: skip
    except LLMUnavailable as e:
        raise SystemExit(f"eval --mode agent stopped, no run written: {e}. Replies so far are cached in "
                         f"{ag.cache_dir}; run the same command again to resume.") from None  # fmt: skip
    steps = [s for _, s in out]
    st = agent.summarize(steps)
    ui.say(f"[agent] {ag.model}: {st['gate_passed']}/{st['queries']} queries passed the gate (top {ag.gate_k} pages "
           f"hold 2+ pages of one file); {st['llm_calls']} calls, {st['cache_hits']} cached, {st['chosen']} pages "
           f"chosen ({st['moved']} moved up), {st['no_page']} no page, {st['malformed']} malformed, {st['errors']} "
           f"errors; {st['llm_seconds']}s of LLM time, {time.time() - t0:.0f}s wall"
           + (f"; {st['deck_gate_passed']} passed only as a deck (top {agent.DECK_PAGES} pages read)"
              if "deck_gate_passed" in st else ""))  # fmt: skip
    if "research" in st:
        rs = st["research"]
        ui.say(f"[agent] research: {rs['researched']} queries with no page got rewrites; {rs['llm_calls']} calls, "
               f"{rs['cache_hits']} cached, {rs['new_pages']} new pages in the merged top pages, {rs['chosen']} pages "
               f"chosen ({rs['moved']} moved up), {rs['no_page']} no page, {rs['errors']} errors; "
               f"{rs['llm_seconds']}s of LLM time")  # fmt: skip
    return [h for h, _ in out], steps, st


def _top_score(hits: list[dict], use_rr: bool) -> float | None:
    if not hits:
        return None
    if use_rr:  # best reranker score (slides skipped by rerank_skip_formats carry none)
        rs = [h["rerank_score"] for h in hits if h.get("rerank_score") is not None]
        return float(max(rs)) if rs else None
    v = hits[0].get("score")
    return None if v is None else float(v)


def abstention(per_query: list[dict], threshold: float | None = None, pct: float = 10.0) -> dict:
    """No-answer questions abstain when the top score is below a threshold. Default threshold: the
    pct-th percentile of top scores on answerable questions of the same run (so ~90% of those would be
    answered); set eval.abstain_threshold to fix it. Also reports the false-abstain rate on answerables."""
    ans = [r["top_score"] for r in per_query if r["gold"] and r.get("top_score") is not None]
    none = [r for r in per_query if not r["gold"]]
    if not none:
        return {}
    thr = float(threshold) if threshold is not None else (float(np.percentile(ans, pct)) if ans else 0.0)
    sc = [r["top_score"] for r in none if r.get("top_score") is not None]
    abst = [s < thr for s in sc] + [True] * (len(none) - len(sc))  # nothing retrieved = abstain
    return {"n": len(none), "threshold": round(thr, 4), "threshold_rule": "fixed" if threshold is not None
            else f"p{pct:g} of answerable top scores", "abstain_rate": round(sum(abst) / len(abst), 4),
            "false_abstain_rate": round(sum(s < thr for s in ans) / len(ans), 4) if ans else None,
            "median_top_score_no_answer": round(float(np.median(sc)), 4) if sc else None,
            "median_top_score_answerable": round(float(np.median(ans)), 4) if ans else None}  # fmt: skip


def format_custom(name: str, extra: dict) -> str:
    lines = [f"{name} by question type:"]
    for t, m in sorted(extra["by_type"].items()):
        lines.append(f"  {t:10s} n={m.get('n', '?'):>4}  recall@10 {m['recall@10']:.3f}  all@10 {m['all@10']:.3f}"
                     f"  nDCG@10 {m['ndcg@10']:.3f}")  # fmt: skip
    ab = extra.get("abstention")
    jd = extra.get("abstention_judge")
    if ab and jd:
        f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(f"  abstention  {'method':28s} {'abstain (no_answer)':>20s} {'false-abstain (answerable)':>27s}")
        lines.append(
            f"              {'threshold < ' + str(ab['threshold']):28s} {f(ab['abstain_rate']):>20s} "
            f"{f(ab['false_abstain_rate']):>27s}"
        )
        lines.append(
            f"              {'judge ' + jd['model']:28s} {f(jd['abstain_rate']):>20s} "
            f"{f(jd['false_abstain_rate']):>27s}"
        )
        lines.append(f"              judge: {jd['failed_calls']} failed calls; answerable verdicts citing a gold page: "
                     f"{f(jd['answerable_cites_gold'])}")  # fmt: skip
        return "\n".join(lines)
    if ab:
        lines.append(f"  no_answer  n={ab['n']:>4}  abstain {ab['abstain_rate']:.3f} at top score < {ab['threshold']}"
                     f" ({ab['threshold_rule']}; answerable false-abstain {ab['false_abstain_rate']})")  # fmt: skip
    return "\n".join(lines)


def load_custom(cfg: Config, reviewed_only: bool = False, which: str = "custom") -> list[Question]:
    from .hard import hard_path
    from .questions import custom_path, load_for_eval, para_path

    path = {"custom_para": para_path, "hard": hard_path, "tables": tables_path}.get(which, custom_path)(cfg)
    if not path.exists():
        if which == "tables":
            raise SystemExit(
                f"no table questions at {path}; they come from Phase C3 (scripts/draft_table_questions.py)"
            )
        raise SystemExit(f"no custom questions at {path}; run `docsearch questions draft` first")
    return [Question(**{k: r[k] for k in ("qid", "query", "source", "format", "level", "gold", "meta")})
            for r in load_for_eval(path, reviewed_only)]  # fmt: skip


def tables_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "questions_tables.jsonl"


def _indexed_docs(cfg: Config, variants: set[str]) -> set[str]:
    from .chunk import chunks_path

    docs: set[str] = set()
    for v in variants:
        p = chunks_path(cfg, v)
        if p.exists():
            for line in p.open():
                docs.add(json.loads(line)["doc_id"])
    return docs


def _model_label(model: str) -> str:
    """A model id as runs.csv records it, never a local path: an HF-cache snapshot path (/ or \\ separators) becomes
    repo@revision, and any other absolute folder (Unix, Windows drive or UNC) becomes the folder's name."""
    m = re.search(r"models--([^/\\]+)--([^/\\]+)[/\\]snapshots[/\\]([0-9a-f]+)", model)
    if m:
        return f"{m[1]}/{m[2]}@{m[3][:12]}"
    if re.match(r"(/|[A-Za-z]:[/\\]|\\\\)", model):
        return re.split(r"[/\\]", model.rstrip("/\\"))[-1] or model
    return model


def _summarise(cfg, name, variant, mode, use_rr, per_query, seconds, notes, embedder=None) -> dict:
    overall = mean_metrics([r["metrics"] for r in per_query])
    by_format = aggregate(per_query, "format")
    by_source = aggregate(per_query, "source")
    s, m = cfg["search"], cfg["models"]
    return {
        "run_id": f"{datetime.now():%Y%m%d-%H%M%S}-{name}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "chunking": variant,
        "retrieval": mode,
        "rerank": use_rr,
        "embedder": _model_label(cfg["models"]["embedders"][embedder]["id"] if embedder else m["embedder"])
        if mode != "bm25"
        else "",
        "reranker": m["reranker"] if use_rr else "",
        "rerank_candidates": s["rerank_candidates"] if use_rr else "",
        "rerank_max_length": m["rerank_max_length"] if use_rr else "",
        "bm25_k": s["bm25_k"] if mode != "vector" else "",
        "vector_k": s["vector_k"] if mode != "bm25" else "",
        "rrf_k": s["rrf_k"] if mode == "hybrid" else "",
        "device": "api" if m.get("backend") == "api" else pick_device(m["device"]),
        "n_queries": len(per_query),
        **{k: round(v, 4) for k, v in overall.items()},
        "macro_recall@10_by_format": round(
            float(np.mean([v["recall@10"] for v in by_format.values()])) if by_format else 0.0, 4
        ),
        "by_format": by_format,
        "by_source": by_source,
        "seconds": round(seconds, 1),
        "notes": notes,
    }


def _write_run(cfg: Config, res: dict, per_query: list[dict]) -> None:
    out = cfg.path("results_dir")
    (out / "runs").mkdir(parents=True, exist_ok=True)
    path = out / "runs.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if new:
            w.writeheader()
        w.writerow(
            {k: (json.dumps(res.get(k)) if isinstance(res.get(k), dict) else res.get(k, "")) for k in RUN_FIELDS}
        )
    with (out / "runs" / f"{res['run_id']}.jsonl").open("w") as f:
        for r in per_query:
            f.write(json.dumps(r) + "\n")


def _notes_settings(notes: str) -> dict:
    """The JSON settings blob eval appends to notes for visual runs ({} if none): the object that ends the notes."""
    dec = json.JSONDecoder()
    for i in (i for i, ch in enumerate(notes) if ch == "{"):
        try:
            blob, end = dec.raw_decode(notes, i)
        except ValueError:
            continue
        if end == len(notes.rstrip()) and isinstance(blob, dict):
            return blob
    return {}


# M28 (adopted 2026-10-13): the hybrid chunker shares a spreadsheet's chunk cap over its tables. Runs from before
# it carry no such key, so they never count as a reference for a run after it.
SHEET_CAP = "shared"


def setting_blob(cfg: Config, sc, page_reranker: str, mode: str = "plain") -> dict:
    """The search settings a visual run records in its notes (page_reranker: the model id, "" = off). A plain run
    names no mode, so runs from before the modes still match it; an agent run also names its LLM settings."""
    v = cfg["visual"]
    agent = {}
    if mode == "agent":
        from .agent import settings_of

        agent = {"agent": settings_of(cfg)}
    return {**{k: v[k] for k in ("dpi", "max_side", "office_max_pages", "visual_k")},
            "rerank_min_chars": sc.rerank_min_chars,
            **({"rerank_skip_formats": list(sc.rerank_skip_formats)} if sc.rerank_skip_formats else {}),
            **({"distinct_boost": sc.distinct_boost} if sc.distinct_boost else {}),
            **({"rerank_context": True} if sc.rerank_context else {}),
            **({"late_visual_formats": list(sc.late_visual_formats)} if sc.late_visual_formats else {}),
            **({"number_boost": sc.number_boost} if sc.number_boost else {}),
            **({"rerank_unit_chunks": sc.rerank_unit_chunks} if sc.rerank_unit_chunks else {}),
            **({"visual_weights": dict(sc.visual_weights)} if sc.visual_weights else {}),
            **({"period_demote": True} if sc.period_demote else {}),
            **({"table_lookup": True} if sc.table_lookup else {}),
            **({"page_reranker": page_reranker, "page_rerank_pages": sc.page_rerank_pages,
                "page_rerank_formats": list(sc.page_rerank_formats)} if page_reranker else {}),
            **({"visual_blind_formats": list(v["blind_formats"])} if v.get("blind_formats") else {}),
            **({"text_rich_chars": int(v["text_rich_chars"]), "text_rich_formats": list(v.get("text_rich_formats") or [])}
               if v.get("text_rich_chars") else {}),
            **({"office_shared_pages": int(v["office_shared_pages"]),
                "office_own_formats": list(v.get("office_own_formats") or ["docx", "pptx"])}
               if v.get("office_shared_pages") else {}),
            "sheet_cap": SHEET_CAP,
            **({"mode": mode} if mode != "plain" else {}),
            **agent}  # fmt: skip


# Per-run counts in the blob, not settings.
_BLOB_STATS = ("number_boost_stats", "page_pairs_scored", "table_lookup_stats", "agent_stats")


def run_settings(cfg: Config, mode: str = "plain") -> dict:
    """Settings a format-aware reference must share with the current config to count as the same system. The mode
    is the eval mode the run was made in (the one-change reference is always plain)."""
    from .search import SearchConfig

    s, m = cfg["search"], cfg["models"]
    sc = SearchConfig.from_cfg(cfg)
    pr = (m.get("page_rerankers") or {}).get(sc.page_reranker, sc.page_reranker) if sc.page_reranker else ""
    blob = json.loads(json.dumps(setting_blob(cfg, sc, pr, mode)))
    return {"rerank_candidates": str(s["rerank_candidates"]), "rerank_max_length": str(m["rerank_max_length"]),
            "reranker": m["reranker"], "blob": blob}  # fmt: skip


def cached_reference(cfg: Config, full: bool, variant: str, visual_id: str, mode: str = "plain") -> dict | None:
    """Latest format-aware + rerank run (any name) with the current settings and eval mode on the same question
    scope (full corpus vs box subset) whose per-query file exists, so a paired bootstrap needs no re-run."""
    want = run_settings(cfg, mode)

    def same(r: dict) -> bool:
        n = _notes_settings(r.get("notes") or "")
        got = {"rerank_candidates": str(r.get("rerank_candidates")), "rerank_max_length": str(r.get("rerank_max_length")),
               "reranker": r.get("reranker"), "blob": {k: v for k, v in n.items() if k not in _BLOB_STATS}}  # fmt: skip
        return got == want

    rows = [r for r in read_runs(cfg.path("results_dir") / "runs.csv")
            if r.get("retrieval") == "hybrid_visual_fa" and r.get("rerank") == "True" and r.get("chunking") == variant
            and visual_id in (r.get("embedder") or "")
            and ("box subset" in (r.get("notes") or "")) == (not full) and same(r)
            and "questions=" not in (r.get("notes") or "")
            and (cfg.path("results_dir") / "runs" / f"{r['run_id']}.jsonl").exists()]  # fmt: skip
    return rows[-1] if rows else None


def read_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("by_format", "by_source"):
            r[k] = json.loads(r[k]) if r.get(k) else {}
    return rows
