"""Static dark-mode HTML report: corpus, parsing, chunking, timings, and retrieval runs."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from datetime import datetime

from .chunk import VARIANTS, chunks_path
from .config import Config
from .eval import read_runs
from .manifest import read_manifest, read_questions

CSS = """
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#9da7b3;--acc:#58a6ff;--good:#3fb950;--bad:#f85149}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:1240px;margin:0 auto;padding:32px 24px 80px}h1{font-size:30px;margin:0 0 4px}h2{font-size:22px;margin:40px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}
p.mut,.mut{color:var(--mut)}table{border-collapse:collapse;width:100%;background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:8px 0 16px}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{background:#1f2630;color:var(--mut);font-weight:600;font-size:15px}
td:first-child,th:first-child{text-align:left}tr:last-child td{border-bottom:none}.best{color:var(--good);font-weight:700}.bar{display:inline-block;height:10px;background:var(--acc);border-radius:3px;vertical-align:middle;margin-left:8px}
code{background:#1f2630;padding:2px 6px;border-radius:4px;font-size:15px}.cards{display:flex;gap:12px;flex-wrap:wrap}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 16px;min-width:170px}.card b{display:block;font-size:26px}
"""


def _t(headers: list[str], rows: list[list], best_cols: set[int] | None = None) -> str:
    best: dict[int, float] = {}
    for c in best_cols or set():
        vals = [r[c] for r in rows if isinstance(r[c], float)]
        if vals:
            best[c] = max(vals)
    out = ["<table><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr>"]
    for r in rows:
        cells = []
        for i, v in enumerate(r):
            cls = ' class="best"' if i in best and isinstance(v, float) and v == best[i] else ""
            s = f"{v:.3f}" if isinstance(v, float) else html.escape(str(v))
            cells.append(f"<td{cls}>{s}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(out) + "</table>"


def build_report(cfg: Config) -> str:
    res_dir = cfg.path("results_dir")
    res_dir.mkdir(parents=True, exist_ok=True)
    parts = [f"<h1>docsearch — phase 1 report</h1><p class='mut'>Generated {datetime.now():%Y-%m-%d %H:%M}. "
             "Gold is page-level (ViDoRe, FinSlides, images) or file-level (OCB), so chunking variants are scored on the same targets.</p>"]  # fmt: skip

    # Corpus + questions
    man = cfg.path("corpus_dir") / "manifest.csv"
    docs = read_manifest(man) if man.exists() else []
    qpath = cfg.path("data_dir") / "questions.jsonl"
    qs = read_questions(qpath) if qpath.exists() else []
    if docs:
        agg: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])
        for d in docs:
            a = agg[(d.source, d.format)]
            a[0] += 1
            a[1] += d.pages
            a[2] += d.bytes
        qc = Counter((q.source, q.format) for q in qs)
        rows = [[s, f, n, p or "", f"{b / 1e6:.1f}", qc.get((s, f), 0)] for (s, f), (n, p, b) in sorted(agg.items())]
        rows.append(
            ["total", "", len(docs), sum(d.pages for d in docs), f"{sum(d.bytes for d in docs) / 1e6:.1f}", len(qs)]
        )
        parts.append(
            "<h2>Corpus and questions</h2>" + _t(["source", "format", "files", "pages", "MB", "questions"], rows)
        )
        lic = Counter((d.source, d.license) for d in docs)
        parts.append(_t(["source", "license", "files"], [[s, lic_name, n] for (s, lic_name), n in sorted(lic.items())]))

    # Parsing
    plog = cfg.path("parsed_dir") / "parse_log.csv"
    if plog.exists():
        with plog.open() as f:
            prow = list(csv.DictReader(f))
        g: dict[str, list[dict]] = defaultdict(list)
        for r in prow:
            g[r["format"]].append(r)
        rows = []
        for fmt, rs in sorted(g.items()):
            ok = [r for r in rs if r["status"] == "ok"]
            secs = [float(r["seconds"] or 0) for r in rs]
            pe = sum(int(r["pages_expected"] or 0) for r in rs)
            pw = sum(int(r["pages_with_text"] or 0) for r in rs)
            rows.append([fmt, len(rs), len(ok), len(rs) - len(ok), f"{len(ok) / len(rs):.0%}",
                         f"{pw / pe:.0%}" if pe else "n/a", f"{sum(secs):.0f}", f"{sum(secs) / len(rs):.2f}",
                         sum(int(r["tables"] or 0) for r in rs), f"{sum(int(r['chars'] or 0) for r in rs) / 1e6:.2f}"])  # fmt: skip
        parts.append("<h2>Parsing (Docling)</h2>" + _t(
            ["format", "files", "ok", "not ok", "success", "page coverage", "total s", "s/file", "tables", "M chars"], rows))  # fmt: skip
        bad = [r for r in prow if r["status"] != "ok"]
        if bad:
            parts.append(
                _t(
                    ["doc", "format", "status", "error"],
                    [[r["doc_id"], r["format"], r["status"], r["error"][:120]] for r in bad[:40]],
                )
            )

    # Chunking
    crow = []
    for v in VARIANTS:
        sp = chunks_path(cfg, v).with_suffix(".stats.json")
        if sp.exists():
            st = json.loads(sp.read_text())
            for fmt, b in sorted(st["by_format"].items()):
                crow.append(
                    [
                        v,
                        fmt,
                        b["docs"],
                        b["chunks"],
                        f"{b['chunks'] / max(b['docs'], 1):.1f}",
                        f"{b['tokens'] / max(b['chunks'], 1):.0f}",
                    ]
                )
            crow.append([v, "all", st["docs"], st["chunks"], f"{st['chunks'] / max(st['docs'], 1):.1f}", ""])
    if crow:
        parts.append(
            "<h2>Chunking</h2>" + _t(["variant", "format", "docs", "chunks", "chunks/doc", "avg tokens"], crow)
        )

    # Timings
    tp = res_dir / "timings.json"
    if tp.exists():
        tm = json.loads(tp.read_text())
        parts.append(
            "<h2>Timings</h2>"
            + _t(["step", "seconds", "minutes"], [[k, f"{v:.0f}", f"{v / 60:.1f}"] for k, v in tm.items()])
        )

    # Runs
    runs = read_runs(res_dir / "runs.csv")
    if runs:
        latest: dict[str, dict] = {}
        for r in runs:
            latest[r["name"]] = r
        rs = list(latest.values())
        parts.append("<h2>Retrieval runs (latest per name)</h2>")
        parts.append(_t(
            ["run", "chunking", "retrieval", "rerank", "queries", "recall@10", "nDCG@10", "MRR@10", "hit@10", "macro R@10 (formats)", "seconds"],
            [[r["name"], r["chunking"], r["retrieval"], r["rerank"], r["n_queries"], float(r["recall@10"]), float(r["ndcg@10"]),
              float(r["mrr@10"]), float(r["hit@10"]), float(r["macro_recall@10_by_format"]), r["seconds"]] for r in rs],
            best_cols={5, 6, 7, 8, 9}))  # fmt: skip
        for dim in ("by_format", "by_source"):
            keys = sorted({k for r in rs for k in r[dim]})
            for metric in ("recall@10", "ndcg@10"):
                rows = []
                for k in keys:
                    n = next((r[dim][k]["n"] for r in rs if k in r[dim]), 0)
                    rows.append([k, n] + [float(r[dim].get(k, {}).get(metric, 0.0)) for r in rs])
                parts.append(f"<h2>{metric} {dim.replace('_', ' ')}</h2>" + _t(
                    [dim.split("_")[1], "n"] + [r["name"] for r in rs], rows, best_cols=set(range(2, 2 + len(rs)))))  # fmt: skip
        parts.append(
            "<p class='mut'>Full history: <code>results/runs.csv</code>; per-query rankings: <code>results/runs/*.jsonl</code>.</p>"
        )
    doc = f"<!doctype html><html lang=en><head><meta charset=utf-8><title>docsearch report</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>"
    out = res_dir / "report.html"
    out.write_text(doc)
    return str(out)
