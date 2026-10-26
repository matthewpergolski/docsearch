"""Checks for scripts/verify.sh. Standard library only, so it reads the CLI's outputs, not its internals.

  verify_check.py sync   --db MANIFEST.sqlite --corpus DIR --source verify
  verify_check.py search --stdout SEARCH.txt --expect verify/handbook.docx [--label "TITLE | DATE"]
  verify_check.py json   --file SEARCH.json --stdout SEARCH.txt --expect verify/handbook.docx
  verify_check.py doc    --stdout DOC_SYNC.txt --db MANIFEST.sqlite --parse-log PARSE_LOG.csv --file SEARCH.json
                         --expect verify/handbook.doc --empty empty.doc
  verify_check.py meta   --file META.jsonl --expect META.expected.jsonl --label WHO_WROTE_IT
  verify_check.py filter --stdout SEARCH.txt --meta META.jsonl [--formats xlsx ...] [--after D] [--before D]
                         [--absent DOC_ID]
  verify_check.py eval   --results RESULTS_DIR --questions QUESTIONS.jsonl --run RUN_NAME [--mode plain]
  verify_check.py refusal --output STEP.txt --expect MESSAGE --root ROOT --since FILE_TOUCHED_BEFORE_THE_STEP
  verify_check.py isolation --repo REPO --since FILE_TOUCHED_AT_START
  verify_check.py api    --requests SERVER_LOG.jsonl --steps STEP.txt ... --results RESULTS_DIR --run RUN_NAME
                         --questions QUESTIONS.jsonl --guard GUARD_STEP.txt

Each check prints one `PASS <what>` or `FAIL <what>` line. The exit code is 1 when any check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

CHECKS: list[bool] = []


def check(ok: bool, what: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {what}", flush=True)
    CHECKS.append(ok)
    return ok


def check_sync(db: Path, corpus: Path, source: str) -> None:
    want = sorted(f"{source}/{p.relative_to(corpus).as_posix()}" for p in corpus.rglob("*")
                  if p.is_file() and not p.name.startswith(".") and p.name != "manifest.csv")  # fmt: skip
    if not check(db.exists(), f"sync: state DB exists at {db}"):
        return
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = {d: (s, e) for d, s, e in con.execute("SELECT doc_id, status, error FROM files")}
    finally:
        con.close()
    check(set(want) <= set(rows), f"sync: state DB lists all {len(want)} fixture files (it lists {len(rows)})")
    for d in want:
        status, err = rows.get(d, ("missing", ""))
        check(status == "ok", f"sync: {d} status {status}" + (f" ({err})" if err else ""))


GROUP = re.compile(r"^ ?(\d+)\. (.+)$")  # a result group's rank line; its page lines are indented 4 spaces


def groups(stdout: Path) -> list[tuple[str, str]]:
    """(file, header) per printed result group: the header (name, then " | " and the date or period when known) is
    on the rank line, the file on the line under it."""
    lines = stdout.read_text().splitlines() if stdout.exists() else []
    return [(lines[i + 1].strip(), m.group(2)) for i, m in enumerate(map(GROUP.match, lines))
            if m and i + 1 < len(lines) and lines[i + 1].startswith("    ")]  # fmt: skip


def check_search(stdout: Path, expect: str, label: str | None) -> None:
    found = groups(stdout)
    top = found[0] if found else (None, None)
    check(top[0] == expect, f"search: first result is {expect} (got {top[0]})")
    if label is not None:
        check(top[1] == label, f"search: first result shows {label!r} (got {top[1]!r})")
    files = [f for f, _ in found]
    check(bool(files) and len(files) == len(set(files)),
          f"search: each file is one result group ({len(files)} groups: {files})")  # fmt: skip


GROUP_KEYS = ["rank", "file", "format", "name", "date", "period", "pages"]
PAGE_KEYS = ["page", "rank", "title", "snippet", "score", "scorer"]


def check_json(file: Path, stdout: Path, expect: str) -> None:
    """The --json output of the same search: its shape, and the same groups in the same order as the text."""
    try:
        d = json.loads(file.read_text())
    except (OSError, ValueError) as e:
        check(False, f"json: {file.name} is one JSON object ({type(e).__name__})")
        return
    check(sorted(d) == ["answer", "chunk_ids", "confident", "groups", "query"],
          f"json: keys query, confident, answer, groups, chunk_ids (got {sorted(d)})")  # fmt: skip
    gs, ids = d.get("groups") or [], d.get("chunk_ids") or []
    check(bool(gs) and gs[0].get("file") == expect, f"json: first group is {expect} (got {gs[0].get('file') if gs else None})")  # fmt: skip
    check(all(list(g) == GROUP_KEYS for g in gs) and [g["rank"] for g in gs] == list(range(1, len(gs) + 1)),
          f"json: every group has {', '.join(GROUP_KEYS)}, ranked 1..{len(gs)}")  # fmt: skip
    pages = [p for g in gs for p in g["pages"]] if all("pages" in g for g in gs) else []
    check(bool(pages) and all(list(p) == PAGE_KEYS and p["scorer"] in ("rerank", "page", "fused")
                              and 1 <= p["rank"] <= len(ids) and not p["snippet"].startswith("File:") for p in pages),
          f"json: every page has {', '.join(PAGE_KEYS)}, a scorer, a rank in the hit list and no 'File:' snippet")  # fmt: skip
    check(d.get("answer") is None, "json: no answer without --agent")
    text = [f for f, _ in groups(stdout)]
    check([g.get("file") for g in gs] == text, f"json: the same files in the same order as {stdout.name} ({text})")


def check_doc(stdout: Path, db: Path, parse_log: Path, file: Path, expect: str, empty: str) -> None:
    """An old Word file (.doc) next to a 0-byte .doc: sync names the empty one and does not parse it, converts and
    parses the .doc, and the --json search finds it first."""
    text = stdout.read_text() if stdout.exists() else ""
    check("empty (0 bytes), skipped: 1" in text and bool(re.search(rf"^  empty +{re.escape(empty)}$", text, re.M)),
          f"doc: sync says 'empty (0 bytes), skipped: 1' and names {empty}")  # fmt: skip
    rows = {}
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = {d: (s, e) for d, s, e in con.execute("SELECT doc_id, status, error FROM files")}
        finally:
            con.close()
    status, err = rows.get(expect, ("missing", ""))
    check(status == "ok", f"doc: sync state of {expect} is ok (got {status}{f': {err}' if err else ''})")
    check(not any(d.endswith(empty) for d in rows), f"doc: {empty} is not in the sync state DB")
    log = {}
    if parse_log.exists():
        with parse_log.open(newline="") as f:
            log = {r["doc_id"]: r for r in csv.DictReader(f)}
    r = log.get(expect, {})
    check(r.get("status") == "ok" and int(r.get("chars") or 0) > 0,
          f"doc: parse log has {expect} status ok with text (status {r.get('status')}, chars {r.get('chars')})")  # fmt: skip
    try:
        gs = json.loads(file.read_text()).get("groups") or []
    except (OSError, ValueError) as e:
        check(False, f"doc: {file.name} is one JSON object ({type(e).__name__})")
        return
    check(bool(gs) and gs[0].get("file") == expect, f"doc: first group is {expect} (got {gs[0].get('file') if gs else None})")  # fmt: skip


def check_meta(file: Path, expect: Path, label: str) -> None:
    want = {r["doc_id"]: r for r in map(json.loads, expect.read_text().splitlines())}
    got = {r["doc_id"]: r for r in map(json.loads, file.read_text().splitlines())} if file.exists() else {}
    check(set(got) == set(want), f"meta ({label}): {file.name} has a row for each of the {len(want)} fixture files "
          f"(missing {sorted(set(want) - set(got))}, extra {sorted(set(got) - set(want))})")  # fmt: skip
    for doc_id in sorted(want):
        r = got.get(doc_id, {})
        diff = {k: r.get(k) for k in want[doc_id] if r.get(k) != want[doc_id][k]}
        check(not diff, f"meta ({label}): {doc_id} title {want[doc_id]['title']!r}, created "
              f"{want[doc_id]['created']!r}, modified {want[doc_id]['modified']!r}" + (f" (got {diff})" if diff else ""))  # fmt: skip


def check_filter(stdout: Path, meta: Path, formats: list[str], after: str | None, before: str | None,
                 absent: str | None) -> None:  # fmt: skip
    fmt = {r["doc_id"]: r["format"] for r in map(json.loads, meta.read_text().splitlines())} if meta.exists() else {}
    found = groups(stdout)
    if not check(bool(found), f"filter: {stdout.name} lists at least one result"):
        return
    if formats:
        bad = sorted({d for d, _ in found if fmt.get(d) not in formats})
        check(not bad, f"filter: every result is of format {', '.join(formats)}" + (f" (not: {bad})" if bad else ""))
    if after or before:
        dates = {d: head.split(" | ")[-1].split(", ")[-1] if " | " in head else "" for d, head in found}
        bad = sorted(d for d, x in dates.items()
                     if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", x) or (after and x < after) or (before and x > before))  # fmt: skip
        what = " and ".join(x for x in (after and f"on or after {after}", before and f"on or before {before}") if x)
        check(not bad, f"filter: every result shows a date {what}" + (f" (not: {[(d, dates[d]) for d in bad]})" if bad else ""))  # fmt: skip
        rows = [json.loads(x) for x in meta.read_text().splitlines()] if meta.exists() else []
        undated = sum(1 for r in rows if not (r.get("modified") or r.get("created")))
        said = [x for x in stdout.read_text().splitlines() if x.startswith("date filter: ")]
        check(bool(said) and said[0].startswith(f"date filter: {undated} file") if undated else not said,
              f"filter: search says it left out {undated} undated file(s) (said: {said[:1]})")  # fmt: skip
    if absent:
        check(absent not in {d for d, _ in found}, f"filter: {absent} is left out")


def check_eval(results: Path, questions: Path, run: str, mode: str) -> None:
    gold = {q["qid"]: q["gold"] for q in (json.loads(x) for x in questions.read_text().splitlines() if x.strip())}
    runs_csv = results / "runs.csv"
    rows = []
    if runs_csv.exists():
        with runs_csv.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["name"] == run]
    if not check(bool(rows), f"eval: run {run} is in {runs_csv}"):
        return
    row = rows[-1]
    check("box subset" not in row["notes"], f"eval: {row['run_id']} used every question (no box subset)")
    i = row["notes"].rfind("{")
    settings = json.loads(row["notes"][i:]) if i >= 0 else {}
    got_mode = settings.get("mode", "plain")
    check(got_mode == mode, f"eval: the run settings record mode {mode} (got {got_mode})")
    per_query = results / "runs" / f"{row['run_id']}.jsonl"
    if not check(per_query.exists(), f"eval: per-question file {per_query.name} exists"):
        return
    got = {r["qid"]: r for r in map(json.loads, per_query.read_text().splitlines())}
    check(set(got) == set(gold) and row["n_queries"] == str(len(gold)),
          f"eval: covers all {len(gold)} fixture questions (n_queries {row['n_queries']}, "
          f"missing {sorted(set(gold) - set(got))}, extra {sorted(set(got) - set(gold))})")  # fmt: skip
    for qid in sorted(gold):
        r = got.get(qid)
        if r is None:
            check(False, f"eval: {qid} has no result")
            continue
        mrr = r["metrics"].get("mrr@10", 0.0)
        rank = round(1 / mrr) if mrr else None
        where = f"rank {rank}" if rank else "not in the top 10"
        top = r["top"][0] if r["top"] else None
        check(rank == 1, f"eval: {qid} gold {', '.join(gold[qid])} at {where} (first: {top})")


EXIT_LINE = re.compile(r"^\[exit (\d+) after \d+s\]$", re.M)


def check_refusal(output: Path, expect: str, root: Path, since: Path) -> None:
    text = output.read_text() if output.exists() else ""
    codes = EXIT_LINE.findall(text)
    rc = int(codes[-1]) if codes else None
    check(rc not in (None, 0), f"refusal: the command exited non-zero (exit {rc})")
    check(expect in text, f"refusal: the output says {expect!r}")
    t = since.stat().st_mtime
    changed = [p for p in root.rglob("*") if not p.is_relative_to(root / "evidence") and p.lstat().st_mtime > t]
    names = ", ".join(str(p.relative_to(root)) for p in changed[:5])
    check(not changed, f"refusal: no run and no other file written under {root}"
          + (f" (changed: {names}{', ...' if len(changed) > 5 else ''})" if changed else ""))  # fmt: skip


LOCAL_LOAD = re.compile(r"^loading (embedder|reranker)\b", re.M)


def check_api(requests: Path, steps: list[Path], results: Path, run: str, questions: Path, guard: Path) -> None:
    """models.backend api: the server got both kinds of request, no step loaded a local embedder or text reranker,
    the eval run is recorded as api over every question, and a search on an index of another dimension stopped."""
    routes = [json.loads(x)["route"] for x in requests.read_text().splitlines()] if requests.exists() else []
    for route in ("/v1/embeddings", "/rerank"):
        check(route in routes, f"api: the server answered {route} ({routes.count(route)} requests)")
    for step in steps:
        text = step.read_text() if step.exists() else ""
        loads = sorted(set(LOCAL_LOAD.findall(text)))
        check(step.exists() and not loads, f"api: {step.stem} loaded no local embedder or text reranker"
              + (f" (loaded: {', '.join(loads)})" if loads else ""))  # fmt: skip
    n = sum(1 for x in questions.read_text().splitlines() if x.strip())
    rows = []
    if (results / "runs.csv").exists():
        with (results / "runs.csv").open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["name"] == run]
    got = (rows[-1]["device"], rows[-1]["n_queries"]) if rows else None
    check(got == ("api", str(n)), f"api: eval run {run} recorded device api over all {n} questions (got {got})")
    text = guard.read_text() if guard.exists() else ""
    codes = EXIT_LINE.findall(text)
    check(bool(codes) and codes[-1] != "0" and "was built with" in text and "dims" in text,
          f"api: a search on the local index with the api embedder stopped and named both models "
          f"(exit {codes[-1] if codes else None})")  # fmt: skip


PROTECTED = ("data", "corpus", "work", "results", "logs")


def check_web(file: Path, cli_json: Path, page: Path | None, opened: Path, source: Path, config: Path,
              stream: Path) -> None:  # fmt: skip
    """GET /api/search from `docsearch serve`: the CLI's --json results unchanged, plus notes and terms; the
    built page at / when it was built; GET /api/file of the first hit (the file itself, or the PDF copy that
    render made of it); /api/config saying the models are loaded; and GET /api/search/stream (what the page
    uses) ending with the same results."""
    try:
        d = json.loads(file.read_text())
    except (OSError, ValueError) as e:
        check(False, f"web: {file.name} is one JSON object ({type(e).__name__})")
        return
    check(sorted(d) == ["notes", "results", "terms"], f"web: keys notes, results, terms (got {sorted(d)})")
    cli = json.loads(cli_json.read_text()) if cli_json.exists() else None
    check(d.get("results") == cli, f"web: the API results equal `docsearch search --json` ({cli_json.name})")
    check(
        isinstance(d.get("terms"), list) and bool(d.get("terms")), f"web: the query terms are named ({d.get('terms')})"
    )
    if page is not None:
        html = page.read_text() if page.exists() else ""
        check('<div id="root">' in html and "/assets/" in html, "web: / serves the built page with its assets")
    body = opened.read_bytes() if opened.exists() else b""
    same = source.exists() and body == source.read_bytes()
    check(same or body.startswith(b"%PDF"), f"web: /api/file gives the first hit's file, {source.name} or its PDF copy "
          f"({'the file' if same else 'a PDF' if body.startswith(b'%PDF') else f'{len(body)} other bytes'})")  # fmt: skip
    try:
        ready = json.loads(config.read_text()).get("ready")
    except (OSError, ValueError):
        ready = None
    check(ready is True, f"web: /api/config says the models are loaded (ready: {ready})")
    try:
        events = [json.loads(line) for line in stream.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as e:
        events = []
        check(False, f"web: {stream.name} is lines of JSON ({type(e).__name__})")
    last = events[-1] if events else {}
    check(last.get("event") == "done" and last.get("results") == cli,
          f"web: /api/search/stream ends with done and the results of `docsearch search --json` "
          f"(events: {[e.get('event') for e in events]})")  # fmt: skip


def check_isolation(repo: Path, since: Path) -> None:
    t = since.stat().st_mtime
    for d in PROTECTED:
        top = repo / d
        changed = [p for p in (top, *top.rglob("*")) if p.lstat().st_mtime > t] if top.exists() else []
        names = ", ".join(str(p.relative_to(repo)) for p in changed[:5])
        check(not changed, f"isolation: nothing in the repo's {d}/ changed during the run"
              + (f" (changed: {names}{', ...' if len(changed) > 5 else ''})" if changed else ""))  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sync")
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--source", required=True)
    p = sub.add_parser("search")
    p.add_argument("--stdout", type=Path, required=True)
    p.add_argument("--expect", required=True)
    p.add_argument("--label", help="the rank line of the first result without its rank: title | date")
    p = sub.add_parser("json")
    p.add_argument("--file", type=Path, required=True, help="stdout of the same search with --json")
    p.add_argument("--stdout", type=Path, required=True, help="the text output of the same search")
    p.add_argument("--expect", required=True)
    p = sub.add_parser("doc")
    p.add_argument("--stdout", type=Path, required=True, help="the step file of the .doc sync")
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--parse-log", type=Path, required=True)
    p.add_argument("--file", type=Path, required=True, help="stdout of the --json search")
    p.add_argument("--expect", required=True)
    p.add_argument("--empty", required=True, help="the 0-byte file, relative to the synced folder")
    p = sub.add_parser("meta")
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--expect", type=Path, required=True)
    p.add_argument("--label", required=True, help="the command that wrote the file")
    p = sub.add_parser("filter")
    p.add_argument("--stdout", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--formats", nargs="+", default=[])
    p.add_argument("--after")
    p.add_argument("--before")
    p.add_argument("--absent", help="a doc id that must not be among the hits")
    p = sub.add_parser("eval")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--mode", default="plain", help="the eval mode the run must record (plain records none)")
    p = sub.add_parser("refusal")
    p.add_argument("--output", type=Path, required=True, help="the step file verify.sh wrote")
    p.add_argument("--expect", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--since", type=Path, required=True, help="a file touched just before the step")
    p = sub.add_parser("api")
    p.add_argument("--requests", type=Path, required=True, help="the request log of scripts/model_server.py")
    p.add_argument("--steps", type=Path, nargs="+", required=True, help="step files run with the api backend")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--guard", type=Path, required=True, help="the step file of the search that must stop")
    p = sub.add_parser("web")
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--cli-json", type=Path, required=True)
    p.add_argument("--page", type=Path)
    p.add_argument("--opened", type=Path, required=True, help="the body of GET /api/file for the first hit")
    p.add_argument("--source", type=Path, required=True, help="the fixture file of the first hit")
    p.add_argument("--config", type=Path, required=True, help="the body of GET /api/config after the search")
    p.add_argument("--stream", type=Path, required=True, help="the body of GET /api/search/stream, the same search")
    p = sub.add_parser("isolation")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--since", type=Path, required=True, help="a file touched when the run started")
    a = ap.parse_args(argv)
    if a.cmd == "sync":
        check_sync(a.db, a.corpus, a.source)
    elif a.cmd == "search":
        check_search(a.stdout, a.expect, a.label)
    elif a.cmd == "json":
        check_json(a.file, a.stdout, a.expect)
    elif a.cmd == "doc":
        check_doc(a.stdout, a.db, a.parse_log, a.file, a.expect, a.empty)
    elif a.cmd == "meta":
        check_meta(a.file, a.expect, a.label)
    elif a.cmd == "filter":
        check_filter(a.stdout, a.meta, a.formats, a.after, a.before, a.absent)
    elif a.cmd == "eval":
        check_eval(a.results, a.questions, a.run, a.mode)
    elif a.cmd == "refusal":
        check_refusal(a.output, a.expect, a.root, a.since)
    elif a.cmd == "api":
        check_api(a.requests, a.steps, a.results, a.run, a.questions, a.guard)
    elif a.cmd == "web":
        check_web(a.file, a.cli_json, a.page, a.opened, a.source, a.config, a.stream)
    else:
        check_isolation(a.repo, a.since)
    return 0 if CHECKS and all(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
