"""Build the experiment board from the run files.

Usage: uv run python scripts/board.py
Reads results/experiments.tsv (one row per experiment: its run, what it is compared with, and the plain-words
change, verdict and reason, written by hand when the experiment ends), results/pipelines.tsv (the pipelines to
compare, each with its run), results/phases.tsv (the project phases and their state, by hand) and
results/runs/*.jsonl (every number).
Writes results/board/experiments.json, pipelines.json and phases.json (the data), and results/board.html (open it
in a browser). The page body is scripts/board_page.html; the same body and the three data files feed the hosted board.
Intervals: paired bootstrap, as in scripts/bootstrap_delta.py.
"""

import csv
import glob
import json
import random
from pathlib import Path

FMTS = ["docx", "pdf", "pptx", "xlsx", "csv", "slide_image", "image"]
LABEL = {"all": "All file types", "docx": "Word", "pdf": "PDF", "pptx": "PowerPoint", "xlsx": "Excel", "csv": "CSV",
         "slide_image": "Slide images", "image": "Scans and photos"}  # fmt: skip
# What the test counts as a hit: the labels of the public sets name a page for some formats and only a file for others.
UNIT = {"all": "page or file", "docx": "the right file", "pdf": "the right page", "pptx": "the right file",
        "xlsx": "the right file", "csv": "the right file", "slide_image": "the right slide", "image": "the right image"}  # fmt: skip
TOKENS = """--color-fg:{fg};--color-fg-muted:{mut};--color-bg:{bg};--color-panel:{panel};--color-border-line:{line};
--color-ok:{ok};--color-warn:{warn};--color-bad:{bad};--cds-chart-muted:{cm};--cds-chart-grid:{line};--cds-chart-axis:{mut};
--cds-chart-reference:{mut};--cds-surface-popover:{panel};--cds-border:{line};--cds-text-secondary:{mut};"""
LIGHT = dict(fg="#16202a", mut="#5b6876", bg="#f4f6f8", panel="#ffffff", line="#d3dae2", ok="#1c7a45", warn="#9a5b00",
             bad="#b3261e", cm="#b9c2cc")  # fmt: skip
DARK = dict(fg="#e6edf4", mut="#93a1af", bg="#0f151b", panel="#182029", line="#2f3c49", ok="#5fd08f", warn="#e2a04a",
            bad="#ff8a80", cm="#4a5766")  # fmt: skip
SIZES = ("--cds-radius:8px;--cds-pad-xs:4px;--cds-pad-sm:8px;--cds-pad-md:12px;--cds-pad-lg:16px;--cds-pad-xl:24px;"
         "--cds-gap-xs:4px;--cds-gap-sm:8px;--cds-gap-md:12px;--cds-gap-lg:16px;--cds-gap-xl:28px;"
         "--cds-font-size-caption:12.5px;--cds-font-size-body:14.5px;--cds-font-size-heading:17px;"
         "--cds-font-size-title:28px;--cds-font-weight-medium:600;--cds-shadow-popover:0 4px 16px rgba(0,0,0,.2);"
         "--font-sans:system-ui,-apple-system,sans-serif;--font-serif:Georgia,serif;")  # fmt: skip
# The hosted board gives the page a `dash` object; this stands in for it in the local file.
SHIM = ("const FNS=[];const P0={pipeline:'default',measure:'top10'};let P={...P0};const again=()=>FNS.forEach((f)=>f());"
        "window.dash={colors:['#3b82c4','#2f9e6b','#b7791f','#8a5cc2'],"
        "data:(id)=>({status:id in DATA?'ok':'missing',data:DATA[id]||[]}),onData:(fn)=>{FNS.push(fn);fn();},"
        "calc:(id,o)=>{DATA[id]=o.fn(...o.inputs.map((i)=>DATA[i]));},params:()=>({...P}),"
        "setParams:(o)=>{P={...P,...o};again();},resetParams:()=>{P={...P0};again();},setLink:()=>{}};"
        "addEventListener('resize',again);")  # fmt: skip


def load(name: str) -> dict | None:
    """A run by name (the latest file wins) or by its full run id."""
    exact = Path(f"results/runs/{name}.jsonl")
    fs = [str(exact)] if exact.exists() else sorted(glob.glob(f"results/runs/*-{name}.jsonl"))
    if not fs:
        return None
    with open(fs[-1]) as fh:
        return {d["qid"]: d for d in map(json.loads, fh) if d.get("metrics")}


def ci(d: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rnd = random.Random(seed)
    bs = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def hit(run: dict, q: str, k: int) -> float:
    return 100.0 if run[q]["metrics"]["mrr@10"] >= 1 / k - 1e-9 else 0.0


def rate(run: dict, ids: list[str], k: int) -> int:
    return round(sum(hit(run, q, k) for q in ids) / len(ids))


def measures(row: dict, key: str, b: dict, c: dict, ids: list[str]) -> None:
    """The change in nDCG@10 under `key`, and in 'right result first' and 'in the top 10' (percentage points)
    under key_first and key_top10: each with its paired bootstrap interval; the two rates also before and after."""
    for m, k, places in ((key, 0, 3), (key + "_first", 1, 1), (key + "_top10", 10, 1)):
        if not ids:
            row[m] = row[m + "_lo"] = row[m + "_hi"] = None
        elif k:
            row[m], row[m + "_lo"], row[m + "_hi"] = (
                round(x, places) for x in ci([hit(c, q, k) - hit(b, q, k) for q in ids])
            )
        else:
            d = [c[q]["metrics"]["ndcg@10"] - b[q]["metrics"]["ndcg@10"] for q in ids]
            row[m], row[m + "_lo"], row[m + "_hi"] = (round(x, places) for x in ci(d))
        if k:
            row[m + "_before"], row[m + "_after"] = (rate(b, ids, k), rate(c, ids, k)) if ids else (None, None)


def main() -> None:
    with open("results/experiments.tsv") as fh:
        reg = list(csv.DictReader(fh, delimiter="\t"))
    rows = []
    for i, e in enumerate(reg):
        b, c = load(e["compared_run"]), load(e["run"])
        if not b or not c:
            print(f"skipped {e['id']}: run file missing")
            continue
        ids = sorted(set(b) & set(c))
        row = {"order": i + 1, "id": e["id"], "change": e["change"], "compared_with": e["compared_with"],
               "verdict": e["verdict"], "why": e["why"], "seconds_added": int(float(e["seconds_added"] or 0)),
               "questions": len(ids)}  # fmt: skip
        measures(row, "all", b, c, ids)
        for f in FMTS:
            measures(row, f, b, c, [q for q in ids if b[q]["format"] == f])
        hb, hc = load(e["compared_hard_run"]), load(e["run"] + "_hard")
        measures(row, "hard", hb or {}, hc or {}, sorted(set(hb) & set(hc)) if hb and hc else [])
        rows.append(row)
    pipes = []
    with open("results/pipelines.tsv") as fh:
        for i, p in enumerate(csv.DictReader(fh, delimiter="\t")):
            d = load(p["run"])
            if not d:
                print(f"skipped pipeline {p['id']}: run file missing")
                continue
            for f in ["all", *FMTS]:
                ids = [q for q in d if f == "all" or d[q]["format"] == f]
                pipes.append({"pipeline": p["id"], "label": p["label"], "note": p["note"], "order": i + 1, "format": f,
                              "format_label": LABEL[f], "hit_means": UNIT[f], "questions": len(ids), "first_pct": rate(d, ids, 1),
                              "top3_pct": rate(d, ids, 3), "top10_pct": rate(d, ids, 10),
                              "ndcg": round(sum(d[q]["metrics"]["ndcg@10"] for q in ids) / len(ids), 3)})  # fmt: skip
    with open("results/phases.tsv") as fh:
        phases = [{"order": i + 1, **p} for i, p in enumerate(csv.DictReader(fh, delimiter="\t"))]
    out = Path("results/board")
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiments.json").write_text(json.dumps(rows))
    (out / "pipelines.json").write_text(json.dumps(pipes))
    (out / "phases.json").write_text(json.dumps(phases))
    data = json.dumps({"experiments": rows, "pipelines": pipes, "phases": phases}).replace("</", "<\\/")
    page = Path("scripts/board_page.html").read_text()
    Path("results/board.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>docsearch experiment board</title><style>:root{" + SIZES + TOKENS.format(**LIGHT) + "}"
        "@media (prefers-color-scheme: dark){:root{" + TOKENS.format(**DARK) + "color-scheme:dark}}"
        "body{margin:0;background:var(--color-bg)}#dash-root{max-width:1320px;margin:0 auto;padding:24px 20px 48px}</style>"
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js'></script><script>"
        "const DATA=" + data + ";" + SHIM + "</script></head><body><div id='dash-root'>" + page + "</div></body></html>"
    )  # fmt: skip
    print(
        f"board: {len(rows)} experiments, {len(pipes) // 8} pipelines, {len(phases)} phases -> results/board.html, results/board/*.json"
    )


if __name__ == "__main__":
    main()
