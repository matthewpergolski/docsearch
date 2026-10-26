"""`docsearch diagram`: Mermaid drawings of the ingest, query and eval flows from pipeline.py and the config.

A stage that is off in the config is drawn with a dashed border and dashed arrows. Off experiment steps share one
dashed box. `--write` puts each drawing between its markers in the Markdown files:

    <!-- diagram:ingest -->
    ...generated, do not edit...
    <!-- /diagram -->

`--check` writes nothing and exits 1 when a file differs from the generated text (tests/test_diagram.py does
the same check).
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

from .config import Config
from .pipeline import AGENT, ENRICH, FLOWS, Resolved, resolve

DOCS = ("README.md", "REPORT.md", "DEPLOYMENT.md")
DIRECTION = {"ingest": "TB", "query": "TB", "eval": "LR"}
MARKER = re.compile(r"(<!-- diagram:(\w+) -->\n)(.*?)(<!-- /diagram -->)", re.S)
LEGEND = ("Solid: on in docsearch.yaml. Dashed: off, or an option. Orange border: +enrich (an LLM at index "
          "time). Purple border: +agent (an LLM after search). Run `docsearch diagram` to draw your own config.")  # fmt: skip
# One class per node, and no opacity: Safari draws the label of a node with opacity outside its box.
CLASSES = {"off": "stroke:#8b949e,stroke-dasharray:5 5",
           "enrich": "stroke:#d97706,stroke-width:2px",
           "agent": "stroke:#7c3aed,stroke-width:2px",
           "off_enrich": "stroke:#d97706,stroke-width:2px,stroke-dasharray:5 5",
           "off_agent": "stroke:#7c3aed,stroke-width:2px,stroke-dasharray:5 5"}  # fmt: skip
LAYER_CLASS = {ENRICH: "enrich", AGENT: "agent"}
WRAP = 26, 32  # characters per line: a label (16 px, mermaid wraps it at 200 px), small text


def _text(s: str, width: int = WRAP[0]) -> str:
    """s broken into short lines: GitHub on a phone can draw a long line outside its box."""
    return "<br/>".join(textwrap.wrap(s, width, break_long_words=False, break_on_hyphens=False)).replace('"', "#quot;")


def _node(r: Resolved, n: str) -> str:
    label = _text(r.stage.label) + (f"<br/><small>{_text(r.detail, WRAP[1])}</small>" if r.detail else "")
    return f'{n}[("{label}")]' if r.stage.store else f'{n}["{label}"]'


def mermaid(cfg: Config, flow: str, generic: bool = False) -> str:
    """generic: the drawing for the public docs, which names the settings that pick an LLM, not the model."""
    rows = resolve(cfg, flow, generic)
    by_id = {r.stage.id: r for r in rows}
    folded = {r.stage.id for r in rows if r.stage.experiment and not r.on}

    def feeds(sid: str) -> list[str]:  # a folded experiment step passes its own inputs on
        return [x for a in by_id[sid].after for x in (feeds(a) if a in folded else [a])]

    def n(sid: str) -> str:  # ids carry the flow name: one page can hold all three drawings
        return f"{flow}_{sid}"

    out = [f"flowchart {DIRECTION[flow]}"]
    for r in rows:
        if r.stage.id not in folded:
            out.append("  " + _node(r, n(r.stage.id)))
    if folded:
        names = ", ".join(by_id[s].stage.label.lower() for s in sorted(folded, key=list(by_id).index))
        out.append(f'  {n("exp")}["{_text("Experiments, off: " + names)}"]')
    seen: set[tuple[str, str]] = set()
    for r in rows:
        if r.stage.id in folded:
            continue
        for a in dict.fromkeys(x for p in r.after for x in (feeds(p) if p in folded else [p])):
            if (a, r.stage.id) in seen:
                continue
            seen.add((a, r.stage.id))
            arrow = "-->" if by_id[a].on and r.on else "-.->"
            out.append(f"  {n(a)} {arrow} {n(r.stage.id)}")
    if folded:
        first = min(folded, key=list(by_id).index)
        out.append(f"  {n(feeds(first)[0])} -.- {n('exp')}")
    out += [f"  classDef {cls} {style}" for cls, style in CLASSES.items()]
    groups: dict[str, list[str]] = {"off": []}
    for r in rows:  # one class per node: off_enrich, not off and enrich
        cls = "_".join(c for c in (None if r.on else "off", LAYER_CLASS.get(r.stage.layer)) if c)
        if cls and r.stage.id not in folded:
            groups.setdefault(cls, []).append(r.stage.id)
    groups["off"] += ["exp"] if folded else []
    for cls in CLASSES:
        if groups.get(cls):
            out.append(f"  class {','.join(map(n, groups[cls]))} {cls}")
    return "\n".join(out) + "\n"


def block(cfg: Config, flow: str) -> str:
    """The text between the markers (generic: no LLM model names in the docs)."""
    return f"```mermaid\n{mermaid(cfg, flow, generic=True)}```\n\n{LEGEND}\n"


def splice(text: str, cfg: Config) -> str:
    """text with every marked block replaced by the generated one. An unknown flow name is an error."""

    def sub(m: re.Match) -> str:
        if m.group(2) not in FLOWS:
            raise ValueError(f"unknown diagram {m.group(2)!r}; choose from {', '.join(FLOWS)}")
        return m.group(1) + block(cfg, m.group(2)) + m.group(4)

    return MARKER.sub(sub, text)


def stale(root: Path, cfg: Config) -> list[str]:
    """The Markdown files whose marked blocks differ from the generated text."""
    return [n for n in DOCS if (root / n).exists() and splice((root / n).read_text(), cfg) != (root / n).read_text()]


def write(root: Path, cfg: Config) -> list[str]:
    changed = stale(root, cfg)
    for n in changed:
        (root / n).write_text(splice((root / n).read_text(), cfg))
    return changed


TITLES = {"ingest": "Ingestion (docsearch sync)", "query": "One query (docsearch search)", "eval": "Evaluation"}
MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs"  # the only file from the web


def _stage_table(cfg: Config, flow: str) -> str:
    from html import escape

    rows = "".join(
        f"<tr class={'on' if r.on else 'off'}><td>{escape(r.stage.label)}</td><td>{'on' if r.on else 'off'}</td>"
        f"<td>{escape(r.stage.layer)}</td><td><code>{escape(r.stage.ref)}</code></td>"
        f"<td>{escape(', '.join(r.stage.keys))}</td></tr>"
        for r in resolve(cfg, flow)
    )
    return ("<table><thead><tr><th>Stage</th><th>State</th><th>Layer</th><th>Code</th>"
            f"<th>docsearch.yaml keys</th></tr></thead><tbody>{rows}</tbody></table>")  # fmt: skip


def html(cfg: Config, flows: list[str] | None = None) -> str:
    """One page: each drawing, then its stage table. Light or dark from the system setting. `report --html` can
    put the same sections in its page."""
    from html import escape

    secs = "".join(
        f'<section><h2>{TITLES[f]}</h2><pre class="mermaid">{escape(mermaid(cfg, f))}</pre>'
        f"<details><summary>Stages and settings</summary>{_stage_table(cfg, f)}</details></section>"
        for f in (flows or list(FLOWS))
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark"><title>docsearch architecture</title>
<style>
:root {{ --bg:#fff; --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; --card:#f6f8fa; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --line:#3d444d;
  --card:#151b23; }} }}
body {{ margin:0 auto; max-width:1200px; padding:24px; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui, sans-serif; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; margin:16px 0; }}
pre.mermaid {{ background:none; text-align:center; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
td, th {{ border-top:1px solid var(--line); padding:4px 8px; text-align:left; vertical-align:top; }}
tr.off td {{ color:var(--muted); font-style:italic; }}
p.legend {{ color:var(--muted); }}
</style></head><body>
<h1>docsearch architecture</h1><p class="legend">{escape(LEGEND)}</p>{secs}
<script type="module">
import mermaid from "{MERMAID_JS}";
const dark = matchMedia("(prefers-color-scheme: dark)").matches;
mermaid.initialize({{ startOnLoad: false, theme: dark ? "dark" : "default", securityLevel: "strict" }});
for (const [i, el] of [...document.querySelectorAll("pre.mermaid")].entries()) {{
  el.innerHTML = (await mermaid.render(`drawing-${{i}}`, el.textContent)).svg;  // one fixed id per drawing
}}
</script></body></html>
"""
