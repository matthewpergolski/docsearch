"""Phase C, PC2: a second model (Codex CLI) checks a seeded sample of the held-out table questions that Grok drafted
and reviewed. For each: is the gold value right on the source text, and can the question be answered from it alone?
Writes nothing to data/; replies are cached under work/cache/xmodel_review/.

  uv run python scripts/xmodel_review.py --n 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from draft_table_questions import build_targets  # noqa: E402

from docsearch.config import load_config  # noqa: E402
from docsearch.llm import LLMClient, settings  # noqa: E402

CACHE = Path("work/cache/xmodel_review")
PROMPT = """You check one question from a test set about tables. Use only the source text below.

Source ({what} of {doc}):
{text}

Question: {query}
Claimed answer: {answer}

Reply with one JSON object and nothing else:
{{"value_correct": true or false, "answerable": true or false, "one_answer": true or false, "reason": "<one sentence>"}}
value_correct: the source holds the claimed answer for exactly what the question asks.
answerable: a reader of this source alone can find the answer (entity, measure and period are clear).
one_answer: no other value in the source fits the question equally well."""


def ask(client: LLMClient, prompt: str) -> dict:
    path = CACHE / f"{hashlib.sha256(prompt.encode()).hexdigest()}.json"
    if path.exists():
        reply = json.loads(path.read_text())["reply"]
    else:
        reply = client.complete(prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"prompt": prompt, "reply": reply}))
    start, end = reply.find("{"), reply.rfind("}")
    try:
        return json.loads(reply[start : end + 1])
    except ValueError:
        return {"value_correct": None, "answerable": None, "one_answer": None, "reason": f"bad reply: {reply[:80]}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--model", default="gpt-6-astra", help="codex model; the CLI default may not serve a ChatGPT login")
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path("data/questions_tables.jsonl").read_text().splitlines() if x.strip()]
    held = [r for r in rows if r["meta"].get("status") == "keep" and r["meta"].get("split") == "heldout"]
    sample = random.Random(a.seed).sample(held, min(a.n, len(held)))
    targets: dict[str, list] = {}
    for t in build_targets():
        for u in {t.unit, t.sheet_unit} - {None}:
            targets.setdefault(u, []).append(t)
    client = LLMClient(settings(load_config(), "questions", backend="cli", provider="openai", model=a.model))
    out, tally = [], Counter()
    for r in sample:
        cell = r["meta"]["gold_cell"]
        found = targets.get(cell["unit"], [])
        t = next(
            (x for x in found if cell.get("table") and x.tid.endswith("@" + cell["table"])), found[0] if found else None
        )
        if t is None:
            tally["no source text"] += 1
            out.append((r["qid"], "no source text"))
            continue
        v = ask(client, PROMPT.format(what=t.what, doc=t.doc_id, text=t.text[:6000], query=r["query"],
                                      answer=r["meta"]["answer"]))  # fmt: skip
        ok = all(v.get(k) is True for k in ("value_correct", "answerable", "one_answer"))
        tally["pass" if ok else "fail"] += 1
        for k in ("value_correct", "answerable", "one_answer"):
            tally[f"{k}={v.get(k)}"] += 1
        out.append((r["qid"], "pass" if ok else f"fail: {v.get('reason', '')}"))
        print(f"{r['qid']:12s} {r['format']:5s} {out[-1][1]}", flush=True)
    print(f"\n{len(sample)} held-out questions checked by codex: {dict(tally)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
