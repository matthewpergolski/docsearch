"""Retrieve, then answer with an LLM, citing file + page. Says 'not found' instead of guessing."""

from __future__ import annotations

import re
from pathlib import Path

from .config import Config
from .llm import LLMClient, check_available, settings
from .manifest import read_manifest

NOT_FOUND = "Not found in the documents."
SYSTEM = (
    "You answer questions using ONLY the numbered passages provided. Cite every claim with the "
    "passage number in square brackets, e.g. [2]. If the passages do not contain the answer, reply "
    f"exactly: {NOT_FOUND} Do not use outside knowledge. Be concise."
)


def source_label(hit: dict, paths: dict[str, str]) -> str:
    name = Path(paths.get(hit["doc_id"], hit["doc_id"])).name
    pages = hit.get("pages") or []
    if not pages:
        return name
    return f"{name} p.{pages[0]}" if len(pages) == 1 else f"{name} pp.{pages[0]}-{pages[-1]}"


def build_prompt(question: str, hits: list[dict], paths: dict[str, str], max_chars: int = 2500) -> str:
    blocks = [f"[{i}] ({source_label(h, paths)})\n{h['text'][:max_chars]}" for i, h in enumerate(hits, 1)]
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def parse_citations(answer: str, hits: list[dict], paths: dict[str, str]) -> list[dict]:
    seen, out = set(), []
    for m in re.finditer(r"\[(\d+)\]", answer):
        i = int(m.group(1))
        if 1 <= i <= len(hits) and i not in seen:
            seen.add(i)
            h = hits[i - 1]
            out.append({"n": i, "doc_id": h["doc_id"], "pages": h.get("pages") or [], "label": source_label(h, paths)})
    return out


def answer(
    cfg: Config,
    question: str,
    k: int = 8,
    variant: str = "hybrid",
    client: LLMClient | None = None,
    retrieve=None,
    **llm_overrides,
) -> dict:
    """Raises llm.LLMUnavailable before any retrieval work if the backend can't run."""
    client = client or LLMClient(settings(cfg, "answer", **llm_overrides))
    check_available(client.s, client.env, client.which)
    if retrieve is None:
        from .search import search

        def retrieve(q):
            return search(cfg, q, variant=variant, k=k)

    hits = retrieve(question)[:k]
    man = cfg.path("corpus_dir") / "manifest.csv"
    paths = {d.doc_id: d.path for d in read_manifest(man)} if man.exists() else {}
    if not hits:
        return {"answer": NOT_FOUND, "citations": [], "hits": []}
    text = client.complete(build_prompt(question, hits, paths), system=SYSTEM)
    return {
        "answer": text,
        "citations": parse_citations(text, hits, paths),
        "not_found": NOT_FOUND.rstrip(".").lower() in text.lower(),
        "hits": [{"doc_id": h["doc_id"], "pages": h.get("pages"), "label": source_label(h, paths)} for h in hits],
        "llm": {"backend": client.s.backend, "provider": client.s.provider, "model": client.s.model},
    }
