"""Empty-page example searches from the ingested corpus (not the static yaml list).

Samples indexed chunks, drafts short search questions with the questions.Drafter machinery,
caches the strings under data/web_examples.json, and regenerates when the chunks file changes.
With no chunks (or a failed draft), falls back to web.examples from docsearch.yaml.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from .config import Config

log = logging.getLogger("docsearch.web_examples")

CACHE_NAME = "web_examples.json"
DEFAULT_N = 6
DraftFn = Callable[..., list[str]]


def cache_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / CACHE_NAME


def fallback_examples(cfg: Config) -> list[str]:
    return [str(q) for q in cfg["web"]["examples"]]


def corpus_key(cfg: Config) -> str | None:
    """Identity of the searchable chunks file, or None when nothing is ingested."""
    from .chunk import chunks_path, search_variant

    path = chunks_path(cfg, search_variant(cfg))
    if not path.is_file():
        return None
    st = path.stat()
    return f"{path.name}:{st.st_size}:{st.st_mtime_ns}"


def _read_cache(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict) or "corpus_key" not in obj or "examples" not in obj:
        return None
    ex = obj["examples"]
    if not isinstance(ex, list) or not all(isinstance(q, str) and q.strip() for q in ex):
        return None
    return obj


def _write_cache(path: Path, key: str, examples: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"corpus_key": key, "examples": examples}, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def draft_from_chunks(cfg: Config, n: int = DEFAULT_N, seed: int = 11, client=None,
                      chunks: list[dict] | None = None) -> list[str]:  # fmt: skip
    """Sample chunks via questions.Drafter and keep the drafted query strings."""
    from .chunk import search_variant
    from .llm import LLMClient, LLMError, LLMUnavailable, check_available, settings
    from .questions import Drafter, load_chunks, page_images

    # Same variant corpus_key fingerprints (hybrid_ctx_slides when enrich.llm is on).
    chunks = chunks if chunks is not None else load_chunks(cfg, search_variant(cfg))
    if not chunks:
        return []
    if client is None:
        client = LLMClient(settings(cfg, "questions"))
        check_available(client.s, client.env, client.which)
    # closed_book off: UI examples only need grounded, searchable questions
    d = Drafter(cfg, client, chunks, page_images(cfg), seed=seed, closed_book=False, use_images=False)
    out: list[str] = []
    seen: set[str] = set()
    for i in range(max(n * 3, n)):
        if len(out) >= n:
            break
        try:
            row = d.image(i) or d.multi_doc(i) or d.no_answer(i)
        except (LLMError, LLMUnavailable):
            break
        if not row:
            continue
        q = (row.get("query") or "").strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def examples(cfg: Config, draft: DraftFn | None = None, n: int = DEFAULT_N) -> list[str]:
    """Example searches for the empty page: cache → draft from chunks → yaml fallback."""
    key = corpus_key(cfg)
    if key is None:
        return fallback_examples(cfg)

    path = cache_path(cfg)
    cached = _read_cache(path)
    if cached and cached["corpus_key"] == key:
        return list(cached["examples"])

    draft_fn = draft or draft_from_chunks
    try:
        got = [q.strip() for q in draft_fn(cfg, n=n) if isinstance(q, str) and q.strip()]
    except Exception as e:  # noqa: BLE001 - UI must still load; yaml examples are fine
        log.warning("drafting web examples failed (%s); using web.examples fallback", e)
        return fallback_examples(cfg)
    if not got:
        return fallback_examples(cfg)
    try:
        _write_cache(path, key, got)
    except OSError:
        log.exception("could not write %s", path)
    return got
