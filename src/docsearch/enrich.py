"""+enrich (Phase H): an LLM writes one short context line per chunk at index time.

The line goes after the chunk's `File:` line, so keyword search, the embedding and the text reranker all see it.
Input per call: the file's outline (its name and the heading lines of its chunks) and a batch of its chunks.
Nothing else reaches the model: this module reads chunks only, never a test set or a search log.
Replies are cached by a hash of model + prompt, so a chunk whose file outline and batch are unchanged is not sent
again; a changed chunk sends its batch again.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
from pathlib import Path

from .chunk import Chunk

PROMPT_VERSION = "h1"
OUTLINE_CHARS = 1500
CHUNK_CHARS = 1800
PROMPT = """You write one short line of context for each chunk of one document, so that search can find the chunk.
Use only the document outline and the chunk. Say which document it is (title, organisation, period when shown)
and what the chunk covers (section, table, sheet or slide topic). At most 25 words per line. Do not invent facts.

Document outline:
{outline}

Chunks:
{chunks}

Reply with JSON only, one line per chunk number: {{"1": "...", "2": "..."}}"""


def _body(c: Chunk) -> str:
    return c.text.split("\n", 1)[1] if c.text.startswith("File:") and "\n" in c.text else c.text


def outline(chunks: list[Chunk]) -> str:
    """File name, then the first line of each chunk (Docling puts the heading path there), each once, capped."""
    name = chunks[0].doc_id.split("/")[-1]
    out, seen = [f"File name: {name}"], set()
    for c in chunks:
        head = _body(c).split("\n", 1)[0].strip()
        if head and len(head) <= 160 and head not in seen:
            seen.add(head)
            out.append(head)
    return "\n".join(out)[:OUTLINE_CHARS]


def _parse(reply: str | None, n: int) -> dict[int, str] | None:
    m = re.search(r"\{.*\}", reply or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    got = {int(k): " ".join(str(v).split()) for k, v in d.items() if str(k).isdigit() and str(v).strip()}
    return {k: v for k, v in got.items() if 1 <= k <= n} or None


class Enricher:
    """client: anything with complete(prompt) -> str. model: 'provider:model', part of the cache key."""

    def __init__(self, client, model: str, cache_dir: Path, batch: int = 8):
        self.client, self.model, self.cache_dir, self.batch = client, model, Path(cache_dir), batch
        self.stats = {"calls": 0, "cached_batches": 0, "batches": 0, "seconds": 0.0, "missing": 0, "errors": 0}

    def _ask(self, prompt: str, n: int) -> dict[int, str] | None:
        """The parsed reply, cached. A reply that misses a chunk is asked once more and the fuller one kept; a cached
        reply from one call that misses a chunk gets that second try too (H1)."""
        from .llm import LLMError

        key = hashlib.sha256(f"{self.model}\n{PROMPT_VERSION}\n{prompt}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        best, reply, calls = None, None, 0
        if path.exists():
            hit = json.loads(path.read_text())
            best, reply, calls = _parse(hit["reply"], n), hit["reply"], hit.get("calls", 1)
            if (best and len(best) == n) or calls >= 2:
                self.stats["cached_batches"] += 1
                return best
        t0, asked = time.time(), 0
        try:
            while calls < 2 and not (best and len(best) == n):
                calls += 1
                asked += 1
                r = self.client.complete(prompt)
                got = _parse(r, n)
                if got and len(got) > len(best or {}):
                    best, reply = got, r
        except LLMError:
            self.stats["errors"] += 1
        finally:
            self.stats["calls"] += asked
            self.stats["seconds"] += time.time() - t0
        if best is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{time.time_ns()}.tmp")
        tmp.write_text(json.dumps({"model": self.model, "prompt": prompt, "reply": reply, "calls": calls,
                                   "seconds": round(time.time() - t0, 2)}))  # fmt: skip
        tmp.replace(path)
        return best

    def lines(self, chunks: list[Chunk], on_batch=None) -> dict[str, str]:
        """chunk_id -> context line, for chunks of one or more files (in file order). A chunk the model gave no
        line for is left out and counted in stats['missing']."""
        by_doc: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)
        out = {}
        for doc_chunks in by_doc.values():
            head = outline(doc_chunks)
            for i in range(0, len(doc_chunks), self.batch):
                part = doc_chunks[i : i + self.batch]
                body = "\n\n".join(f"[{j}] {_body(c)[:CHUNK_CHARS]}" for j, c in enumerate(part, 1))
                got = self._ask(PROMPT.format(outline=head, chunks=body), len(part)) or {}
                self.stats["batches"] += 1
                for j, c in enumerate(part, 1):
                    if j in got:
                        out[c.chunk_id] = got[j]
                    else:
                        self.stats["missing"] += 1
                if on_batch:
                    on_batch(len(part))
        return out


def apply(chunks: list[Chunk], lines: dict[str, str]) -> list[Chunk]:
    """The chunks with each one's context line after its `File:` line (or first, when it has none)."""
    out = []
    for c in chunks:
        line = lines.get(c.chunk_id)
        if not line:
            out.append(c)
            continue
        if c.text.startswith("File:") and "\n" in c.text:
            first, rest = c.text.split("\n", 1)
            text = f"{first}\nContext: {line}\n{rest}"
        else:
            text = f"Context: {line}\n{c.text}"
        out.append(dataclasses.replace(c, text=text))
    return out
