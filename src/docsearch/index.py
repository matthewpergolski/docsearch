"""Build a LanceDB table per chunking variant: vectors (Qwen3-Embedding) + native full-text index."""

from __future__ import annotations

import json
import time

import pyarrow as pa

from . import ui
from .chunk import chunks_path, read_chunks
from .config import Config

BLOCK = 256


def table_name(variant: str) -> str:
    return f"chunks_{variant}"


def connect(cfg: Config):
    import lancedb

    cfg.path("index_dir").mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(cfg.path("index_dir")))


def schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("chunk_id", pa.string()),
            ("doc_id", pa.string()),
            ("format", pa.string()),
            ("source", pa.string()),
            ("pages", pa.list_(pa.int32())),
            ("text", pa.string()),
            ("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def check_model(cfg: Config, tbl, variant: str, embedder: str, dim: int) -> None:
    """Stop when the table was built with another embedder or dimension than `embedder`/`dim`. The record is the
    table's stats.json ("embedder", written by build_index) and the width of its vector column."""
    have_dim = tbl.schema.field("vector").type.list_size
    sp = cfg.path("index_dir") / f"{table_name(variant)}.stats.json"
    have = json.loads(sp.read_text()).get("embedder") if sp.exists() else None
    if (have and have != embedder) or have_dim != dim:
        fix = (f"Set models.embedder back to {have}" if have and have != embedder
               else "Check that the model (or model server) behind models.embedder is that model")  # fmt: skip
        raise SystemExit(f"index {table_name(variant)} was built with {have or 'an unrecorded embedder'} "
                         f"({have_dim} dims), but this run embeds with {embedder} ({dim} dims). {fix}, or rebuild: "
                         "docsearch index --rebuild")  # fmt: skip


def build_index(cfg: Config, variant: str, rebuild: bool = False) -> dict:
    from .models import load_embedder

    chunks = read_chunks(chunks_path(cfg, variant))
    db = connect(cfg)
    name = table_name(variant)
    emb = load_embedder(cfg)
    if rebuild and name in db.table_names():
        db.drop_table(name)
    if name in db.table_names():
        tbl = db.open_table(name)
        check_model(cfg, tbl, variant, emb.name, emb.dim)
        n_rows = tbl.count_rows()
        stored = tbl.search().select(["chunk_id", "text"]).limit(n_rows).to_arrow().to_pylist() if n_rows else []
        # A chunk whose id is kept but whose text changed (e.g. a re-chunk, M28) is stale: embed it again.
        want = {c.chunk_id: c.text for c in chunks}
        done = {r["chunk_id"] for r in stored}
        changed = {r["chunk_id"] for r in stored if r["chunk_id"] in want and want[r["chunk_id"]] != r["text"]}
    else:
        tbl = db.create_table(name, schema=schema(emb.dim))
        done, changed = set(), set()
    current = {c.chunk_id for c in chunks}
    stale = sorted((done - current) | changed)
    done -= changed
    for i in range(0, len(stale), 500):  # chunks that no longer exist (re-parsed/re-chunked docs)
        ids = ", ".join("'" + x.replace("'", "''") + "'" for x in stale[i : i + 500])
        tbl.delete(f"chunk_id IN ({ids})")
    from .chunk import base_variant

    base = base_variant(variant)
    missing = {c.chunk_id: c.text for c in chunks if c.chunk_id not in done}
    if base and missing and table_name(base) in db.table_names():
        # Derived or enriched variant: reuse the base table's vectors for chunks whose id and text are unchanged,
        # so only re-chunked (or enriched) chunks are embedded. Also on a table that exists (sync, H2).
        src = db.open_table(table_name(base))
        check_model(cfg, src, base, emb.name, emb.dim)
        rows = src.search().limit(src.count_rows()).to_arrow().to_pylist()
        seed = [r for r in rows if missing.get(r["chunk_id"]) == r["text"]]
        for i in range(0, len(seed), 1000):
            tbl.add(pa.Table.from_pylist(seed[i : i + 1000], schema=schema(emb.dim)))
        done |= {r["chunk_id"] for r in seed}
        ui.info(f"[index] {variant}: seeded {len(seed)} unchanged chunks from {table_name(base)}", flush=True)
    todo = [c for c in chunks if c.chunk_id not in done]
    # Longest-first blocks keep padding low and surface memory problems early.
    todo.sort(key=lambda c: -len(c.text))
    t0 = time.time()
    pb = ui.progress(len(todo), f"embed {variant}", "chunks ").start()
    for i in range(0, len(todo), BLOCK):
        block = todo[i : i + BLOCK]
        vecs = emb.docs([c.text for c in block], on_batch=pb.advance)
        tbl.add(
            pa.Table.from_pylist(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "format": c.format,
                        "source": c.source,
                        "pages": c.pages,
                        "text": c.text,
                        "vector": v.tolist(),
                    }
                    for c, v in zip(block, vecs, strict=True)
                ],
                schema=schema(emb.dim),
            )
        )
        el = time.time() - t0
        n = i + len(block)
        ui.info(f"[index] {variant}: {n}/{len(todo)} embedded, {el:.0f}s ({n / el:.1f}/s)", flush=True)
    pb.close()
    embed_s = time.time() - t0
    t1 = time.time()
    tbl.create_fts_index("text", use_tantivy=False, replace=True)
    stats = {
        "variant": variant,
        "rows": tbl.count_rows(),
        "embedded_now": len(todo),
        "deleted_stale": len(stale),
        "embed_seconds": round(embed_s, 1),
        "fts_seconds": round(time.time() - t1, 1),
        "device": emb.device,
        "embedder": emb.name,
    }
    (cfg.path("index_dir") / f"{name}.stats.json").write_text(json.dumps(stats, indent=2))
    ui.info(f"[index] {json.dumps(stats)}", flush=True)
    return stats
