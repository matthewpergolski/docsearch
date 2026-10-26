"""Configuration loading: docsearch.yaml + DOCSEARCH_* env overrides. No machine-specific paths."""

from __future__ import annotations

import copy
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "paths": {
        "data_dir": "data",
        "corpus_dir": "corpus",
        "parsed_dir": "work/parsed",
        "index_dir": "work/index",
        "cache_dir": "work/cache",
        "results_dir": "results",
        "sync_manifest": "work/manifest.sqlite",  # per-file state for `docsearch sync`
    },
    "models": {
        "embedder": "Qwen/Qwen3-Embedding-0.6B",
        "reranker": "Qwen/Qwen3-Reranker-0.6B",
        "rerankers": {},
        "page_rerankers": {},  # M16: page-image (vision-language) rerankers, alias -> HF id
        "embedders": {},
        "device": "auto",
        "embed_dtype": "auto",
        "rerank_dtype": "auto",
        "embed_batch_size": 16,
        "rerank_batch_size": 16,
        "rerank_batch_sizes": {},  # per reranker (alias or HF id) overrides, e.g. {qwen3-4b: 4} for big models
        "rerank_max_length": 512,
        # Phase G: "api" calls the embedder and the text reranker on an HTTP server (OpenAI-compatible
        # {embed_api_url}/embeddings, and {rerank_api_url}/rerank) instead of loading them; api runs text-only.
        "backend": "local",
        "embed_api_url": "",
        "embed_api_model": "",  # the name the server knows; "" = models.embedder
        "embed_api_key_env": "DOCSEARCH_EMBED_API_KEY",  # the env var that holds the key; never the key itself
        "embed_api_query_prefix": None,  # the query lead-in; None = read it from the model's HF config
        "rerank_api_url": "",
        "rerank_api_model": "",  # "" = models.reranker
        "rerank_api_key_env": "DOCSEARCH_RERANK_API_KEY",
        "rerank_api_field": "documents",  # the request field that holds the passages; TEI wants "texts"
        "api_timeout_s": 60.0,
        "api_retries": 3,
        "api_batch_size": 32,  # texts per request (TEI's default client batch limit is 32)
    },
    "ingest": {
        "pdf_ocr": False,  # M22: OCR the regions of a PDF page that have no text layer (scanned pages)
        "doc_converter": "auto",  # DOC1: .doc -> .docx program: auto (soffice, then textutil) | soffice | textutil | off
    },
    "chunking": {
        "max_tokens": 512,
        "max_chunks_per_doc": 300,
        "max_parsed_mb": 100,
        "pdf_max_tokens": 384,  # C3 (hybrid_pdf variant) only
        "pdf_overlap_tokens": 64,
    },
    "search": {
        "bm25_k": 100,
        "vector_k": 100,
        "rrf_k": 60,
        "rerank_candidates": 50,
        "top_k": 10,
        "rerank_skip_formats": [],  # formats never reranked (keep fused rank), e.g. [slide_image]; M3
        "rerank_min_chars": 0,
        "distinct_boost": 0.0,  # M11 near-duplicate tie-break (rank positions per distinguishing query term)
        "rerank_context": False,
        "late_visual_formats": [],  # M13: formats whose reranked order is re-fused with the visual rank
        "late_visual_k": 10,
        "number_boost": 0.0,  # M14: positions per exact query number on a candidate page
        "rerank_unit_chunks": 0,  # M17: cap on the chunks one page/file puts in a row in the fused list; 0 = off
        "page_reranker": "",  # M16: page-image reranker (alias in models.page_rerankers or HF id); "" = off
        "page_rerank_pages": 10,
        "page_rerank_formats": ["pdf", "slide_image"],
        "visual_weights": {},  # M20: {format: weight of its visual rank in format-aware fusion}; {} = 1 for all
        "period_demote": False,  # M21: files named for another quarter than the query's go behind the rest
        "table_lookup": False,  # M33 (C2): a file's page whose table row and column the query names goes first
        "number_boost_formats": [
            "pdf"
        ],  # M12 doc/page context in reranker input; also reranks rerank_skip_formats  # >0: candidates with less text keep their fused rank (not reranked)
    },
    "sync": {
        "variants": ["hybrid"],  # chunk variants kept up to date (their LanceDB tables)
        "gate_tolerance": 0.005,  # --gate: promote only if nDCG@10 >= live - tolerance
        "interval": 900,  # --watch default, seconds
        "source": "",  # source tag for files not in corpus/manifest.csv ("" = folder name)
    },
    "logging": {
        "dir": "logs",  # relative to the config file
        "keep_runs": 20,  # newest logs/run-*.log kept; other logs are never touched
    },
    "trial": {
        "name": "",  # DOCSEARCH_TRIAL_NAME: step timings go to results/timings_<name>.json, not the shared file
    },
    "visual": {
        "dpi": 100,  # PDF/office page render resolution
        "max_side": 1280,  # longest image side fed to the visual encoder
        "office_max_pages": 4,  # docx/pptx pages rendered per file (after LibreOffice -> PDF)
        "shard_pages": 8,
        "visual_k": 100,  # visual units fused with the hybrid ranking
        "text_rich_chars": 0,  # M25: >0 = a page/file with at least this much text keeps its text rank in the fusion
        "text_rich_formats": ["pdf", "docx", "pptx"],  # M25: the formats that rule applies to
        "office_own_formats": ["docx", "pptx"],  # M24b: the Office formats whose pages past the first N are used
        "office_shared_pages": 0,  # M24: >0 = only the first N pages of a docx/pptx compete with other formats
        "blind_formats": [],  # format-aware fusion treats these as unseen (keep text rank), e.g. [pptx, csv]; M4
        "default": None,  # visual alias used by `search`/`answer` (format-aware fusion) when its store exists
    },
    "sampling": {
        "seed": 13,
        "vidore_corpora": ["vidore_v3_hr"],
        "vidore_queries_per_corpus": 150,
        "finslides_decks": 12,
        "ocb_max_file_mb": 3,
        "ocb_max_office_doc_mb": 40,
    },
    "subset": {
        "enabled": False,
        "seed": 13,
        "questions_per_source": {},
        "distractor_ratio": 2,
        "office_max_chunks": 12,
    },
    "llm": {
        "backend": "api",
        "provider": "claude",
        "timeout_s": 180,
        "timeout_s_by_provider": {},
        "api_models": {},
        "cli_models": {},
        "ollama_api_base": "http://localhost:11434",
    },
    "enrich": {"llm": "off", "backend": "cli", "provider": "claude", "model": "haiku"},
    "agent": {
        "mode": "off",
        "backend": "cli",
        "provider": "claude",
        "model": "haiku",
        "gate_k": 5,
        "pages": 10,
        "page_chars": 4000,
        "workers": 4,
        "research": "off",
        "deck_gate": "off",
    },
    "web": {"examples": []},
}

# The LLM layers' switches and their allowed values. Search never reads them.
SWITCHES: dict[tuple[str, str], tuple[str, ...]] = {
    ("enrich", "llm"): ("off", "local", "api"),
    ("agent", "mode"): ("off", "on"),
    ("agent", "research"): ("off", "on"),
    ("agent", "deck_gate"): ("off", "on"),
    ("ingest", "doc_converter"): ("auto", "soffice", "textutil", "off"),
}
CHEAP_MODELS = ("haiku", "mini", "nano", "flash")  # enrich.model must name one: one call per slide adds up
BACKENDS = ("local", "api")  # models.backend: where the embedder and the text reranker run


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(old: Any, raw: str) -> Any:
    if isinstance(old, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(old, int):
        return int(raw)
    if isinstance(old, float):
        return float(raw)
    if isinstance(old, list):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return raw


def _apply_env(cfg: dict, env: dict[str, str]) -> dict:
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key, old in values.items():
            name = f"DOCSEARCH_{section}_{key}".upper()
            if name in env and not isinstance(old, dict):
                values[key] = _coerce(old, env[name])
    return cfg


def find_config_file(start: Path | None = None) -> Path | None:
    if os.environ.get("DOCSEARCH_CONFIG"):
        return Path(os.environ["DOCSEARCH_CONFIG"]).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for d in [here, *here.parents]:
        if (d / "docsearch.yaml").exists():
            return d / "docsearch.yaml"
    return None


@dataclass
class Config:
    raw: dict
    root: Path

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key]).expanduser()
        return p if p.is_absolute() else (self.root / p)

    def __getitem__(self, section: str) -> dict:
        return self.raw[section]


def load_config(path: Path | None = None, env: dict[str, str] | None = None) -> Config:
    path = path or find_config_file()
    data: dict = {}
    root = Path.cwd()
    if path and path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        root = path.parent
    cfg = _apply_env(_merge(DEFAULTS, data), dict(os.environ if env is None else env))
    for (section, key), allowed in SWITCHES.items():
        v = cfg[section][key]
        if isinstance(v, bool):  # YAML reads a bare on/off as a boolean
            v = cfg[section][key] = "on" if v else "off"
        if v not in allowed:
            raise ValueError(f"{section}.{key} must be one of {', '.join(allowed)}; got {v!r}")
    em = str(cfg["enrich"]["model"])
    words = set(re.split(r"[^a-z0-9]+", em.lower()))  # whole words: gemini-pro holds mini, but is not a small model
    if cfg["enrich"]["provider"] != "ollama" and not words & set(CHEAP_MODELS):
        raise ValueError(f"enrich.model must be a small, cheap model (its name holds {', '.join(CHEAP_MODELS)}), "
                         f"or set enrich.provider to ollama (a local model); got {em!r}")  # fmt: skip
    if cfg["models"]["backend"] not in BACKENDS:
        raise ValueError(f"models.backend must be one of {', '.join(BACKENDS)}; got {cfg['models']['backend']!r}")
    if cfg["models"]["backend"] == "api" and cfg["visual"].get("default"):
        # The page-picture model has no common server API (Phase G), so api runs text-only.
        logging.getLogger("docsearch.models").info("models.backend api: visual.default %s ignored (text-only)",
                                                   cfg["visual"]["default"])  # fmt: skip
        cfg["visual"]["default"] = None
    return Config(raw=cfg, root=root.resolve())


def resolve_reranker(cfg: Config, name: str) -> str:
    """Point models.reranker at an alias from models.rerankers (or a raw HF id); return a short label."""
    aliases = cfg["models"].get("rerankers") or {}
    if name in aliases:
        cfg["models"]["reranker"] = aliases[name]
        return name
    cfg["models"]["reranker"] = name
    return name.split("/")[-1]


DEVICES = ("auto", "cuda", "mps", "cpu")
_dev_log = logging.getLogger("docsearch.device")


def pick_device(pref: str = "auto", env: dict | None = None) -> str:
    """The one place a torch device is chosen (embedders, rerankers, vision reranker, Docling layout/OCR).
    DOCSEARCH_DEVICE (env) wins over models.device (config); "auto" picks cuda, then mps (Apple Silicon), then cpu.
    The choice is logged on the "docsearch.device" logger."""
    env = os.environ if env is None else env
    want = (env.get("DOCSEARCH_DEVICE") or pref or "auto").strip().lower()
    if want not in DEVICES:
        raise ValueError(f"device must be one of {', '.join(DEVICES)}; got {want!r}")
    if want != "auto":
        dev, why = want, "pinned by DOCSEARCH_DEVICE" if env.get("DOCSEARCH_DEVICE") else "pinned by models.device"
    else:
        import torch

        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        why = "auto: cuda -> mps -> cpu"
    _dev_log.info("device: %s (%s)", dev, why)
    return dev


def pick_dtype(device: str, pref: str = "auto", role: str = "embed"):
    """float16 on cuda/mps. On CPU: float32 for the embedder (measured faster on the box) and
    bfloat16 for the reranker when the CPU has AVX512/AMX (measured ~2x faster than float32)."""
    import torch

    if pref != "auto":
        return getattr(torch, pref)
    if device in ("cuda", "mps"):
        return torch.float16
    if role == "embed":
        return torch.float32
    try:
        cpu_bf16 = torch.backends.cpu.get_cpu_capability() in {"AVX512", "AVX512_BF16", "AMX"}
    except Exception:  # noqa: BLE001
        cpu_bf16 = False
    return torch.bfloat16 if cpu_bf16 else torch.float32
