#!/usr/bin/env sh
# Phase-1 run as executed on a CPU-only machine with 8 vCPU. Steps run strictly one after another:
# parsing and embedding compete for the same cores. On an Apple-silicon laptop (MPS) or a CUDA machine,
# `uv run docsearch all` with the yaml defaults is the intended path.
set -e
export DOCSEARCH_SEARCH_RERANK_CANDIDATES=${DOCSEARCH_SEARCH_RERANK_CANDIDATES:-30}
export DOCSEARCH_MODELS_RERANK_MAX_LENGTH=${DOCSEARCH_MODELS_RERANK_MAX_LENGTH:-384}
[ -n "$SKIP_CHUNK" ] || uv run docsearch chunk
uv run docsearch index
uv run docsearch eval --notes "CPU machine run: rerank top-${DOCSEARCH_SEARCH_RERANK_CANDIDATES} @ ${DOCSEARCH_MODELS_RERANK_MAX_LENGTH} tokens"
uv run docsearch report
