#!/bin/sh
# Embedder challengers vs the locked baseline (hybrid chunks, BM25 from the same LanceDB FTS index,
# gte-reranker-modernbert-base, top-30 @ 384 tokens, box subset 168 q, CPU machine). Resumable: vector stores are
# committed shard by shard, rerank scores are cached; a re-run only does missing work.
set -e
cd "$(dirname "$0")"
export DOCSEARCH_SEARCH_RERANK_CANDIDATES=30
export DOCSEARCH_MODELS_RERANK_MAX_LENGTH=384
for e in ${EMBEDDERS:-lateon pplx-context}; do
  echo "[challenger] $e index start $(date '+%F %T %Z')"
  uv run docsearch index --embedder "$e"
  echo "[challenger] $e eval start $(date '+%F %T %Z')"
  uv run docsearch eval --embedder "$e" --notes "embedder challenger $e; gte reranker; CPU machine; top30@384"
  echo "[challenger] $e done $(date '+%F %T %Z')"
done
uv run docsearch report
