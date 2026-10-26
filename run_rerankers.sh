#!/bin/sh
# Reranker challengers vs the locked baseline (hybrid chunks, same index, top-30 @ 384 tokens, box subset, CPU machine).
# Resumable: scores are cached in work/cache/rerank_<model>_<len>.jsonl, so a re-run only scores missing pairs.
set -e
cd "$(dirname "$0")"
export DOCSEARCH_SEARCH_RERANK_CANDIDATES=30
export DOCSEARCH_MODELS_RERANK_MAX_LENGTH=384
for r in ${RERANKERS:-gte-modernbert zerank-2}; do
  echo "[challenger] $r start $(date '+%F %T %Z')"
  uv run docsearch eval --reranker "$r" --notes "reranker challenger $r; CPU machine; top30@384"
  echo "[challenger] $r done $(date '+%F %T %Z')"
done
uv run docsearch report
