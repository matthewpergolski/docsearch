#!/bin/sh
# Visual page retrieval vs the locked baseline (hybrid chunks + Qwen3 vectors + BM25, gte rerank top-30 @ 384, CPU machine,
# box subset 168 q). Resumable: render reuses images, page vectors are committed shard by shard,
# rerank scores are cached.
set -e
cd "$(dirname "$0")"
export DOCSEARCH_SEARCH_RERANK_CANDIDATES=30
export DOCSEARCH_MODELS_RERANK_MAX_LENGTH=384
V=${VISUAL:-pplx-v2-late}
echo "[visual-run] render start $(date '+%F %T %Z')"
uv run docsearch render
echo "[visual-run] embed start $(date '+%F %T %Z')"
uv run docsearch index --visual "$V"
echo "[visual-run] eval start $(date '+%F %T %Z')"
uv run docsearch eval --visual "$V" --notes "visual $V; gte reranker; CPU machine; top30@384"
echo "[visual-run] done $(date '+%F %T %Z')"
uv run docsearch report
