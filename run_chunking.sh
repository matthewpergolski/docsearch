#!/bin/sh
# Chunking challengers vs locked V2b (format-aware visual fusion + gte top30@384, rerank_min_chars 40).
# C2: structure-aware spreadsheet chunks (hybrid_tbl). C3: PDF 384-token chunks + 64-token overlap (hybrid_pdf).
# Resumable: derived variants copy unchanged chunks; their LanceDB tables are seeded with the base vectors,
# so only re-chunked docs are embedded, and an interrupted index resumes.
set -e
cd "$(dirname "$0")"
export DOCSEARCH_SEARCH_RERANK_CANDIDATES=30
export DOCSEARCH_MODELS_RERANK_MAX_LENGTH=384
V=${VISUAL:-pplx-v2-late}
for spec in ${RUNS:-hybrid_tbl:_c2 hybrid_pdf:_c3}; do
  var=${spec%%:*}; suf=${spec#*:}
  echo "[chunk-run] $var start $(date '+%F %T %Z')"
  uv run docsearch chunk --variant "$var"
  uv run docsearch index --variant "$var"
  uv run docsearch eval --visual "$V" --fusion format_aware --variant "$var" --suffix "$suf" \
    --notes "chunking challenger $var; V2b otherwise; CPU machine"
  echo "[chunk-run] $var done $(date '+%F %T %Z')"
done
uv run docsearch report
