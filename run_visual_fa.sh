#!/bin/sh
# V2: format-aware fusion (one change vs V1), plus a variant that skips gte for near-empty text.
set -e
cd "$(dirname "$0")"
export DOCSEARCH_SEARCH_RERANK_CANDIDATES=30
export DOCSEARCH_MODELS_RERANK_MAX_LENGTH=384
V=${VISUAL:-pplx-v2-late}
uv run docsearch eval --visual "$V" --fusion format_aware --notes "V2 format-aware fusion; gte; CPU machine; top30@384"
DOCSEARCH_SEARCH_RERANK_MIN_CHARS=40 uv run docsearch eval --visual "$V" --fusion format_aware --suffix _skipshort \
  --notes "V2b format-aware fusion; gte skipped for <40 chars of text; CPU machine; top30@384"
uv run docsearch report
