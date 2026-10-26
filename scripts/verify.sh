#!/usr/bin/env bash
# End-to-end proof that the docsearch CLI works on the current default pipeline. See docs/VERIFY.md.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
FIXTURE="$REPO/tests/fixtures/verify"
SOURCE=verify
VISUAL=pplx-v2-late
RUN_NAME="hybrid_visual_fa_${VISUAL}_rerank"
SEARCH_QUERY="How many days of paid leave does a new baker get each year?"
SEARCH_EXPECT="$SOURCE/handbook.docx"
SEARCH_LABEL="Brindlewick Bakery staff handbook | 2024-03-04"
FILTER_FORMAT=xlsx
FILTER_AFTER=2025-01-01
SABOTAGE_QID=verify-06
SABOTAGE_GOLD="$SOURCE/handbook.docx"
UNBUILT_MODE=enrich
UNBUILT_MSG="the enrich layer is not built yet (a later phase builds it)"
RESEARCH_GUARD_MSG="--agent-research needs --mode agent; nothing was run"
DECK_GUARD_MSG="--agent-deck needs --mode agent; nothing was run"
DOC_FILE=handbook.doc
DOC_EMPTY=empty.doc
DOC_EXPECT="$SOURCE/$DOC_FILE"
MARKER=.docsearch-verify
API_RUN=api_text

usage() {
  cat <<EOF
Usage: scripts/verify.sh [--clean] [--sabotage] [--doctor]

  --clean     after the run, remove the scratch state and keep <root>/evidence (or VERIFY_CLEAN=1)
  --sabotage  point question $SABOTAGE_QID at the wrong file and sync the .doc with no converter; the run must
              then fail (or VERIFY_SABOTAGE=1)
  --doctor    read-only: report the tools, the fixture files and the model cache, then exit

VERIFY_DIR sets the scratch root (default: a new temporary directory).
EOF
}

die() {
  echo "verify: $*" >&2
  exit 2
}

clean=${VERIFY_CLEAN:-0}
sabotage=${VERIFY_SABOTAGE:-0}
doctor=0
for arg in "$@"; do
  case $arg in
    --clean) clean=1 ;;
    --sabotage) sabotage=1 ;;
    --doctor) doctor=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPO"
# Inherited overrides (a challenger embedder, another config, API keys) would change what is verified.
# DOCSEARCH_DEVICE stays: it only picks the hardware.
for v in $(compgen -e | grep -E '^DOCSEARCH_|_API_KEY$|^(ANTHROPIC|OPENAI|XAI|GROQ|GEMINI|MISTRAL|COHERE|OPENROUTER)_' || true); do
  [[ $v == DOCSEARCH_DEVICE ]] || unset "$v"
done

git_rev() {
  local rev
  rev=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) || {
    echo "not a git checkout"
    return
  }
  if [[ -n $(git -C "$REPO" status --porcelain 2>/dev/null) ]]; then rev="$rev, with uncommitted changes"; fi
  echo "$rev"
}

run_doctor() {
  local bad=0 f
  echo "repo: $REPO ($(git_rev))"
  if command -v uv >/dev/null; then echo "ok      uv: $(uv --version)"; else
    echo "MISSING uv"
    bad=1
  fi
  if command -v soffice >/dev/null; then echo "ok      soffice: $(command -v soffice)"; else
    echo "MISSING soffice (LibreOffice): Word pages are not rendered"
    bad=1
  fi
  for f in files/handbook.docx files/report.pdf files/budget.xlsx files/suppliers.csv questions.jsonl meta.expected.jsonl; do
    if [[ -f "$FIXTURE/$f" ]]; then echo "ok      fixture: $f"; else
      echo "MISSING fixture: $f (run: uv run python scripts/make_verify_fixture.py)"
      bad=1
    fi
  done
  uv run python -I - "$VISUAL" <<'PY' || bad=1
import sys
from argparse import Namespace

from docsearch.config import load_config
from docsearch.hfcache import missing, models_for

cfg = load_config()
refs = {}
for cmd, a in (("sync", None), ("search", None), ("eval", Namespace(visual=sys.argv[1], fusion="format_aware"))):
    refs.update({str(r): r for r in models_for(cfg, cmd, a)})
gone = {str(r) for r in missing(list(refs.values()))}
for name, r in refs.items():
    print(f"{'MISSING' if name in gone else 'ok     '} model {r.label}: {name}")
sys.exit(1 if gone else 0)
PY
  if [[ $bad == 0 ]]; then echo "doctor: ready"; else echo "doctor: see MISSING lines (a missing model downloads on first use)"; fi
  return "$bad"
}

if [[ $doctor == 1 ]]; then
  run_doctor && exit 0 || exit 1
fi

realpath_py() { uv run python -I -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }

if [[ -n ${VERIFY_DIR:-} ]]; then
  root=$(realpath_py "$VERIFY_DIR")
else
  root=$(realpath_py "$(mktemp -d "${TMPDIR:-/tmp}/docsearch-verify.XXXXXX")")
fi
if [[ "$REPO/" == "$root/"* ]]; then die "VERIFY_DIR $root is the repo or holds it; pick a scratch directory"; fi
for d in data corpus work results logs; do
  protected=$(realpath_py "$REPO/$d")
  if [[ "$root/" == "$protected/"* ]]; then die "VERIFY_DIR $root is inside the repo's $d/; pick a scratch directory"; fi
done
if [[ -d $root && -n $(ls -A "$root") && ! -f "$root/$MARKER" ]]; then
  die "VERIFY_DIR $root is not empty and was not made by verify.sh; pick an empty or new directory"
fi
mkdir -p "$root/evidence" "$root/corpus" "$root/data"
touch "$root/$MARKER"
evidence="$root/evidence"
summary="$evidence/summary.txt"
failed=0

cleanup() {
  if [[ $clean == 1 && -f "$root/$MARKER" ]]; then
    find "$root" -mindepth 1 -maxdepth 1 ! -name evidence ! -name "$MARKER" -exec rm -rf {} +
    echo "verify: removed the scratch state under $root; kept $evidence"
  fi
}
trap cleanup EXIT

record() {
  printf '%s %s\n' "$1" "$2" | tee -a "$summary"
  [[ $1 == PASS ]] || failed=1
}

capture() {
  local log="$evidence/$1.txt" t0=$SECONDS rc=0
  shift
  printf '$ %s\n\n' "$*" >"$log"
  "$@" 2>&1 | tee -a "$log" || rc=$?
  printf '\n[exit %s after %ss]\n' "$rc" "$((SECONDS - t0))" >>"$log"
  return "$rc"
}

run_step() {
  local name=$1 t0=$SECONDS rc=0 dt
  capture "$@" || rc=$?
  dt=$((SECONDS - t0))
  if [[ $rc == 0 ]]; then record PASS "step $name (${dt}s)"; else record FAIL "step $name (exit $rc, ${dt}s)"; fi
  return "$rc"
}

check() {
  local name=$1
  shift
  run_step "$name" uv run python scripts/verify_check.py "$@" || true
  grep -E '^(PASS|FAIL) ' "$evidence/$name.txt" >>"$summary" || true
}

{
  echo "docsearch verify, $(date '+%Y-%m-%d %H:%M:%S')"
  echo "commit: $(git_rev)"
  echo "root: $root"
  echo "sabotage: $sabotage ($SABOTAGE_QID -> $SABOTAGE_GOLD when 1)"
  echo "API key variables set: $(compgen -e | grep -cE '_API_KEY$' || true)"
  echo
} >"$summary"

run_step ruff_check uv run ruff check src tests scripts || true
run_step ruff_format uv run ruff format --check src tests scripts || true
run_step unittest uv run python -m unittest discover -s tests || true

export DOCSEARCH_PATHS_DATA_DIR="$root/data"
export DOCSEARCH_PATHS_CORPUS_DIR="$root/corpus"
export DOCSEARCH_PATHS_PARSED_DIR="$root/work/parsed"
export DOCSEARCH_PATHS_INDEX_DIR="$root/work/index"
export DOCSEARCH_PATHS_CACHE_DIR="$root/work/cache"
export DOCSEARCH_PATHS_RESULTS_DIR="$root/results"
export DOCSEARCH_PATHS_SYNC_MANIFEST="$root/work/manifest.sqlite"
export DOCSEARCH_LOGGING_DIR="$root/logs"
export DOCSEARCH_SYNC_SOURCE="$SOURCE"
export DOCSEARCH_PLAIN=1

run_step paths uv run python -I - "$root" <<'PY' || die "a docsearch path is outside $root; nothing was driven (see $evidence/paths.txt)"
import sys
from pathlib import Path

from docsearch.chunk import chunks_path
from docsearch.config import load_config
from docsearch.meta import meta_path
from docsearch.visual import pages_dir

root = Path(sys.argv[1])
cfg = load_config()
paths = {f"paths.{k}": cfg.path(k) for k in cfg["paths"]}
paths |= {"logging.dir": cfg.root / cfg["logging"]["dir"], "chunks": chunks_path(cfg, "hybrid").parent,
          "pages": pages_dir(cfg), "meta": meta_path(cfg)}
bad = [k for k, p in paths.items() if not p.resolve().is_relative_to(root)]
for k, p in paths.items():
    print(f"{'FAIL' if k in bad else 'PASS'} paths: {k} -> {p}")
sys.exit(1 if bad else 0)
PY

cp -p "$FIXTURE"/files/* "$root/corpus/"
if [[ $sabotage == 1 ]]; then
  sed -E "/\"qid\": \"$SABOTAGE_QID\"/s#\"gold\": \{[^}]*\}#\"gold\": {\"$SABOTAGE_GOLD\": 1}#" \
    "$FIXTURE/questions.jsonl" >"$root/data/questions.jsonl"
  grep -q "\"qid\": \"$SABOTAGE_QID\".*\"gold\": {\"$SABOTAGE_GOLD\": 1}" "$root/data/questions.jsonl" ||
    die "sabotage did not apply to $SABOTAGE_QID"
  echo "verify: sabotage on: $SABOTAGE_QID now names $SABOTAGE_GOLD as its gold, and the .doc sync has no converter;"
  echo "verify: the eval check and the .doc check must fail"
else
  cp -p "$FIXTURE/questions.jsonl" "$root/data/questions.jsonl"
fi

if run_step sync uv run docsearch sync --full; then
  cp -p "$root/work/meta.jsonl" "$evidence/meta_after_sync.jsonl" 2>/dev/null || true
  run_step meta uv run docsearch meta || true
  run_step search uv run docsearch search "$SEARCH_QUERY" || true
  run_step search_json bash -c 'uv run docsearch search "$1" --json >"$2"' search_json "$SEARCH_QUERY" \
    "$evidence/search.json" || true
  run_step search_format uv run docsearch search "$SEARCH_QUERY" --format "$FILTER_FORMAT" || true
  run_step search_after uv run docsearch search "$SEARCH_QUERY" --after "$FILTER_AFTER" || true
  run_step eval uv run docsearch eval --full --visual "$VISUAL" --fusion format_aware --mode plain || true
fi

# models.backend api against a deterministic stand-in server (no model, no network), with its own work folders.
api="$root/api"
mkdir -p "$api"
uv run python scripts/model_server.py --fake --port 0 --port-file "$api/port" --log "$evidence/api_requests.jsonl" \
  >"$evidence/api_server.txt" 2>&1 &
api_pid=$!
for _ in $(seq 100); do [[ -s "$api/port" ]] && break || sleep 0.1; done
api_url="http://127.0.0.1:$(cat "$api/port" 2>/dev/null || echo 9)"
api_env=(env DOCSEARCH_MODELS_BACKEND=api DOCSEARCH_MODELS_EMBED_API_URL="$api_url/v1"
  DOCSEARCH_MODELS_RERANK_API_URL="$api_url")
api_paths=(DOCSEARCH_PATHS_PARSED_DIR="$api/work/parsed" DOCSEARCH_PATHS_INDEX_DIR="$api/work/index"
  DOCSEARCH_PATHS_CACHE_DIR="$api/work/cache" DOCSEARCH_PATHS_RESULTS_DIR="$api/results"
  DOCSEARCH_PATHS_SYNC_MANIFEST="$api/work/manifest.sqlite" DOCSEARCH_LOGGING_DIR="$api/logs")
if run_step api_sync "${api_env[@]}" "${api_paths[@]}" uv run docsearch sync --full; then
  run_step api_search "${api_env[@]}" "${api_paths[@]}" uv run docsearch search "$SEARCH_QUERY" || true
  run_step api_eval "${api_env[@]}" "${api_paths[@]}" uv run docsearch eval --full --runs baseline_hybrid_rerank \
    --name "$API_RUN" --mode plain || true
fi
capture api_guard "${api_env[@]}" uv run docsearch search "$SEARCH_QUERY" || true
kill "$api_pid" 2>/dev/null || true
wait "$api_pid" 2>/dev/null || true

# An old Word file (.doc) made from the fixture handbook, next to a 0-byte .doc, with its own work folders.
doc="$root/doc"
mkdir -p "$doc/corpus"
doc_paths=(DOCSEARCH_PATHS_CORPUS_DIR="$doc/corpus" DOCSEARCH_PATHS_PARSED_DIR="$doc/work/parsed"
  DOCSEARCH_PATHS_INDEX_DIR="$doc/work/index" DOCSEARCH_PATHS_CACHE_DIR="$doc/work/cache"
  DOCSEARCH_PATHS_RESULTS_DIR="$doc/results" DOCSEARCH_PATHS_SYNC_MANIFEST="$doc/work/manifest.sqlite"
  DOCSEARCH_LOGGING_DIR="$doc/logs" DOCSEARCH_INGEST_DOC_CONVERTER="$([[ $sabotage == 1 ]] && echo off || echo auto)")
doc_ok=0
if command -v soffice >/dev/null; then
  run_step doc_build soffice --headless --norestore "-env:UserInstallation=file://$doc/profile" --convert-to doc \
    --outdir "$doc/corpus" "$FIXTURE/files/handbook.docx" && doc_ok=1
elif command -v textutil >/dev/null; then
  run_step doc_build textutil -convert doc -output "$doc/corpus/$DOC_FILE" "$FIXTURE/files/handbook.docx" && doc_ok=1
else
  echo "SKIP doc: no soffice or textutil on PATH, so no .doc was made and the .doc check did not run" | tee -a "$summary"
fi
if [[ $doc_ok == 1 ]]; then
  : >"$doc/corpus/$DOC_EMPTY"
  if run_step doc_sync env "${doc_paths[@]}" uv run docsearch sync --full; then
    run_step doc_search bash -c 'env "${@:3}" uv run docsearch search "$1" --json >"$2"' doc_search "$SEARCH_QUERY" \
      "$evidence/doc_search.json" "${doc_paths[@]}" || true
  fi
fi
# The search page: its own checks (Biome, bun test, build, Playwright in light and dark at two sizes, API mocked),
# then `docsearch serve` on this root with that build, one search through the API and the page at /.
ui_built=0
if command -v bun >/dev/null && [[ -d "$REPO/ui" ]]; then
  ui=(env -C "$REPO/ui")
  if run_step ui_install "${ui[@]}" bun install --frozen-lockfile; then
    run_step ui_lint "${ui[@]}" bunx biome check . || true
    run_step ui_test "${ui[@]}" bun test src || true
    if run_step ui_build "${ui[@]}" bash -c 'bunx tsc -b && bunx vite build --outDir "$1" --emptyOutDir' ui_build \
      "$root/ui_dist"; then
      ui_built=1
      run_step ui_e2e "${ui[@]}" UI_DIST="$root/ui_dist" UI_E2E_OUTPUT="$evidence/ui_e2e" bunx playwright test || true
    fi
  fi
else
  echo "SKIP ui: no bun on PATH, so the page was not linted, tested or built" | tee -a "$summary"
fi
web_ok=0
if uv run python -I -c 'import importlib.util as u, sys; sys.exit(not (u.find_spec("django") and u.find_spec("ninja")))'; then
  web_port=$(uv run python -I -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  DOCSEARCH_UI_DIST="$root/ui_dist" uv run docsearch serve --port "$web_port" >"$evidence/web_serve.txt" 2>&1 &
  web_pid=$!
  for _ in $(seq 120); do curl -sf "http://127.0.0.1:$web_port/api/config" >/dev/null && break; sleep 1; done
  if run_step web_search curl -sf -o "$evidence/web_search.json" --get "http://127.0.0.1:$web_port/api/search" \
    --data-urlencode "q=$SEARCH_QUERY"; then
    web_ok=1
    [[ $ui_built == 1 ]] && run_step web_page curl -sf -o "$evidence/web_index.html" "http://127.0.0.1:$web_port/"
    run_step web_file curl -sf -o "$evidence/web_file.bin" --get "http://127.0.0.1:$web_port/api/file" \
      --data-urlencode "file=$SEARCH_EXPECT" || true
    run_step web_config curl -sf -o "$evidence/web_config.json" "http://127.0.0.1:$web_port/api/config" || true
    run_step web_stream curl -sfN -o "$evidence/web_stream.ndjson" --get "http://127.0.0.1:$web_port/api/search/stream" \
      --data-urlencode "q=$SEARCH_QUERY" || true
  fi
  kill "$web_pid" 2>/dev/null || true
  wait "$web_pid" 2>/dev/null || true
else
  echo "SKIP web: the ui extra is not installed (uv sync --extra ui), so docsearch serve was not driven" | tee -a "$summary"
fi
touch "$evidence/.eval_unbuilt-start"
capture eval_unbuilt uv run docsearch eval --full --visual "$VISUAL" --fusion format_aware --mode "$UNBUILT_MODE" || true
touch "$evidence/.eval_research_guard-start"
capture eval_research_guard uv run docsearch eval --full --visual "$VISUAL" --fusion format_aware --agent-research || true
touch "$evidence/.eval_deck_guard-start"
capture eval_deck_guard uv run docsearch eval --full --visual "$VISUAL" --fusion format_aware --agent-deck || true

check check_sync sync --db "$DOCSEARCH_PATHS_SYNC_MANIFEST" --corpus "$root/corpus" --source "$SOURCE"
check check_meta_sync meta --file "$evidence/meta_after_sync.jsonl" --expect "$FIXTURE/meta.expected.jsonl" --label sync
check check_meta meta --file "$root/work/meta.jsonl" --expect "$FIXTURE/meta.expected.jsonl" --label "docsearch meta"
check check_search search --stdout "$evidence/search.txt" --expect "$SEARCH_EXPECT" --label "$SEARCH_LABEL"
check check_search_json json --file "$evidence/search.json" --stdout "$evidence/search.txt" --expect "$SEARCH_EXPECT"
check check_filter_format filter --stdout "$evidence/search_format.txt" --meta "$root/work/meta.jsonl" \
  --formats "$FILTER_FORMAT"
check check_filter_after filter --stdout "$evidence/search_after.txt" --meta "$root/work/meta.jsonl" \
  --after "$FILTER_AFTER" --absent "$SEARCH_EXPECT"
check check_eval eval --results "$root/results" --questions "$root/data/questions.jsonl" --run "$RUN_NAME" --mode plain
check check_eval_unbuilt refusal --output "$evidence/eval_unbuilt.txt" --expect "$UNBUILT_MSG" --root "$root" \
  --since "$evidence/.eval_unbuilt-start"
check check_eval_research_guard refusal --output "$evidence/eval_research_guard.txt" --expect "$RESEARCH_GUARD_MSG" \
  --root "$root" --since "$evidence/.eval_research_guard-start"
check check_eval_deck_guard refusal --output "$evidence/eval_deck_guard.txt" --expect "$DECK_GUARD_MSG" \
  --root "$root" --since "$evidence/.eval_deck_guard-start"
check check_api api --requests "$evidence/api_requests.jsonl" --results "$api/results" --run "$API_RUN" \
  --questions "$root/data/questions.jsonl" --guard "$evidence/api_guard.txt" \
  --steps "$evidence/api_sync.txt" "$evidence/api_search.txt" "$evidence/api_eval.txt"
check check_api_search search --stdout "$evidence/api_search.txt" --expect "$SEARCH_EXPECT"
if [[ $doc_ok == 1 ]]; then
  check check_doc doc --stdout "$evidence/doc_sync.txt" --db "$doc/work/manifest.sqlite" \
    --parse-log "$doc/work/parsed/parse_log.csv" --file "$evidence/doc_search.json" --expect "$DOC_EXPECT" \
    --empty "$DOC_EMPTY"
fi
if [[ $web_ok == 1 ]]; then
  check check_web web --file "$evidence/web_search.json" --cli-json "$evidence/search.json" \
    --opened "$evidence/web_file.bin" --source "$FIXTURE/files/${SEARCH_EXPECT#"$SOURCE/"}" --config "$evidence/web_config.json" \
    --stream "$evidence/web_stream.ndjson" \
    $([[ $ui_built == 1 ]] && printf -- '--page %s' "$evidence/web_index.html")
fi
check check_isolation isolation --repo "$REPO" --since "$root/$MARKER"

mkdir -p "$evidence/results/runs" "$evidence/logs"
find "$root/results" -maxdepth 1 -name runs.csv -exec cp -p {} "$evidence/results/" \;
find "$root/results/runs" -name '*.jsonl' -exec cp -p {} "$evidence/results/runs/" \; 2>/dev/null || true
find "$root/logs" -name '*.log' -exec cp -p {} "$evidence/logs/" \; 2>/dev/null || true

{
  echo
  echo "wall time: ${SECONDS}s"
  if [[ $failed == 0 ]]; then echo "RESULT: PASS"; else echo "RESULT: FAIL"; fi
  echo "evidence: $evidence"
} >>"$summary"

echo
echo "==== summary ($summary)"
cat "$summary"
if [[ $sabotage == 1 ]]; then echo "verify: sabotage run; a FAIL on $SABOTAGE_QID and on the .doc check is the expected outcome"; fi
exit "$failed"
