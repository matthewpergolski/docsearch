#!/usr/bin/env bash
# Build the search page and start `docsearch serve` (README "Search page"). Arguments go to `docsearch serve`,
# for example: scripts/serve.sh --port 9000
# On Linux, set DOCSEARCH_TORCH to the torch extra you synced with (cpu, or cuda for an NVIDIA GPU):
#   DOCSEARCH_TORCH=cuda scripts/serve.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

command -v uv >/dev/null || { echo "serve.sh: uv is not on PATH" >&2; exit 1; }
command -v bun >/dev/null || { echo "serve.sh: bun is not on PATH (https://bun.sh), so the page cannot be built" >&2; exit 1; }

# `--extra ui` adds Django and Django Ninja. Without the torch extra, `uv run` re-syncs to the default torch.
extras=(--extra ui)
case "${DOCSEARCH_TORCH:-}" in
  "") ;;
  cpu | cuda) extras+=(--extra "$DOCSEARCH_TORCH") ;;
  *) echo "serve.sh: DOCSEARCH_TORCH must be cpu or cuda (got $DOCSEARCH_TORCH)" >&2; exit 1 ;;
esac

(cd "$REPO/ui" && bun install --frozen-lockfile && bun run build)
cd "$REPO" && exec uv run "${extras[@]}" docsearch serve "$@"
