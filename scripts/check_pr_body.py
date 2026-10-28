#!/usr/bin/env python3
"""Fail when a pull-request body does not meet the docsearch PR template rules.

Requires, in order, the six ``##`` headings from
``.github/pull_request_template.md``:

  Summary, Motivation, Changes, Test plan, Risk / rollback, Checklist

Also requires a fenced mermaid block (`` ```mermaid ``), and rejects Cursor
footer markers, the string ``Generated with``, and model ids.

Usage:
  python3 scripts/check_pr_body.py                 # read GITHUB_EVENT_PATH
  python3 scripts/check_pr_body.py path/to/body.md
  python3 scripts/check_pr_body.py --body-text "..."

Exit 0 when clean, 1 when problems are found, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Headings the template and CI both enforce, in order.
REQUIRED_HEADINGS: tuple[str, ...] = (
    "Summary",
    "Motivation",
    "Changes",
    "Test plan",
    "Risk / rollback",
    "Checklist",
)

# ATX heading line: optional up to three spaces, then ##, then the title.
_HEADING = re.compile(r"^ {0,3}## (.+?)[ \t]*#*[ \t]*$")

# Opening fence for a mermaid diagram (info string may have trailing attrs).
_MERMAID_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})mermaid\b", re.I | re.M)

# Cursor / agent footer stamps that must not appear in a PR body.
_CURSOR_FOOTER = re.compile(
    r"(?:"
    r"CURSOR_AGENT_PR_BODY_(?:BEGIN|END)"
    r"|<!--\s*cursor[\w.-]*\s*-->"
    r"|made with\s+cursor"
    r"|cursor\s+agent"
    r")",
    re.I,
)

_GENERATED_WITH = re.compile(r"Generated with")

# Model ids / product model slugs (not ordinary prose like "the model loads").
_MODEL_ID = re.compile(
    r"(?i)\b(?:"
    r"claude-(?:opus|sonnet|haiku|[\w.-]+)"
    r"|gpt-[\w.-]+"
    r"|o[1-9](?:-[\w.-]+)?"
    r"|gemini-[\w.-]+"
    r"|composer-[\w.-]+"
    r"|grok-[\w.-]+"
    r"|muse-[\w.-]+"
    r"|chatgpt-[\w.-]+"
    r"|anthropic\.claude[\w.-]*"
    r"|cursor-grok-[\w.-]+"
    r")\b",
)


def _strip_fences(body: str) -> str:
    """Return body with fenced code blocks replaced by blank lines (keep line count)."""
    lines = body.splitlines(keepends=True)
    out: list[str] = []
    fence: str | None = None
    for line in lines:
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
                out.append("\n" if line.endswith("\n") else "")
                continue
            if token[0] == fence[0] and len(token) >= len(fence) and not line[marker.end() :].strip():
                fence = None
                out.append("\n" if line.endswith("\n") else "")
                continue
        if fence is not None:
            out.append("\n" if line.endswith("\n") else "")
        else:
            out.append(line)
    return "".join(out)


def heading_titles(body: str) -> list[str]:
    """Return top-level ``##`` heading titles in document order (ignore fenced text)."""
    titles: list[str] = []
    for line in _strip_fences(body).splitlines():
        match = _HEADING.match(line)
        if match:
            titles.append(match.group(1).strip())
    return titles


def check_headings(body: str) -> list[str]:
    """Return errors when the six required headings are missing or out of order."""
    titles = heading_titles(body)
    errors: list[str] = []
    pos = 0
    for required in REQUIRED_HEADINGS:
        try:
            idx = titles.index(required, pos)
        except ValueError:
            errors.append(f"missing heading in order: ## {required}")
            continue
        pos = idx + 1
    return errors


def check_mermaid(body: str) -> list[str]:
    """Return errors when the body has no mermaid fence."""
    if _MERMAID_FENCE.search(body):
        return []
    return ["missing mermaid fence (```mermaid)"]


def check_forbidden(body: str) -> list[str]:
    """Return errors for Cursor footers, 'Generated with', and model ids."""
    errors: list[str] = []
    if _CURSOR_FOOTER.search(body):
        errors.append("forbidden Cursor footer marker in PR body")
    if _GENERATED_WITH.search(body):
        errors.append('forbidden string "Generated with" in PR body')
    if _MODEL_ID.search(body):
        errors.append("forbidden model id in PR body")
    return errors


def check_body(body: str) -> list[str]:
    """Return all problems for a PR body (empty when the body is clean)."""
    text = body if body is not None else ""
    return check_headings(text) + check_mermaid(text) + check_forbidden(text)


def _body_from_event(path: Path) -> str:
    event = json.loads(path.read_text(encoding="utf-8"))
    pr = event.get("pull_request") or {}
    return pr.get("body") or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        help="markdown file with the PR body (default: GITHUB_EVENT_PATH)",
    )
    parser.add_argument(
        "--body-text",
        help="check this string instead of a file or event payload",
    )
    args = parser.parse_args(argv)

    if args.body_text is not None:
        body = args.body_text
    elif args.path:
        body = Path(args.path).read_text(encoding="utf-8")
    else:
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if not event_path:
            print(
                "usage: check_pr_body.py [body.md] | --body-text TEXT | set GITHUB_EVENT_PATH",
                file=sys.stderr,
            )
            return 2
        body = _body_from_event(Path(event_path))

    problems = check_body(body)
    if not problems:
        print("pr-body: ok")
        return 0
    print(f"pr-body: {len(problems)} problem(s)", file=sys.stderr)
    for line in problems:
        print(f"error: {line}", file=sys.stderr)
        print(f"::error::{line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
