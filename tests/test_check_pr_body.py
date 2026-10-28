"""PR-body gate: bad bodies fail, a filled template body passes."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_pr_body.py"
TEMPLATE = ROOT / ".github" / "pull_request_template.md"


def _load():
    spec = importlib.util.spec_from_file_location("check_pr_body", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _good_body() -> str:
    """A filled body that should pass every rule."""
    return """## Summary

Add a CI check that enforces the PR template.

## Motivation

Reviewers need a stable body shape and no agent attribution stamps.

## Changes

- New template with six headings.
- `scripts/check_pr_body.py` and a workflow that runs it.

```mermaid
flowchart LR
  template[PR template] --> check[check_pr_body.py]
  check --> ci[pr-body workflow]
```

## Test plan

- `uv run python -m unittest tests.test_check_pr_body` (measured).
- Open this PR and confirm the pr-body job is green (measured).

## Risk / rollback

Low: CI-only. Revert the workflow and script to disable the gate.

## Checklist

- [x] Unit tests cover pass and fail cases.
- [x] No agent footer stamps, no generation attribution, no model ids.
"""


class TestCheckPrBody(unittest.TestCase):
    def test_required_headings_match_template(self):
        titles = mod.heading_titles(TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(list(mod.REQUIRED_HEADINGS), titles[:6])
        self.assertEqual(len(mod.REQUIRED_HEADINGS), 6)

    def test_template_has_mermaid_placeholder(self):
        self.assertTrue(mod._MERMAID_FENCE.search(TEMPLATE.read_text(encoding="utf-8")))

    def test_good_body_passes(self):
        self.assertEqual(mod.check_body(_good_body()), [])

    def test_missing_heading_fails(self):
        body = _good_body().replace("## Motivation\n\nReviewers need", "## Motives\n\nReviewers need")
        errors = mod.check_body(body)
        self.assertTrue(any("Motivation" in e for e in errors), errors)

    def test_out_of_order_headings_fail(self):
        body = _good_body()
        body = body.replace("## Summary\n", "## Motivation\n", 1)
        body = body.replace("## Motivation\n\nReviewers", "## Summary\n\nReviewers", 1)
        # After swap the first required heading Summary appears after Motivation.
        errors = mod.check_body(body)
        self.assertTrue(any("Summary" in e or "Motivation" in e for e in errors), errors)

    def test_missing_mermaid_fails(self):
        body = _good_body().replace(
            "```mermaid\nflowchart LR\n  template[PR template] --> check[check_pr_body.py]\n  check --> ci[pr-body workflow]\n```",
            "no diagram",
        )
        errors = mod.check_body(body)
        self.assertTrue(any("mermaid" in e for e in errors), errors)

    def test_heading_inside_fence_ignored(self):
        body = _good_body().replace(
            "## Test plan\n",
            "```\n## Test plan\n```\n\n## Test plan\n",
            1,
        )
        # Still has a real Test plan heading after the fence, so should pass.
        self.assertEqual(mod.check_body(body), [])

    def test_cursor_footer_marker_fails(self):
        body = _good_body() + "\n<!-- CURSOR_AGENT_PR_BODY_END -->\n"
        errors = mod.check_body(body)
        self.assertTrue(any("Cursor footer" in e for e in errors), errors)

    def test_generated_with_fails(self):
        body = _good_body() + "\nGenerated with assistance.\n"
        errors = mod.check_body(body)
        self.assertTrue(any("Generated with" in e for e in errors), errors)

    def test_model_id_fails(self):
        body = _good_body() + "\nWrote this with gpt-4o-mini overnight.\n"
        errors = mod.check_body(body)
        self.assertTrue(any("model id" in e for e in errors), errors)

    def test_main_accepts_body_text(self):
        self.assertEqual(mod.main(["--body-text", _good_body()]), 0)
        self.assertEqual(mod.main(["--body-text", "## Summary\nonly\n"]), 1)


if __name__ == "__main__":
    unittest.main()
