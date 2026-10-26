## What changed and why

## How you checked it
<!-- Commands and their results. Label each number: measured, inferred or guess. -->

## Checklist
- [ ] `uv run ruff check src tests scripts` and `uv run ruff format --check src tests scripts` pass.
- [ ] `uv run python -m unittest discover -s tests` passes. A bug fix comes with a test that failed before it.
- [ ] Tests use fakes: no real models, no downloads, no `data/`.
- [ ] A search change is a flag or config key, off by default, with the paired bootstrap result
      (see "Running an experiment" in CONTRIBUTING.md).
- [ ] A new model or dependency has a permissive license, written next to its alias in `docsearch.yaml`
      and in `THIRD-PARTY.md`.
- [ ] No files from `data/`, `corpus/`, `work/`, `logs/` or `results/runs/`, and no keys or personal paths.
