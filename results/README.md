# results/

Experiment outputs from development. The numbers come from two kinds of data.

- **Published benchmark sets.** Almost every number here was measured on the public benchmark sets used while
  building docsearch (ViDoRe, FinSlides, Office Comprehension Bench, and public-domain images; see the main README
  for sources and licenses), plus question sets drafted from those same documents. `uv run docsearch prepare`
  downloads the documents again. The drafted question files are in `data/`, which is not tracked; draft your own
  with `uv run docsearch questions draft` (see the main README).
- **A real file collection that is not published.** Phase F uses a held-out set on a real file collection that is
  not published. Its files, questions, index and run outputs are not in this repo, and no row of `runs.csv` comes
  from it. Only these results come from it: row F in `phases.tsv` (and on the board), and the two F2 rows in
  `../decisions.tsv`.

- `runs.csv`: one row per eval run (config, model ids, device, metrics).
- `report.html`: the eval report from the latest runs.
- `timings.json`: stage timings.
- `experiments.tsv`, `pipelines.tsv`: hand-written labels for the experiment board.
- `board.html`, `board/*.json`: the experiment board, rebuilt by `uv run python scripts/board.py`. The board page
  loads d3 from a CDN (cdnjs), so it needs network access.

Per-run detail (`results/runs/`) is not tracked. Your own runs append to these files; delete them to start fresh.
