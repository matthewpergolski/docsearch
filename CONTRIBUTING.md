# Contributing

## Setup and checks
- Python runs through [uv](https://docs.astral.sh/uv/): `uv sync`. Do not use pip; do not edit `.venv/`.
- Before a pull request, run both and make them pass:
  - `uv run ruff check src tests scripts` and `uv run ruff format --check src tests scripts`
  - `uv run python -m unittest discover -s tests`
- Write a failing test first for a bug fix, then the fix.
- Keep changes small and on task. Match the local style.
- The default pipeline uses only models with a permissive license (Apache-2.0, MIT or CDLA-Permissive-2.0). A
  new default must keep to this rule.
- An optional challenger may carry other terms, for example a use policy. It stays off by default, and
  `THIRD-PARTY.md` states its terms.
- Write the license next to the alias in `docsearch.yaml`. Add each new model to `THIRD-PARTY.md` with its SPDX
  id and the source URL you checked.
- `data/`, `corpus/`, `work/`, `results/runs/` and `logs/` are ignored by git and can hold hours of compute
  (parses, vectors, score caches).

## Running an experiment
The method is hill-climbing: one change at a time against the locked default.
- A change is a flag or config key that is off by default. The default changes only when the change is adopted.
- The run files of this repo are not published. `scripts/bootstrap_delta.py` and `scripts/board.py` read the
  per-question run files in `results/runs/`, so make your own runs first. Build the corpus and the index with
  `uv run docsearch all --with-ingest --full` (`docsearch.yaml` turns the box subset on, so without `--full` the
  index is partial).
- Make the reference runs once, with the default settings and no flag. First the public set:
  `uv run docsearch eval --full --visual pplx-v2-late --fusion format_aware --name base`
  Then draft a hard question set from that run (`uv run docsearch questions draft --hard --hard-from base`, then
  review it; README, "Hard-page questions"). The `hard_missed` questions need the per-question file of `base`.
  Last, the hard set: the same eval with `--questions hard --reviewed-only --name base_hard`.
- Run the change on the public set and on your hard set:
  `uv run docsearch eval --full --visual pplx-v2-late --fusion format_aware <flag> --name <id>`
  and the same with `--questions hard --reviewed-only --name <id>_hard`.
- Compare with the paired bootstrap, per format and per question type:
  `uv run python scripts/bootstrap_delta.py base <id> --by format`
  `uv run python scripts/bootstrap_delta.py base_hard <id>_hard --by type`
- `results/experiments.tsv` names `m28_default` and `m28_default_hard` as the reference runs of this repo. These
  are the runs `20261013-214948-hybrid_visual_fa_pplx-v2-late_rerank` (public, 734 questions) and
  `20261013-215027-hybrid_visual_fa_pplx-v2-late_rerank` (hard, 87 questions) in `results/runs.csv`.
- Table rule: a change that works only on table cells is adopted when nDCG@10 on the table-cell questions (the
  hard set's table questions plus a held-out table set) has a paired-bootstrap interval above 0 and the public set
  does not fall. Declare it as a table change before the run.
- Adopt only when the overall nDCG@10 interval on the public set is above zero and the hard set does not fall.
  A gain on one format with no overall loss may be adopted for that format only.
- Declare the setting you will judge before the run. Report other settings as exploration, not as results.
- A change that learns from questions (fine-tuning, a tuned threshold) needs held-out questions it never saw.
- Label numbers in write-ups as measured (from a run), inferred, or a guess.

## Project notes
- `decisions.tsv`: one row per try, rejected ones too, with the numbers and the verdict. Append only.
- `REPORT.md`: the research report. A finished try gets a row in a stage table.
- `results/experiments.tsv` + `scripts/board.py`: the experiment board. Add a row, then run
  `uv run python scripts/board.py` to rebuild `results/board.html` and `results/board/*.json` from the run files.
  The script skips each row whose run files are not in `results/runs/`. Without the run files of this repo, a
  rebuild shows only your own rows, so keep the committed board and rebuild it only from your own runs.
- `README.md`: the commands.
