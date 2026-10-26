import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from docsearch.cli import timed
from docsearch.config import load_config


def _run(cfg, step: str, seconds: float) -> None:
    with mock.patch("docsearch.cli.time") as clock:  # cli's name only: logging also calls time.time
        clock.time.side_effect = [0.0, seconds]
        with contextlib.redirect_stdout(io.StringIO()), timed(cfg, step):
            pass


def _recorded(results: Path, step: str) -> list[float]:
    """Every value stored for step in any JSON file in results, nested or flat."""
    out: list[float] = []

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            elif k == step:
                out.append(v)

    for p in results.glob("*.json"):
        walk(json.loads(p.read_text()))
    return sorted(out)


class TestTrialTimings(unittest.TestCase):
    def test_two_trials_do_not_clobber_each_others_timings(self):
        """Trials redirect parsed/index dirs but share results_dir: one trial's ingest time must not replace another's."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "docsearch.yaml"
            p.write_text("paths: {results_dir: results}\n")
            _run(load_config(p, env={"DOCSEARCH_TRIAL_NAME": "pn1"}), "ingest", 53.0)
            _run(load_config(p, env={"DOCSEARCH_TRIAL_NAME": "ps1"}), "ingest", 848.0)
            self.assertEqual(_recorded(Path(td) / "results", "ingest"), [53.0, 848.0])

    def test_default_run_keeps_the_flat_timings_file(self):
        """No trial name: the old flat results/timings.json still loads, and its other keys stay."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "docsearch.yaml"
            p.write_text("paths: {results_dir: results}\n")
            tp = Path(td) / "results" / "timings.json"
            tp.parent.mkdir()
            tp.write_text(json.dumps({"ingest": 53.2, "eval": 70.3}))
            _run(load_config(p, env={}), "eval", 71.0)
            self.assertEqual(json.loads(tp.read_text()), {"ingest": 53.2, "eval": 71.0})


if __name__ == "__main__":
    unittest.main()
