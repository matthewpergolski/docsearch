"""Terminal UX: one progress line per stage, quiet libraries, a final summary.

On a TTY: rich progress bars (`[n/N] stage  ━━━━  done/total  rate  ETA`). Otherwise (nohup, CI, the box):
plain lines, throttled. Everything (library warnings, stray logging, our own detail lines) also goes to
logs/run-<timestamp>.log; only errors reach stderr.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

_state: dict = {"warnings": 0, "log": None, "path": None, "stage": "", "n": 0, "N": 0, "tty": None, "console": None}
logger = logging.getLogger("docsearch")


def is_tty() -> bool:
    if _state["tty"] is None:
        forced = os.environ.get("DOCSEARCH_PLAIN")
        _state["tty"] = (not forced) and sys.stdout.isatty()
    return _state["tty"]


def console():
    if _state["console"] is None:
        from rich.console import Console

        _state["console"] = Console(highlight=False)
    return _state["console"]


class _Counter(logging.Handler):
    """Counts warnings (captured `warnings.warn` and WARNING-level library records) for the end-of-run line."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == logging.WARNING or record.name == "py.warnings":
            _state["warnings"] += 1


def prune(log_dir: Path, keep: int) -> list[Path]:
    """Delete all but the newest `keep` run-*.log files. Other logs (phase1_*.log, ...) are never touched."""
    runs = sorted(log_dir.glob("run-*.log"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    old = runs[max(keep, 0) :]
    for p in old:
        p.unlink(missing_ok=True)
    return old


def setup(log_dir: Path, verbose: bool = False, keep: int = 20) -> Path:
    """Send every warning and library log record to a per-run log file. On screen: errors only, or
    everything at WARNING and above with verbose=True. No warning is filtered out."""
    if _state["path"]:
        return _state["path"]
    log_dir.mkdir(parents=True, exist_ok=True)
    prune(log_dir, keep - 1)  # room for this run's log
    path = log_dir / f"run-{datetime.now():%Y%m%d-%H%M%S}.log"
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    fh.setLevel(logging.INFO)
    err = logging.StreamHandler(sys.stderr)
    err._docsearch = True  # type: ignore[attr-defined]
    err.setLevel(logging.WARNING if verbose else logging.ERROR)
    err.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(fh)
    root.addHandler(err)
    counter = _Counter()
    counter.setLevel(logging.WARNING)
    root.addHandler(counter)
    root.setLevel(logging.INFO)
    logging.captureWarnings(True)  # warnings.warn -> logger "py.warnings" -> the run log
    warnings.simplefilter("always")  # log every occurrence; nothing is dropped
    _quiet_libraries()
    _state["path"] = path
    return path


def close() -> None:
    """Detach and close this run's handlers (main() is also called in-process by tests)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, (logging.FileHandler, _Counter)) or getattr(h, "_docsearch", False):
            root.removeHandler(h)
            h.close()
    logging.captureWarnings(False)
    _state.update(path=None, warnings=0)


def finish() -> None:
    """End-of-run line: how many warnings went to the log, and where."""
    if _state["path"]:
        n = _state["warnings"]
        say(f"{n} warning{'s' if n != 1 else ''} logged → {_state['path']}")


def quiet_libraries() -> None:
    """Re-apply after importing a library that installs its own console handler on import (RapidOCR does)."""
    if _state["path"]:
        _quiet_libraries()


def _quiet_libraries() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import transformers

        transformers.utils.logging.set_verbosity_warning()  # keep warnings; they go to the run log
        transformers.utils.logging.disable_progress_bar()  # "Loading weights" bars
        # transformers attaches its own stderr handler; route its records to the root (file) instead.
        tl = transformers.utils.logging.get_logger()
        for h in list(tl.handlers):
            tl.removeHandler(h)
        tl.propagate = True
    except Exception:  # noqa: BLE001 - transformers is optional for some commands
        pass
    for name in ("docling", "docling_core", "rapidocr", "RapidOCR", "lancedb", "httpx", "urllib3", "filelock",
                 "sentence_transformers", "huggingface_hub"):  # fmt: skip
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def disable_hub_bars() -> None:
    """Hide Hugging Face download bars (used when every model the command needs is already cached)."""
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # noqa: BLE001
        pass


def info(msg: str, *_, **__) -> None:
    """Detail line: always in the run log; on screen only in plain (non-TTY) mode."""
    logger.info(msg)
    if not is_tty():
        print(msg, flush=True)


def say(msg: str) -> None:
    """A line the user should see in both modes (device, stage headers, results)."""
    logger.info(msg)
    if is_tty():
        console().print(msg)
    else:
        print(msg, flush=True)


def stage_label(name: str) -> str:
    return f"[{_state['n']}/{_state['N']}] {name}" if _state["N"] else name


@contextmanager
def stage(n: int, total: int, name: str):
    _state.update(n=n, N=total, stage=name)
    t0 = time.time()
    if not is_tty():
        print(f"{stage_label(name)} start", flush=True)
    logger.info("stage %s start", stage_label(name))
    try:
        yield
    finally:
        el = time.time() - t0
        msg = f"{stage_label(name)} done in {_fmt(el)}"
        logger.info(msg)
        if is_tty():
            console().print(f"[green]✓[/green] {msg}")
        else:
            print(msg, flush=True)
        _state.update(n=0, N=0, stage="")


# Stage weights in seconds for `docsearch all`: measured on an Apple-silicon laptop, --full, MPS (2026-10-09);
# chunk/prepare/ingest/report were not measured there and are rough guesses. Replaced per machine by the
# stored timings of the last `all` run (results/stage_timings.json).
MAC_STAGE_SECONDS = {"prepare": 600, "ingest": 3600, "chunk": 150, "index text": 1512, "render": 501,
                     "index pages": 1578, "eval": 518, "report": 2}  # fmt: skip


def stage_key(name: str) -> str:
    """'index pages (pplx-v2-late)' -> 'index pages'; 'chunk hybrid' -> 'chunk'; 'index hybrid' -> 'index text'."""
    n = name.split(" (")[0]
    if n.startswith("chunk"):
        return "chunk"
    if n.startswith("index") and "pages" not in n:
        return "index text"
    if n.startswith("render"):
        return "render"
    return n


class Overall:
    """Weighted progress across the stages of `all`: `overall 2/6 · 38% · ~25 min left`."""

    def __init__(self, names: list[str], stored: dict | None = None, min_stored: float = 30.0):
        stored = stored or {}
        self.names = names
        # A stored timing under min_stored seconds is usually a resumed no-op stage; keep the default then.
        self.w = [float(stored[k]) if float(stored.get(k, 0) or 0) >= min_stored else float(MAC_STAGE_SECONDS.get(k, 60))
                  for k in map(stage_key, names)]  # fmt: skip
        self.i, self.frac = 0, 0.0

    def at(self, i: int, frac: float = 0.0) -> None:
        self.i, self.frac = i, min(max(frac, 0.0), 1.0)

    def line(self) -> str:
        total = sum(self.w) or 1.0
        done = sum(self.w[: self.i]) + (self.w[self.i] * self.frac if self.i < len(self.w) else 0.0)
        left = total - done
        eta = f"~{max(round(left / 60), 1)} min left" if left >= 60 else f"~{int(left)}s left"
        return f"overall {min(self.i + 1, len(self.w))}/{len(self.w)} · {done / total:.0%} · {eta}"


def set_overall(o: Overall | None) -> None:
    _state["overall"] = o


def _overall_suffix() -> str:
    o = _state.get("overall")
    return f"  ({o.line()})" if o else ""


@contextmanager
def loading(role: str, name: str = ""):
    """`⠋ loading <role> <model>…` spinner (TTY) that becomes `✓ <role> ready (Ns)`; plain lines otherwise."""
    label = f"{role} {name}".strip()
    t0 = time.time()
    logger.info("loading %s", label)
    if is_tty():
        with console().status(f"loading {label}…", spinner="dots"):
            yield
        el = time.time() - t0
        console().print(f"[green]✓[/green] {role} ready ({el:.0f}s)")
    else:
        print(f"loading {label}…", flush=True)
        yield
        el = time.time() - t0
        print(f"{role} ready ({el:.0f}s)", flush=True)
    logger.info("%s ready in %.1fs", label, el)


def _fmt(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else (f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s")


class Progress:
    """`with ui.progress(total, "embed") as p: p.advance(k)`. Rich bar on a TTY, throttled lines otherwise."""

    def __init__(self, total: int, what: str, unit: str = "", every: float = 15.0):
        self.total, self.what, self.unit, self.every = max(int(total), 0), what, unit, every
        self.done, self.t0, self._last = 0, time.time(), 0.0
        self._rich = None

    def __enter__(self):
        if is_tty() and self.total:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
            from rich.progress import (
                Progress as RP,
            )

            self._rich = RP(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("{task.fields[rate]}"),
                TextColumn("ETA"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                console=console(),
                transient=False,
            )
            self._rich.start()
            self._task = self._rich.add_task(self._label(), total=self.total, rate="")
        return self

    def _label(self) -> str:
        lab = stage_label(_state["stage"]) if _state["stage"] else ""
        return f"{lab} {self.what}".strip() if self.what and self.what != _state["stage"] else lab or self.what

    def rate(self) -> float:
        el = time.time() - self.t0
        return self.done / el if el > 0 else 0.0

    def advance(self, k: int = 1) -> None:
        self.done += k
        o = _state.get("overall")
        if o and self.total:
            o.at(o.i, self.done / self.total)
        r = self.rate()
        rate = f"{r:.2f}/s" if r < 10 else f"{r:.0f}/s"
        if self._rich:
            self._rich.update(self._task, advance=k, rate=rate, description=self._label() + _overall_suffix())
            return
        now = time.time()
        if now - self._last >= self.every or self.done >= self.total:
            self._last = now
            eta = (self.total - self.done) / r if r > 0 else 0
            print(f"{self._label()}: {self.done}/{self.total} {self.unit}{rate}, ETA {_fmt(eta)}".replace("  ", " ")
                  + _overall_suffix(), flush=True)  # fmt: skip

    def start(self) -> Progress:
        return self.__enter__()

    def close(self) -> None:
        self.__exit__(None, None, None)

    def __exit__(self, *exc):
        if self._rich:
            self._rich.stop()
        logger.info("%s: %d/%d in %.1fs", self._label(), self.done, self.total, time.time() - self.t0)
        return False


def progress(total: int, what: str = "", unit: str = "") -> Progress:
    return Progress(total, what, unit)


def summary_table(rows: list[dict], report: Path | None) -> None:
    """Final recall@10 / nDCG@10 / MRR per run, plus where the report is."""
    if not rows:
        return
    if is_tty():
        from rich.table import Table

        t = Table(title="Results (this run)", show_lines=False)
        t.add_column("run")
        for c in ("recall@10", "nDCG@10", "MRR@10", "n"):
            t.add_column(c, justify="right")
        for r in rows:
            t.add_row(r["name"], f"{r['recall@10']:.3f}", f"{r['ndcg@10']:.3f}", f"{r['mrr@10']:.3f}",
                      str(r["n_queries"]))  # fmt: skip
        console().print(t)
    else:
        print(f"{'run':48s} {'recall@10':>9s} {'nDCG@10':>8s} {'MRR@10':>7s} {'n':>4s}")
        for r in rows:
            print(f"{r['name']:48s} {r['recall@10']:9.3f} {r['ndcg@10']:8.3f} {r['mrr@10']:7.3f} {r['n_queries']:4d}")
    if report:
        say(f"Report: {report}")
