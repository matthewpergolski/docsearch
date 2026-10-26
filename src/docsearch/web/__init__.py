"""The local search page: a Django + Django Ninja API over the same search as `docsearch search --json`, and the
built React page (ui/dist). Needs the ui extra: `uv sync --extra ui`."""

from __future__ import annotations

import os


def setup() -> None:
    try:
        import django
    except ImportError:
        raise SystemExit("the search page needs the ui extra: uv sync --extra ui") from None
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "docsearch.web.settings")
    django.setup()


def serve(cfg, host: str, port: int) -> int:
    """Run the page and its API until stopped. Models load in the background at start and stay loaded."""
    setup()
    import threading

    from django.core.management import call_command

    from .. import models
    from . import api

    api.CFG = cfg
    models.keep_loaded(True)
    threading.Thread(target=api.warm, args=(cfg,), daemon=True, name="warm").start()
    call_command("runserver", f"{host}:{port}", use_reloader=False)
    return 0
