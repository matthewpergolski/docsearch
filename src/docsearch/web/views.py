"""The built page (ui/dist, or DOCSEARCH_UI_DIST) and its assets."""

import os
from pathlib import Path

from django.http import FileResponse, Http404, HttpResponse


def dist() -> Path:
    return Path(os.environ.get("DOCSEARCH_UI_DIST") or Path(__file__).resolve().parents[3] / "ui" / "dist")


def index(request):
    page = dist() / "index.html"
    if not page.is_file():
        return HttpResponse(f"The search page is not built ({page} is missing). Build it: cd ui && bun install && "
                            "bun run build. The API works without it, at /api/.", status=503,
                            content_type="text/plain; charset=utf-8")  # fmt: skip
    return HttpResponse(page.read_bytes(), content_type="text/html; charset=utf-8")


def asset(request, path: str):
    root = (dist() / "assets").resolve()
    p = (root / path).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise Http404(path)
    return FileResponse(p.open("rb"))
