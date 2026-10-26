"""docsearch: mixed-format document search + RAG test bench.

Warnings are not filtered here: `ui.setup` captures every warning (including Docling's off-page bbox
clamp notices) into logs/run-<timestamp>.log and counts them; `--verbose` shows them live.
"""


def main() -> int:
    from .cli import main as _main

    return _main()
