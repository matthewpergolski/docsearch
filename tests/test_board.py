"""scripts/board.py page tokens: the page body uses only font tokens the board defines, with neutral names."""

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _board():
    spec = importlib.util.spec_from_file_location("board", ROOT / "scripts" / "board.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBoardFonts(unittest.TestCase):
    def test_font_tokens_have_neutral_names_and_cover_the_page(self):
        defined = set(re.findall(r"(--font-[a-z-]+):", _board().SIZES))
        used = set(re.findall(r"var\((--font-[a-z-]+)\)", (ROOT / "scripts" / "board_page.html").read_text()))
        self.assertEqual(defined, {"--font-sans", "--font-serif"})
        self.assertLessEqual(used, defined)


if __name__ == "__main__":
    unittest.main()
