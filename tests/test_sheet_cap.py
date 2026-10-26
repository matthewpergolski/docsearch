import unittest
from types import SimpleNamespace

import pandas as pd

from docsearch.chunk import chunk_table_rows, share_cap
from docsearch.manifest import Doc


def count(s: str) -> int:
    return len(s.split())


class FakeTable:
    def __init__(self, name: str, n_rows: int):
        self.prov, self.parent = [], None
        self.df = pd.DataFrame({"name": [f"{name}{i}" for i in range(n_rows)], "v": list(range(n_rows))})

    def export_to_dataframe(self, doc=None):
        return self.df


def workbook(*sizes: int):
    return SimpleNamespace(tables=[FakeTable(f"t{i}r", n) for i, n in enumerate(sizes)])


def doc_row() -> Doc:
    fields = {f: "" for f in Doc.__dataclass_fields__}
    fields.update(doc_id="ocb/book.xlsx", path="ocb/book.xlsx", format="xlsx", source="ocb")
    return Doc(**fields)


class TestShareCap(unittest.TestCase):
    def test_under_the_cap_keeps_all(self):
        self.assertEqual(share_cap([3, 2, 1], 10), [3, 2, 1])

    def test_big_first_table_leaves_room_for_the_rest(self):
        self.assertEqual(share_cap([100, 4, 2], 12), [6, 4, 2])

    def test_cap_smaller_than_the_table_count(self):
        self.assertEqual(share_cap([5, 5, 5], 2), [1, 1, 0])


class TestChunkTableRows(unittest.TestCase):
    # max_tokens 12 with the fake count: the 7-token prefix leaves room for 1 row per chunk
    def tables_in(self, chunks):
        return [c.meta["table"] for c in chunks]

    def test_plain_cap_gives_every_slot_to_the_first_table(self):
        got = chunk_table_rows(workbook(40, 4, 4), doc_row(), count, 12, max_chunks=6)
        self.assertEqual(self.tables_in(got), [1] * 6)

    def test_spread_keeps_every_table(self):
        got = chunk_table_rows(workbook(40, 4, 4), doc_row(), count, 12, max_chunks=6, spread=True)
        self.assertEqual(self.tables_in(got), [1, 1, 2, 2, 3, 3])
        self.assertEqual([c.chunk_id for c in got], [f"hybrid:ocb/book.xlsx:{i}" for i in range(6)])

    def test_spread_changes_nothing_under_the_cap(self):
        a = chunk_table_rows(workbook(6, 4), doc_row(), count, 12, max_chunks=50)
        b = chunk_table_rows(workbook(6, 4), doc_row(), count, 12, max_chunks=50, spread=True)
        self.assertEqual([c.__dict__ for c in a], [c.__dict__ for c in b])
        self.assertEqual(self.tables_in(a), [1] * 6 + [2] * 4)


if __name__ == "__main__":
    unittest.main()
