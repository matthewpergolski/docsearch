import unittest

from docsearch.models import seq_limit


class TestSeqLimit(unittest.TestCase):
    def test_long_context_model_keeps_the_chunk_length(self):
        self.assertEqual(seq_limit(544, 32768), 544)

    def test_bert_model_is_held_to_its_limit(self):
        self.assertEqual(seq_limit(544, 512), 512)

    def test_no_model_limit(self):
        self.assertEqual(seq_limit(544, None), 544)


if __name__ == "__main__":
    unittest.main()
