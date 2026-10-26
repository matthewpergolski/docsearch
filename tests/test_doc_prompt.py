import unittest

from docsearch.models import doc_prompt


class TestDocPrompt(unittest.TestCase):
    def test_declared_lead_in_is_used(self):
        self.assertEqual(doc_prompt({"query": "q: ", "document": "title: none | text: "}), {"prompt_name": "document"})

    def test_empty_or_missing_lead_in_sends_bare_text(self):
        self.assertEqual(doc_prompt({"query": "Instruct: ...", "document": ""}), {})  # Qwen3-Embedding
        self.assertEqual(doc_prompt({"query": "Represent ...", "passage": ""}), {})  # mxbai
        self.assertEqual(doc_prompt({}), {})


if __name__ == "__main__":
    unittest.main()
