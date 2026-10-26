import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from docsearch.models import named_chat_template


def fake(chat_template, want="reranker"):
    kw = {"chat_template": {"chat_template": want, "add_generation_prompt": True}} if want else {}
    return [SimpleNamespace(processing_kwargs=kw, processor=SimpleNamespace(chat_template=chat_template))]


class TestNamedChatTemplate(unittest.TestCase):
    def test_missing_named_template_is_read_from_the_cache(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "reranker.jinja"
            f.write_text("<Query>: {{ q }}")
            m = fake("generic template")
            with mock.patch("huggingface_hub.try_to_load_from_cache", return_value=str(f)) as hub:
                named_chat_template(m, "org/model")
            hub.assert_called_once_with("org/model", "additional_chat_templates/reranker.jinja")
        self.assertEqual(m[0].processor.chat_template, {"default": "generic template", "reranker": "<Query>: {{ q }}"})

    def test_a_loaded_named_template_is_left_alone(self):
        m = fake({"default": "a", "reranker": "b"})
        with mock.patch("huggingface_hub.try_to_load_from_cache") as hub:
            named_chat_template(m, "org/model")
        hub.assert_not_called()
        self.assertEqual(m[0].processor.chat_template, {"default": "a", "reranker": "b"})

    def test_no_named_template_or_no_cached_file_changes_nothing(self):
        m = fake("generic template", want=None)
        named_chat_template(m, "org/model")
        self.assertEqual(m[0].processor.chat_template, "generic template")
        m = fake("generic template")
        with mock.patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            named_chat_template(m, "org/model")
        self.assertEqual(m[0].processor.chat_template, "generic template")


if __name__ == "__main__":
    unittest.main()
