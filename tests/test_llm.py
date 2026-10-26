import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from docsearch.answer import NOT_FOUND, answer, build_prompt, parse_citations
from docsearch.config import load_config
from docsearch.draft_questions import draft, extract_json
from docsearch.llm import (
    LLMClient,
    LLMError,
    LLMSettings,
    LLMUnavailable,
    check_available,
    cli_command,
    settings,
)
from docsearch.manifest import Doc, write_manifest

FAKE_KEY = "sk-test-SECRET-should-never-appear"


def cfg_in(td: str, extra: str = ""):
    p = Path(td) / "docsearch.yaml"
    p.write_text(
        "paths: {corpus_dir: corpus, data_dir: data}\n"
        "llm:\n  backend: api\n  provider: claude\n  timeout_s: 5\n"
        "  api_models: {claude: {answer: anthropic/claude-sonnet-5-5, draft: anthropic/claude-haiku-5-5},"
        " grok: {answer: xai/grok-4.7}, ollama: {answer: ollama/qwen3:8b}}\n"
        "  cli_models: {claude: {answer: sonnet, draft: haiku}}\n" + extra
    )
    return load_config(p, env={})


def fake_completion(text):
    calls = []

    def completion(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    return completion, calls


class FakeRunner:
    def __init__(self, stdout="ok answer", returncode=0, stderr="", write_last=None, exc=None):
        self.calls = []
        self.stdout, self.returncode, self.stderr, self.write_last, self.exc = (
            stdout,
            returncode,
            stderr,
            write_last,
            exc,
        )

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        if self.exc:
            raise self.exc
        if self.write_last is not None and "-o" in cmd:
            Path(cmd[cmd.index("-o") + 1]).write_text(self.write_last)
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)


def have(_):
    return "/usr/local/bin/x"


def missing(_):
    return None


class TestSettings(unittest.TestCase):
    def test_resolve_and_env_override(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = cfg_in(td)
            s = settings(cfg, "draft")
            self.assertEqual((s.backend, s.provider, s.model), ("api", "claude", "anthropic/claude-haiku-5-5"))
            cfg2 = load_config(Path(td) / "docsearch.yaml", env={"DOCSEARCH_LLM_PROVIDER": "grok"})
            self.assertEqual(settings(cfg2).model, "xai/grok-4.7")
            self.assertEqual(settings(cfg, backend="cli").model, "sonnet")
            self.assertEqual(settings(cfg, provider="ollama").api_base, "http://localhost:11434")

    def test_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = cfg_in(td)
            with self.assertRaises(ValueError):
                settings(cfg, backend="cli", provider="ollama")
            with self.assertRaises(ValueError):
                settings(cfg, provider="gemini")


class TestAvailability(unittest.TestCase):
    def test_missing_key_message_names_var_not_value(self):
        s = LLMSettings("api", "openai", "openai/gpt-5.6", 5)
        with self.assertRaises(LLMUnavailable) as cm:
            check_available(s, env={})
        self.assertIn("OPENAI_API_KEY", str(cm.exception))
        check_available(s, env={"OPENAI_API_KEY": FAKE_KEY})  # no raise

    def test_ollama_needs_no_key(self):
        check_available(LLMSettings("api", "ollama", "ollama/qwen3:8b", 5), env={})

    def test_cli_binary_missing(self):
        with self.assertRaises(LLMUnavailable) as cm:
            check_available(LLMSettings("cli", "openai", "", 5), which=missing)
        self.assertIn("codex", str(cm.exception))


class TestApiBackend(unittest.TestCase):
    def test_litellm_call_shape(self):
        comp, calls = fake_completion(" The rate was 5% [1]. ")
        c = LLMClient(LLMSettings("api", "claude", "anthropic/claude-sonnet-5-5", 7), completion=comp,
                      env={"ANTHROPIC_API_KEY": FAKE_KEY})  # fmt: skip
        out = c.complete("Q?", system="SYS")
        self.assertEqual(out, "The rate was 5% [1].")
        kw = calls[0]
        self.assertEqual(kw["model"], "anthropic/claude-sonnet-5-5")
        self.assertEqual(kw["timeout"], 7)
        self.assertEqual(kw["messages"][0], {"role": "system", "content": "SYS"})
        self.assertNotIn(FAKE_KEY, json.dumps(kw))

    def test_ollama_passes_api_base(self):
        comp, calls = fake_completion("hi")
        LLMClient(LLMSettings("api", "ollama", "ollama/qwen3:8b", 5, "http://localhost:11434"), completion=comp,
                  env={}).complete("x")  # fmt: skip
        self.assertEqual(calls[0]["api_base"], "http://localhost:11434")

    def test_auth_error_maps_to_unavailable_without_key(self):
        class AuthenticationError(Exception):
            pass

        def comp(**kw):
            raise AuthenticationError(f"bad key {FAKE_KEY}")

        c = LLMClient(LLMSettings("api", "grok", "xai/grok-4.7", 5), completion=comp, env={"XAI_API_KEY": FAKE_KEY})
        with self.assertRaises(LLMUnavailable) as cm:
            c.complete("x")
        self.assertNotIn(FAKE_KEY, str(cm.exception))

    def test_other_error_is_llmerror_without_key(self):
        def comp(**kw):
            raise RuntimeError(f"boom {FAKE_KEY}")

        c = LLMClient(LLMSettings("api", "openai", "openai/gpt-5.6", 5), completion=comp,
                      env={"OPENAI_API_KEY": FAKE_KEY})  # fmt: skip
        with self.assertRaises(LLMError) as cm:
            c.complete("x")
        self.assertNotIn(FAKE_KEY, str(cm.exception))


class TestCliBackend(unittest.TestCase):
    def test_commands(self):
        pf, of = Path("/tmp/p.txt"), Path("/tmp/o.txt")
        cmd, stdin = cli_command(LLMSettings("cli", "claude", "sonnet", 5), pf, of)
        self.assertEqual(cmd[:2], ["claude", "-p"])
        self.assertIn("--model", cmd)
        self.assertTrue(stdin)
        cmd, stdin = cli_command(LLMSettings("cli", "openai", "", 5), pf, of)
        self.assertEqual(cmd[:2], ["codex", "exec"])
        self.assertEqual(cmd[-1], "-")
        self.assertIn("--skip-git-repo-check", cmd)
        self.assertNotIn("-m", cmd)
        self.assertTrue(stdin)
        cmd, stdin = cli_command(LLMSettings("cli", "grok", "grok-test", 5), pf, of)
        self.assertEqual(cmd[0], "grok")
        self.assertIn("--prompt-file", cmd)
        self.assertIn("--disable-web-search", cmd)
        self.assertNotIn("--no-auto-update", cmd)
        self.assertFalse(stdin)

    def test_claude_gets_no_tools_and_only_read_for_images(self):
        pf, of = Path("/tmp/p.txt"), Path("/tmp/o.txt")
        cmd, _ = cli_command(LLMSettings("cli", "claude", "haiku", 5), pf, of)
        self.assertEqual(cmd, ["claude", "-p", "--output-format", "text", "--model", "haiku", "--tools", ""])
        cmd, _ = cli_command(LLMSettings("cli", "claude", "", 5), pf, of, ["/x/a.png"])
        self.assertEqual(cmd, ["claude", "-p", "--output-format", "text", "--tools", "Read", "--allowedTools", "Read",
                               "--add-dir", "/x"])  # fmt: skip

    def test_claude_stdin_and_output(self):
        r = FakeRunner(stdout="Answer [1]\n")
        c = LLMClient(LLMSettings("cli", "claude", "sonnet", 9), runner=r, which=have)
        self.assertEqual(c.complete("PROMPT", system="SYS"), "Answer [1]")
        cmd, kw = r.calls[0]
        self.assertIn("SYS", kw["input"])
        self.assertIn("PROMPT", kw["input"])
        self.assertEqual(kw["timeout"], 9)

    def test_codex_reads_last_message_file(self):
        r = FakeRunner(stdout="progress noise\nfinal", write_last="final only")
        c = LLMClient(LLMSettings("cli", "openai", "", 5), runner=r, which=have)
        self.assertEqual(c.complete("p"), "final only")

    def test_grok_prompt_file(self):
        seen = {}

        def runner(cmd, **kw):
            seen["prompt"] = Path(cmd[cmd.index("--prompt-file") + 1]).read_text()
            seen["input"] = kw["input"]
            return SimpleNamespace(returncode=0, stdout="g", stderr="")

        LLMClient(LLMSettings("cli", "grok", "", 5), runner=runner, which=have).complete("hello")
        self.assertEqual(seen["prompt"], "hello")
        self.assertIsNone(seen["input"])

    def test_not_signed_in(self):
        r = FakeRunner(returncode=1, stdout="", stderr="Error: Not logged in. Please run /login")
        c = LLMClient(LLMSettings("cli", "claude", "", 5), runner=r, which=have)
        with self.assertRaises(LLMUnavailable) as cm:
            c.complete("p")
        self.assertIn("sign in", str(cm.exception))

    def test_other_failure_and_timeout_and_missing(self):
        r = FakeRunner(returncode=3, stdout="", stderr="model overloaded")
        with self.assertRaises(LLMError):
            LLMClient(LLMSettings("cli", "grok", "", 5), runner=r, which=have).complete("p")
        r = FakeRunner(exc=subprocess.TimeoutExpired("claude", 5))
        with self.assertRaises(LLMError) as cm:
            LLMClient(LLMSettings("cli", "claude", "", 5), runner=r, which=have).complete("p")
        self.assertIn("timed out", str(cm.exception))
        r = FakeRunner(exc=FileNotFoundError())
        with self.assertRaises(LLMUnavailable):
            LLMClient(LLMSettings("cli", "claude", "", 5), runner=r, which=have).complete("p")
        with self.assertRaises(LLMUnavailable):
            LLMClient(LLMSettings("cli", "claude", "", 5), runner=r, which=missing).complete("p")

    def test_usage_limit_stops_with_the_cli_line(self):
        prompt = "page text about the API rate limit, credentials and login. " * 40
        err = f"OpenAI Codex v0.153.4\n--------\nuser\n{prompt}\nERROR: You've hit your usage limit. Try again at 9:35 PM.\n"
        r = FakeRunner(returncode=1, stdout="", stderr=err)
        with self.assertRaises(LLMUnavailable) as cm:
            LLMClient(LLMSettings("cli", "openai", "", 5), runner=r, which=have).complete(prompt)
        self.assertEqual(
            str(cm.exception), "`codex` hit a usage limit: You've hit your usage limit. Try again at 9:35 PM."
        )

    def test_usage_limit_wording_of_each_cli(self):
        cases = [("openai", "", "ERROR: You've hit your usage limit. Try again at 9:35 PM.\n"),
                 ("openai", "", "stream error: exceeded retry limit, last status: 429 Too Many Requests\n"),
                 ("claude", "Claude AI usage limit reached|1760000000\n", ""),
                 ("claude", "5-hour limit reached \u2219 resets 3pm\n", ""),
                 ("grok", "", "Error: 429 Rate limit exceeded\n")]  # fmt: skip
        for provider, out, err in cases:
            r = FakeRunner(returncode=1, stdout=out, stderr=err)
            with self.subTest(provider=provider, line=(out or err).strip()), self.assertRaises(LLMUnavailable):
                LLMClient(LLMSettings("cli", provider, "", 5), runner=r, which=have).complete("a question")

    def test_page_text_quoting_429_in_the_output_is_not_a_usage_limit(self):
        page = "Errors by day: HTTP 429 (rate limit) on 12 days"
        r = FakeRunner(returncode=1, stdout=f"The page says: {page}", stderr="error: stream disconnected")
        with self.assertRaises(LLMError) as cm:
            LLMClient(LLMSettings("cli", "claude", "", 5), runner=r, which=have).complete("a question")
        self.assertNotIsInstance(cm.exception, LLMUnavailable)
        r = FakeRunner(returncode=1, stdout="", stderr=f"user\n{page}\n[prompt cut]\nboom")  # a cut echo: not masked
        with self.assertRaises(LLMError) as cm:
            LLMClient(LLMSettings("cli", "openai", "", 5), runner=r, which=have).complete(page + "\nOne word.")
        self.assertNotIsInstance(cm.exception, LLMUnavailable)

    def test_an_echoed_prompt_is_not_read_as_the_error(self):
        prompt = ("page text about the API rate limit, credentials and login. " * 40).strip()
        r = FakeRunner(returncode=3, stdout="", stderr=f"banner\nuser\n{prompt}\nmodel overloaded")
        with self.assertRaises(LLMError) as cm:
            LLMClient(LLMSettings("cli", "openai", "", 5), runner=r, which=have).complete(prompt)
        self.assertEqual(str(cm.exception), "`codex` exited 3: banner\nuser\n<prompt>\nmodel overloaded")
        r = FakeRunner(returncode=3, stdout="", stderr="x" * 400 + "\nmodel overloaded")
        with self.assertRaises(LLMError) as cm:
            LLMClient(LLMSettings("cli", "grok", "", 5), runner=r, which=have).complete("p")
        self.assertEqual(str(cm.exception), "`grok` exited 3: " + "x" * 283 + "\nmodel overloaded")

    def test_empty_output(self):
        r = FakeRunner(stdout="   ")
        with self.assertRaises(LLMError):
            LLMClient(LLMSettings("cli", "claude", "", 5), runner=r, which=have).complete("p")


HITS = [
    {"doc_id": "vidore_v3_hr/rep", "pages": [12], "text": "Unemployment fell to 5.9% in 2024."},
    {"doc_id": "ocb/sales.xlsx", "pages": [], "text": "Q3 freight cost: 41,200"},
]


class TestAnswer(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.cfg = cfg_in(self.td.name)
        write_manifest(
            self.cfg.path("corpus_dir") / "manifest.csv",
            [Doc("vidore_v3_hr/rep", "pdf/vidore_v3_hr/rep.pdf", "pdf", "v", "CC-BY-4.0"),
             Doc("ocb/sales.xlsx", "office/ocb/sales.xlsx", "xlsx", "ocb", "x")],
        )  # fmt: skip

    def tearDown(self):
        self.td.cleanup()

    def test_prompt_and_citations(self):
        paths = {"vidore_v3_hr/rep": "pdf/vidore_v3_hr/rep.pdf"}
        p = build_prompt("q?", HITS, paths)
        self.assertIn("[1] (rep.pdf p.12)", p)
        self.assertIn("[2] (sales.xlsx)", p)
        cites = parse_citations("It fell [1]. Also [1] and [7].", HITS, paths)
        self.assertEqual([c["n"] for c in cites], [1])
        self.assertEqual(cites[0]["label"], "rep.pdf p.12")

    def test_answer_with_mocked_llm(self):
        comp, calls = fake_completion("Unemployment was 5.9% in 2024 [1].")
        client = LLMClient(settings(self.cfg), completion=comp, env={"ANTHROPIC_API_KEY": FAKE_KEY})
        res = answer(self.cfg, "What was unemployment in 2024?", client=client, retrieve=lambda q: HITS)
        self.assertEqual(res["citations"][0]["label"], "rep.pdf p.12")
        self.assertFalse(res["not_found"])
        self.assertIn("Passages:", calls[0]["messages"][1]["content"])

    def test_not_found(self):
        comp, _ = fake_completion(NOT_FOUND)
        client = LLMClient(settings(self.cfg), completion=comp, env={"ANTHROPIC_API_KEY": FAKE_KEY})
        res = answer(self.cfg, "Who won in 1950?", client=client, retrieve=lambda q: HITS)
        self.assertTrue(res["not_found"])
        self.assertEqual(res["citations"], [])

    def test_unavailable_skips_before_retrieval(self):
        called = []
        client = LLMClient(settings(self.cfg), env={})
        with self.assertRaises(LLMUnavailable):
            answer(self.cfg, "q", client=client, retrieve=lambda q: called.append(q) or HITS)
        self.assertEqual(called, [])


class TestDraft(unittest.TestCase):
    def test_extract_json(self):
        self.assertEqual(extract_json('sure: {"question": "a"} done'), {"question": "a"})
        self.assertIsNone(extract_json("no json"))

    def test_draft_multi_and_no_answer(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = cfg_in(td)
            write_manifest(
                cfg.path("corpus_dir") / "manifest.csv",
                [Doc("s/a", "pdf/a.pdf", "pdf", "s", "l", pages=3), Doc("s/b", "pdf/b.pdf", "pdf", "s", "l", pages=3)],
            )  # fmt: skip
            replies = iter([
                '{"question": "Compare A and B growth?", "answer": "A>B", "uses": ["A","B"]}',
                "UNKNOWN",
                '{"question": "What was 2031 output?", "why_unanswerable": "future year"}',
            ])  # fmt: skip

            def comp(**kw):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))])

            client = LLMClient(settings(cfg, "draft"), completion=comp, env={"ANTHROPIC_API_KEY": FAKE_KEY})
            out = draft(cfg, n_multi=1, n_none=1, client=client, excerpts=lambda c, d: [(2, "x" * 400)])
            rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        multi = rows[0]
        self.assertEqual(multi["meta"]["status"], "unreviewed")
        self.assertEqual(set(multi["gold"]), {"s/a#p2", "s/b#p2"})
        self.assertEqual(rows[1]["gold"], {})
        self.assertEqual(rows[1]["meta"]["type"], "no_answer")

    def test_closed_book_filter_drops_trivia(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = cfg_in(td)
            write_manifest(cfg.path("corpus_dir") / "manifest.csv",
                           [Doc("s/a", "a.pdf", "pdf", "s", "l"), Doc("s/b", "b.pdf", "pdf", "s", "l")])  # fmt: skip
            replies = iter(['{"question": "What is the capital of France?"}', "Paris."])

            def comp(**kw):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))])

            client = LLMClient(settings(cfg, "draft"), completion=comp, env={"ANTHROPIC_API_KEY": FAKE_KEY})
            out = draft(cfg, n_multi=1, n_none=0, client=client, excerpts=lambda c, d: [(0, "y" * 400)])
            self.assertEqual(out.read_text(), "")


class TestCliSkip(unittest.TestCase):
    def test_answer_cli_skips_cleanly_without_backend(self):
        import contextlib
        import io

        from docsearch.cli import main

        with tempfile.TemporaryDirectory() as td:
            cfg_in(td)
            err = io.StringIO()
            env_backup = {}
            import os

            for k in ("ANTHROPIC_API_KEY", "DOCSEARCH_LLM_BACKEND", "DOCSEARCH_LLM_PROVIDER"):
                env_backup[k] = os.environ.pop(k, None)
            try:
                with contextlib.redirect_stderr(err):
                    rc = main(["--config", str(Path(td) / "docsearch.yaml"), "answer", "what?"])
            finally:
                for k, v in env_backup.items():
                    if v is not None:
                        os.environ[k] = v
        self.assertEqual(rc, 2)
        self.assertIn("ANTHROPIC_API_KEY is not set", err.getvalue())
        self.assertIn("Search and eval do not need an LLM", err.getvalue())


if __name__ == "__main__":
    unittest.main()
