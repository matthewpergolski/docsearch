"""scripts/check_identity.py: only the neutral docsearch identity may land."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_identity.py"

NEUTRAL_NAME = "docsearch contributors"
NEUTRAL_EMAIL = "docsearch@users.noreply.github.com"


def load_mod():
    spec = importlib.util.spec_from_file_location("check_identity", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def git(cwd: Path, *args: str, env: dict | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout


def init_repo(path: Path, *, name: str = NEUTRAL_NAME, email: str = NEUTRAL_EMAIL) -> Path:
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.name", name)
    git(path, "config", "user.email", email)
    # Avoid agent/global hooks rewriting identity in fixture repos.
    empty_hooks = path / ".empty-hooks"
    empty_hooks.mkdir()
    git(path, "config", "core.hooksPath", str(empty_hooks))
    (path / "README.md").write_text("start\n")
    git(path, "add", "README.md")
    git(path, "commit", "-q", "-m", "start")
    return path


def _env_with(**overrides: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()}
    env.update(overrides)
    return env


class TestCheckIdentity(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_mod()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.base = Path(self._tmpdir.name)

    def test_neutral_range_is_clean(self) -> None:
        repo = init_repo(self.base / "ok")
        (repo / "a.txt").write_text("ok\n")
        git(repo, "add", "a.txt")
        git(repo, "commit", "-q", "-m", "add a")
        self.assertEqual(self.mod.check_range(repo, "HEAD~1..HEAD"), [])

    def test_foreign_author_email_is_reported(self) -> None:
        repo = init_repo(self.base / "bad-author")
        (repo / "a.txt").write_text("x\n")
        git(repo, "add", "a.txt")
        env = _env_with(
            GIT_AUTHOR_NAME=NEUTRAL_NAME,
            GIT_AUTHOR_EMAIL="stranger@example.net",
            GIT_COMMITTER_NAME=NEUTRAL_NAME,
            GIT_COMMITTER_EMAIL=NEUTRAL_EMAIL,
        )
        git(repo, "commit", "-q", "-m", "bad author", env=env)
        errors = self.mod.check_range(repo, "HEAD~1..HEAD")
        self.assertTrue(errors)
        self.assertTrue(any("author" in e.lower() and "stranger@example.net" in e for e in errors))

    def test_foreign_committer_email_is_reported(self) -> None:
        repo = init_repo(self.base / "bad-committer")
        (repo / "a.txt").write_text("x\n")
        git(repo, "add", "a.txt")
        env = _env_with(
            GIT_AUTHOR_NAME=NEUTRAL_NAME,
            GIT_AUTHOR_EMAIL=NEUTRAL_EMAIL,
            GIT_COMMITTER_NAME=NEUTRAL_NAME,
            GIT_COMMITTER_EMAIL="other@example.net",
        )
        git(repo, "commit", "-q", "-m", "bad committer", env=env)
        errors = self.mod.check_range(repo, "HEAD~1..HEAD")
        self.assertTrue(any("committer" in e.lower() and "other@example.net" in e for e in errors))

    def test_foreign_co_authored_by_trailer_is_reported(self) -> None:
        repo = init_repo(self.base / "trailer")
        (repo / "a.txt").write_text("x\n")
        git(repo, "add", "a.txt")
        msg = "feat\n\nCo-authored-by: Someone Else <else@example.net>\n"
        git(repo, "commit", "-q", "-m", msg)
        errors = self.mod.check_range(repo, "HEAD~1..HEAD")
        self.assertTrue(any("co-authored-by" in e.lower() and "else@example.net" in e for e in errors))

    def test_foreign_signed_off_by_trailer_is_reported(self) -> None:
        repo = init_repo(self.base / "sob")
        (repo / "a.txt").write_text("x\n")
        git(repo, "add", "a.txt")
        msg = "feat\n\nSigned-off-by: Someone Else <else@example.net>\n"
        git(repo, "commit", "-q", "-m", msg)
        errors = self.mod.check_range(repo, "HEAD~1..HEAD")
        self.assertTrue(any("signed-off-by" in e.lower() and "else@example.net" in e for e in errors))

    def test_added_personal_email_in_file_is_reported(self) -> None:
        # Build the address at runtime so this test file itself stays clean under
        # the identity check (no personal-looking literal in added lines).
        addr = "person@" + "somewhere.org"
        repo = init_repo(self.base / "file-email")
        (repo / "note.md").write_text(f"contact me at {addr} please\n")
        git(repo, "add", "note.md")
        git(repo, "commit", "-q", "-m", "add note")
        errors = self.mod.check_range(repo, "HEAD~1..HEAD")
        self.assertTrue(any(addr in e for e in errors))

    def test_generic_example_emails_in_files_are_allowed(self) -> None:
        repo = init_repo(self.base / "generic")
        (repo / "note.md").write_text(
            f"neutral {NEUTRAL_EMAIL}\nclone git@github.com:org/repo.git\ntests use test@example.invalid\n"
        )
        git(repo, "add", "note.md")
        git(repo, "commit", "-q", "-m", "generic examples")
        self.assertEqual(self.mod.check_range(repo, "HEAD~1..HEAD"), [])

    def test_personal_email_under_tests_path_is_allowed(self) -> None:
        addr = "person@" + "somewhere.org"
        repo = init_repo(self.base / "tests-dir")
        (repo / "tests").mkdir()
        (repo / "tests" / "fixture.txt").write_text(f"{addr}\n")
        git(repo, "add", "tests/fixture.txt")
        git(repo, "commit", "-q", "-m", "fixture under tests")
        self.assertEqual(self.mod.check_range(repo, "HEAD~1..HEAD"), [])

    def test_cli_exits_1_on_bad_range(self) -> None:
        repo = init_repo(self.base / "cli")
        (repo / "a.txt").write_text("x\n")
        git(repo, "add", "a.txt")
        env = _env_with(
            GIT_AUTHOR_NAME="Nope",
            GIT_AUTHOR_EMAIL="nope@example.net",
            GIT_COMMITTER_NAME=NEUTRAL_NAME,
            GIT_COMMITTER_EMAIL=NEUTRAL_EMAIL,
        )
        git(repo, "commit", "-q", "-m", "bad", env=env)
        code = subprocess.run(
            ["python3", str(SCRIPT), "HEAD~1..HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
        ).returncode
        self.assertEqual(code, 1)

    def test_cli_exits_0_on_neutral_range(self) -> None:
        repo = init_repo(self.base / "cli-ok")
        (repo / "a.txt").write_text("ok\n")
        git(repo, "add", "a.txt")
        git(repo, "commit", "-q", "-m", "ok")
        code = subprocess.run(
            ["python3", str(SCRIPT), "HEAD~1..HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
        ).returncode
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
