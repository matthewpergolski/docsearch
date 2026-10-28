#!/usr/bin/env python3
"""Fail when a commit range is not under the neutral docsearch identity.

Allowed identity (author, committer, and Co-authored-by / Signed-off-by):

  docsearch contributors <docsearch@users.noreply.github.com>

Also fails when a commit's added lines introduce a personal-looking email
address other than that neutral address. Generic placeholders such as
git@github.com and *@example.* are allowed in file text. Paths under
tests/ are skipped for the file-text scan so fixtures may use synthetic
addresses; author, committer, and trailers are still checked on every commit.

Usage:
  python3 scripts/check_identity.py [BASE..HEAD]
  python3 scripts/check_identity.py origin/main..HEAD

Exit 0 when clean, 1 when problems are found, 2 on usage/git errors.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ALLOWED_NAME = "docsearch contributors"
ALLOWED_EMAIL = "docsearch@users.noreply.github.com"
ALLOWED = f"{ALLOWED_NAME} <{ALLOWED_EMAIL}>"

# Placeholders that docs and tests may show. Not a personal inbox.
_GENERIC_EMAIL = re.compile(
    r"^(?:"
    r"git@github\.com|"
    r"[A-Za-z0-9._%+\-]+@example\.(?:com|org|net|invalid)"
    r")$",
    re.I,
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_TRAILER = re.compile(
    r"^(Co-authored-by|Signed-off-by):\s*(.+?)\s*$",
    re.I | re.M,
)
_TRAILER_VALUE = re.compile(r"^(.+?)\s*<([^>]+)>\s*$")


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def identity_ok(name: str, email: str) -> bool:
    return name == ALLOWED_NAME and email.lower() == ALLOWED_EMAIL.lower()


def email_ok_in_text(email: str) -> bool:
    """True when an address may appear in added file lines."""
    low = email.lower()
    if low == ALLOWED_EMAIL.lower():
        return True
    return bool(_GENERIC_EMAIL.match(email))


def _parse_trailer_value(value: str) -> tuple[str, str] | None:
    match = _TRAILER_VALUE.match(value.strip())
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def check_commit(
    sha: str,
    author: str,
    email: str,
    committer: str,
    cemail: str,
    body: str,
) -> list[str]:
    """Return problem lines for one commit's identity and trailers."""
    errors: list[str] = []
    short = sha[:7]
    if not identity_ok(author, email):
        errors.append(f"{short}: author {author} <{email}> (want {ALLOWED})")
    if not identity_ok(committer, cemail):
        errors.append(f"{short}: committer {committer} <{cemail}> (want {ALLOWED})")
    for match in _TRAILER.finditer(body):
        kind, value = match.group(1), match.group(2)
        parsed = _parse_trailer_value(value)
        if parsed is None:
            errors.append(f"{short}: {kind} has no Name <email>: {value!r}")
            continue
        name, addr = parsed
        if not identity_ok(name, addr):
            errors.append(f"{short}: {kind} {name} <{addr}> (want {ALLOWED})")
    return errors


def check_added_text(sha: str, path: str, text: str) -> list[str]:
    """Return problems for personal-looking emails in added lines."""
    errors: list[str] = []
    short = sha[:7]
    for line in text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for addr in _EMAIL.findall(line):
            if not email_ok_in_text(addr):
                errors.append(f"{short}: added {path} has {addr} (only {ALLOWED_EMAIL} or generic examples)")
    return errors


def _skip_file_email_scan(path: str) -> bool:
    """True when added-line email rules do not apply (test fixtures)."""
    norm = path.replace("\\", "/").lstrip("./")
    return norm == "tests" or norm.startswith("tests/")


def check_range(repo: Path | str, rev_range: str) -> list[str]:
    """Check every commit in rev_range. Return problem lines (empty when clean)."""
    root = Path(repo)
    listed = _run(root, "rev-list", rev_range)
    if listed.returncode != 0:
        detail = listed.stderr.strip() or listed.stdout.strip()
        return [f"git rev-list failed for {rev_range!r}: {detail}"]
    shas = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    errors: list[str] = []
    for sha in shas:
        meta = _run(
            root,
            "show",
            "-s",
            "--format=%an%n%ae%n%cn%n%ce%n%B",
            sha,
        )
        if meta.returncode != 0:
            errors.append(f"{sha[:7]}: git show failed")
            continue
        parts = meta.stdout.split("\n", 4)
        if len(parts) < 5:
            errors.append(f"{sha[:7]}: could not parse commit metadata")
            continue
        author, email, committer, cemail, body = parts
        errors += check_commit(sha, author, email, committer, cemail, body)
        diff = _run(root, "show", "--format=", "--unified=0", sha)
        if diff.returncode != 0:
            errors.append(f"{sha[:7]}: git show diff failed")
            continue
        path = ""
        chunk: list[str] = []
        for line in diff.stdout.splitlines():
            if line.startswith("diff --git "):
                if path and chunk and not _skip_file_email_scan(path):
                    errors += check_added_text(sha, path, "\n".join(chunk))
                chunk = []
                # diff --git a/path b/path
                bits = line.split(" b/", 1)
                path = bits[1] if len(bits) == 2 else line
            elif line.startswith("+++ "):
                continue
            else:
                chunk.append(line)
        if path and chunk and not _skip_file_email_scan(path):
            errors += check_added_text(sha, path, "\n".join(chunk))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "range",
        nargs="?",
        default="origin/main..HEAD",
        help="git revision range (default: origin/main..HEAD)",
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="git repository path (default: .)",
    )
    args = parser.parse_args(argv)
    problems = check_range(Path(args.repo), args.range)
    if not problems:
        print(f"identity: ok ({args.range})")
        return 0
    print(f"identity: {len(problems)} problem(s) in {args.range}", file=sys.stderr)
    for line in problems:
        print(f"error: {line}", file=sys.stderr)
    print(f"Fix: use {ALLOWED} for author, committer, and trailers.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
