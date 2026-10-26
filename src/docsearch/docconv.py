"""Old Word files (.doc): convert to .docx with an external program, so the Docling docx parser can read them.

No Python dependency. The converter is LibreOffice (`soffice`, MPL-2.0, installed by the user) or macOS
`textutil`, found on PATH. ingest.doc_converter picks one: auto (soffice, then textutil), soffice, textutil, or
off. The .docx is cached in paths.cache_dir/doc_docx, keyed by the source sha256 and the converter name. The
document keeps its own doc_id, path and format ("docx": the parser it goes through, as .xlsm is "xlsx").
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_S = 120
OLE_MAGIC = bytes.fromhex("d0cf11e0a1b11ae1")  # the compound file that holds a Word 97-2003 document
ZIP_MAGIC = b"PK\x03\x04"  # a .docx (OOXML) that an exporter named .doc: the parser reads it as it is
_PROFILES: set[Path] = set()


def profile_dir(cache: Path) -> Path:
    """The LibreOffice profile this process uses, removed at exit. Its own profile: with the user's profile, a
    running LibreOffice window takes the job and writes nothing. One per process, not per file: a new profile
    costs LibreOffice's first start each time (CR2)."""
    d = Path(cache).resolve() / f"soffice-profile-{os.getpid()}"
    if d not in _PROFILES:
        _PROFILES.add(d)
        atexit.register(shutil.rmtree, d, True)
    return d


def soffice_command(exe: str, cache: Path, *args: str) -> list[str]:
    """argv for a headless LibreOffice run with this process's profile; parse (.doc) and render share it."""
    return [exe, "--headless", "--norestore", f"-env:UserInstallation={profile_dir(cache).as_uri()}", *args]


def find_soffice(which: Callable[[str], str | None] | None = None) -> str | None:
    which = which or shutil.which
    return next((exe for exe in map(which, CONVERTERS["soffice"][0]) if exe), None)


def _soffice(exe: str, src: Path, out: Path, cache: Path) -> list[str]:
    return soffice_command(exe, cache, "--convert-to", "docx", "--outdir", str(out.parent), str(src))


def _textutil(exe: str, src: Path, out: Path, cache: Path) -> list[str]:
    return [exe, "-convert", "docx", "-output", str(out), str(src)]


# name -> (program names to look for on PATH, command that writes <out folder>/<source stem>.docx); auto order
CONVERTERS: dict[str, tuple[tuple[str, ...], Callable[[str, Path, Path, Path], list[str]]]] = {
    "soffice": (("soffice", "libreoffice"), _soffice),
    "textutil": (("textutil",), _textutil),
}


class DocConvertError(RuntimeError):
    pass


def is_word_file(head: bytes) -> bool:
    """A Word 97-2003 file, or the RTF or HTML (UTF-8 or UTF-16) that Word and many programs also save as .doc.
    Both converters import any other bytes as plain text without an error, which would index noise (DOC1)."""
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        head = head.decode("utf-16", "ignore").encode()
    text = head.removeprefix(b"\xef\xbb\xbf").lstrip()
    return head.startswith(OLE_MAGIC) or text.startswith((b"{\\rtf", b"<"))


def for_want_of_converter(error: str) -> bool:
    """The parse error a .doc gets when no converter could run (pick's reasons): a converter found later retries it."""
    return "install LibreOffice or set ingest.doc_converter" in error or "ingest.doc_converter is off" in error


def converted_with(cache: Path, sha: str) -> set[str]:
    """The converters whose .docx of this source version is in the cache."""
    return {p.name.removeprefix(f"{sha}-").removesuffix(".docx") for p in Path(cache).glob(f"{sha}-*.docx")}


def drop_cached(cache: Path, sha: str) -> None:
    """Remove the cached .docx files of a source version that is gone (CR2: one copy per version stayed)."""
    for p in Path(cache).glob(f"{sha}-*.docx"):
        p.unlink(missing_ok=True)


@dataclass(frozen=True)
class Converter:
    name: str
    exe: str


def pick(setting: str, which: Callable[[str], str | None] | None = None) -> tuple[Converter | None, str]:
    """The converter ingest.doc_converter names and PATH has, or None and the reason a .doc is skipped."""
    which = which or shutil.which
    if setting == "off":
        return None, "ingest.doc_converter is off"
    for name in CONVERTERS if setting == "auto" else (setting,):
        for program in CONVERTERS[name][0]:
            exe = which(program)
            if exe:
                return Converter(name, exe), ""
    what = "no converter" if setting == "auto" else f"{setting} not found"
    return None, f"{what} (install LibreOffice or set ingest.doc_converter)"


def to_docx(src: Path, sha: str, cache: Path, conv: Converter, run=None, timeout: float = TIMEOUT_S) -> Path:
    """The cached .docx of `src`; converts it first when the cache has none. Raises DocConvertError."""
    dst = cache / f"{sha}-{conv.name}.docx"
    if dst.exists():
        return dst
    with src.open("rb") as f:
        head = f.read(512)
    cache.mkdir(parents=True, exist_ok=True)
    if head.startswith(ZIP_MAGIC):
        shutil.copyfile(src, dst)
        return dst
    if not is_word_file(head):
        raise DocConvertError(f"{src.name} is not a Word file (no Word 97-2003, RTF or HTML header)")
    run = run or subprocess.run
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        out = Path(tmp) / f"{src.stem}.docx"
        try:
            cmd = CONVERTERS[conv.name][1](conv.exe, src, out, cache)
            r = run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise DocConvertError(f"{conv.name} timed out after {timeout:g} s on {src.name}") from None
        if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            tail = (r.stderr or r.stdout or "").strip().splitlines()
            why = tail[-1] if r.returncode != 0 and tail else "no .docx written"
            raise DocConvertError(f"{conv.name} could not convert {src.name} (exit {r.returncode}): {why}")
        os.replace(out, dst)
    return dst


def docx_for(cfg, src: Path, sha: str) -> Path:
    """The .docx the parser reads for the .doc `src`, under the converter ingest.doc_converter picks."""
    conv, why = pick(cfg["ingest"]["doc_converter"])
    if conv is None:
        raise DocConvertError(f"{src.name}: {why}")
    return to_docx(src, sha, cfg.path("cache_dir") / "doc_docx", conv)
