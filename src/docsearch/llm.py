"""LLM access with two backends, chosen by config (llm.backend / llm.provider) or env.

backend=api : LiteLLM -> Claude (Anthropic), OpenAI, Grok (xAI), or local Ollama. Needs API keys
              (ANTHROPIC_API_KEY / OPENAI_API_KEY / XAI_API_KEY); Ollama needs a running server.
backend=cli : subscription CLIs on your machine, called by subprocess:
              claude -> `claude -p` (Anthropic CLI), openai -> `codex exec`, grok -> `grok --prompt-file` (xAI grok CLI).

Search and plain eval never import this module; eval --mode agent does (agent.py). Key values are never printed or logged; only the
names of missing env vars are reported.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import Config

API_KEY_ENV = {"claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "grok": "XAI_API_KEY", "ollama": None}
CLI_BIN = {"claude": "claude", "openai": "codex", "grok": "grok"}
CLI_LOGIN_HINT = {
    "claude": "run `claude` once and sign in (or `claude /login`)",
    "openai": "run `codex login`",
    "grok": "run `grok login` (or `grok login --device-auth`)",
}
PROVIDERS = ("claude", "openai", "grok", "ollama")
BACKENDS = ("api", "cli")
_AUTH_PAT = re.compile(
    r"not (logged|signed) in|log ?in|sign ?in|authenticat|unauthori[sz]ed|\b401\b|api key|credential", re.I
)  # fmt: skip
# The CLIs' own usage-limit lines: codex "You've hit your usage limit" / "429 Too Many Requests", claude "Claude AI
# usage limit reached" / "5-hour limit reached", grok "Rate limit exceeded". A bare "429" or "rate limit" can be page
# text, so it does not count.
_LIMIT_PAT = re.compile(r"usage limit|limit reached|hit your limit|rate limit exceeded|too many requests", re.I)


_log = logging.getLogger("docsearch.llm")


class LLMUnavailable(RuntimeError):
    """Backend cannot be used here (missing key, binary, server, or sign-in). Message is user-facing."""


class LLMError(RuntimeError):
    """The backend was reachable but the call failed."""


class LLMTimeout(LLMError):
    """One call ran past llm.timeout_s (the backend itself may be fine)."""


@dataclass
class LLMSettings:
    backend: str
    provider: str
    model: str
    timeout_s: float
    api_base: str | None = None


def settings(
    cfg: Config, role: str = "answer", backend: str | None = None, provider: str | None = None, model: str | None = None
) -> LLMSettings:
    """Resolve settings. Precedence: explicit args > DOCSEARCH_LLM_* env > docsearch.yaml."""
    c = cfg["llm"]
    backend = (backend or c["backend"]).lower()
    provider = (provider or c["provider"]).lower()
    if backend not in BACKENDS:
        raise ValueError(f"llm.backend must be one of {BACKENDS}, got {backend!r}")
    if provider not in PROVIDERS:
        raise ValueError(f"llm.provider must be one of {PROVIDERS}, got {provider!r}")
    if backend == "cli" and provider == "ollama":
        raise ValueError("ollama is only available with backend=api (LiteLLM talks to the local server)")
    table = c.get("api_models" if backend == "api" else "cli_models", {}) or {}
    model = model if model is not None else (table.get(provider, {}) or {}).get(role, "")
    return LLMSettings(
        backend=backend,
        provider=provider,
        model=model or "",
        timeout_s=float((c.get("timeout_s_by_provider") or {}).get(provider, c.get("timeout_s", 180))),
        api_base=c.get("ollama_api_base") if provider == "ollama" else None,
    )


def check_available(s: LLMSettings, env: dict[str, str] | None = None, which: Callable = shutil.which) -> None:
    """Raise LLMUnavailable with a clear, key-free message if this backend cannot run."""
    env = dict(os.environ if env is None else env)
    if s.backend == "api":
        var = API_KEY_ENV[s.provider]
        if var and not env.get(var):
            raise LLMUnavailable(
                f"backend=api provider={s.provider}: {var} is not set. Export it in your shell "
                f"(or switch with --backend cli / --provider ...)."
            )
        if not s.model:
            raise LLMUnavailable(f"no model configured for llm.api_models.{s.provider}")
    else:
        binary = CLI_BIN[s.provider]
        if which(binary) is None:
            raise LLMUnavailable(
                f"backend=cli provider={s.provider}: `{binary}` is not on PATH. Install it, then "
                f"{CLI_LOGIN_HINT[s.provider]}."
            )


# Image input per CLI (only what each CLI's --help documents):
#   claude: no image flag; with `--allowedTools Read --add-dir DIR` it can open a PNG path named in the prompt.
#   codex exec: `-i/--image FILE`.
#   grok (1.0.50): headless `--prompt-file`; `--prompt-json` takes content blocks but the help does not
#   document an image block, so grok is text-only here.
CLI_IMAGES = {"claude": True, "openai": True, "grok": False}


def cli_command(
    s: LLMSettings, prompt_file: Path, out_file: Path, images: list[str] | None = None
) -> tuple[list[str], bool]:  # noqa: E501
    """argv for the subscription CLI, and whether the prompt goes to stdin."""
    images = images or []
    if s.provider == "claude":
        cmd = ["claude", "-p", "--output-format", "text"]
        if s.model:
            cmd += ["--model", s.model]
        cmd += ["--tools", "Read" if images else ""]
        if images:
            cmd += ["--allowedTools", "Read"]
            for d in sorted({str(Path(i).parent) for i in images}):
                cmd += ["--add-dir", d]
        return cmd, True
    if s.provider == "openai":
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--ephemeral",
               "--color", "never", "-o", str(out_file)]  # fmt: skip
        if s.model:
            cmd += ["-m", s.model]
        for i in images:
            cmd += ["-i", i]
        return cmd + ["-"], True
    if s.provider == "grok":
        # headless single turn (grok --help: --prompt-file, --output-format plain, -m); no web search so the
        # closed-book filter stays honest, no subagents.
        cmd = ["grok", "--prompt-file", str(prompt_file), "--output-format", "plain", "--disable-web-search",
               "--no-subagents"]  # fmt: skip
        if s.model:
            cmd += ["-m", s.model]
        return cmd, False
    raise ValueError(f"no CLI for provider {s.provider}")


def _content(prompt: str, images: list[str] | None):
    if not images:
        return prompt
    import base64

    parts: list[dict] = [{"type": "text", "text": prompt}]
    for i in images:
        b64 = base64.b64encode(Path(i).read_bytes()).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return parts


class LLMClient:
    def __init__(
        self,
        s: LLMSettings,
        runner: Callable = subprocess.run,
        completion: Callable | None = None,
        env: dict[str, str] | None = None,
        which: Callable = shutil.which,
    ):
        self.s = s
        self.runner = runner
        self._completion = completion
        self.env = env
        self.which = which

    def supports_images(self) -> bool:
        return self.s.backend == "api" or CLI_IMAGES.get(self.s.provider, False)

    def complete(self, prompt: str, system: str | None = None, images: list[str] | None = None) -> str:
        """images: PNG paths; dropped silently when the backend cannot take them (see supports_images)."""
        check_available(self.s, self.env, self.which)
        images = images if images and self.supports_images() else None
        if self.s.backend == "api":
            return self._api(prompt, system, images)
        return self._cli(prompt, system, images)

    # -- api (LiteLLM)
    def _api(self, prompt: str, system: str | None, images: list[str] | None = None) -> str:
        completion = self._completion
        if completion is None:
            import litellm

            completion = litellm.completion
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": _content(prompt, images)}
        ]
        kw = {"model": self.s.model, "messages": msgs, "timeout": self.s.timeout_s}
        if self.s.api_base:
            kw["api_base"] = self.s.api_base
        try:
            resp = completion(**kw)
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__
            if "Authentication" in name or "Unauthorized" in name:
                raise LLMUnavailable(f"{self.s.provider} rejected the API key ({name}).") from None
            if self.s.provider == "ollama" and ("Connection" in name or "connect" in str(e).lower()):
                raise LLMUnavailable(
                    f"Ollama server not reachable at {self.s.api_base} (run `ollama serve`)."
                ) from None
            raise LLMError(f"{self.s.provider} API call failed: {name}") from None
        text = resp.choices[0].message.content
        return (text or "").strip()

    # -- cli (subscriptions)
    def _cli(self, prompt: str, system: str | None, images: list[str] | None = None) -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        if images and self.s.provider == "claude":
            full += "\n\nPage image(s) to look at with the Read tool:\n" + "\n".join(images)
        with tempfile.TemporaryDirectory() as td:
            pf, of = Path(td) / "prompt.txt", Path(td) / "last_message.txt"
            pf.write_text(full)
            cmd, use_stdin = cli_command(self.s, pf, of, images)
            try:
                proc = self.runner(
                    cmd,
                    input=full if use_stdin else None,
                    capture_output=True,
                    text=True,
                    timeout=self.s.timeout_s,
                    cwd=td,
                )
            except FileNotFoundError:
                raise LLMUnavailable(f"`{cmd[0]}` is not on PATH.") from None
            except subprocess.TimeoutExpired:
                raise LLMTimeout(f"`{cmd[0]}` timed out after {self.s.timeout_s:.0f}s (llm.timeout_s).") from None
            out = proc.stdout or ""
            if self.s.provider == "openai" and of.exists():
                out = of.read_text()
        if proc.returncode != 0:
            err = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
            err = err.replace(full.strip(), "<prompt>")  # codex echoes the prompt; its words are not the error
            # The CLI's own lines: stderr, and the last line of stdout (a failed claude -p prints its error there);
            # earlier stdout can be a partial answer that quotes the page.
            own = (proc.stderr or "").replace(full.strip(), "<prompt>").splitlines()
            own += (proc.stdout or "").strip().splitlines()[-1:]
            limit = [ln for ln in own if _LIMIT_PAT.search(ln)]
            if limit:
                raise LLMUnavailable(f"`{cmd[0]}` hit a usage limit: {limit[-1].strip().removeprefix('ERROR: ')}")
            if _AUTH_PAT.search(err):
                raise LLMUnavailable(f"`{cmd[0]}` is not signed in: {CLI_LOGIN_HINT[self.s.provider]}.")
            raise LLMError(f"`{cmd[0]}` exited {proc.returncode}: {err[-300:]}")
        if (proc.stderr or "").strip():  # CLI warnings go to the run log, never the screen
            _log.warning("%s stderr: %s", cmd[0], proc.stderr.strip()[:2000])
        text = out.strip()
        if not text:
            raise LLMError(f"`{cmd[0]}` returned no output.")
        return text


def get_client(cfg: Config, role: str = "answer", **overrides) -> LLMClient:
    return LLMClient(settings(cfg, role, **overrides))


def status(cfg: Config) -> list[tuple[str, str, str]]:
    """(backend, provider, 'ready' | reason) for every combination. Never includes key values."""
    out = []
    for b in BACKENDS:
        for p in PROVIDERS:
            if b == "cli" and p == "ollama":
                continue
            try:
                check_available(settings(cfg, "answer", backend=b, provider=p))
                out.append((b, p, "ready"))
            except (LLMUnavailable, ValueError) as e:
                out.append((b, p, str(e)))
    return out
