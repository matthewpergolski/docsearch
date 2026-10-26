# Security

## Report a vulnerability

Do not open a public issue for a security problem. Use the private vulnerability report of this repository
(GitHub: the Security tab, then "Report a vulnerability"). Give the version or commit, the steps to reproduce,
and what an attacker can do. We answer when we can; this is a volunteer project with no response-time promise.

Only the latest commit on the default branch gets fixes.

## What docsearch trusts

Read these before you run docsearch on a shared machine or a server.

- **The search page has no login.** `docsearch serve` listens on 127.0.0.1 only. Do not expose it to a network.
  `/api/file` and `/api/page` serve only the files in `corpus/manifest.csv` and the pages in
  `work/pages/pages.jsonl`, but anyone who can reach the port can search and read those files.
- **Documents are untrusted input.** Docling, LibreOffice (for `.doc` and Office renders) and the OCR models parse
  every file in the folder you sync. Sync only folders whose files you would open yourself.
- **Remote model code.** The `pplx-context` challenger embedder runs the model's own code from Hugging Face
  (`trust_remote_code`). Its revision is pinned in `docsearch.yaml`. The default pipeline does not use it.
- **LLM backends are optional.** With `llm.backend: cli`, docsearch runs the vendor CLI (`claude`, `codex`,
  `grok`) as a subprocess with a prompt file. API keys come only from environment variables; docsearch never
  logs or prints them. Do not put keys in `docsearch.yaml` or in the repo.
- **Model servers.** With `models.backend: api`, query text and passages go to the server in
  `models.embed_api_url` and `models.rerank_api_url`. Use a server you trust.
