import type { Filters, PageText, SearchResponse, ServerConfig, StreamEvent } from "../types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export function searchUrl(query: string, filters: Filters, agent: boolean, k = 10, path = "/api/search"): string {
  const p = new URLSearchParams({ q: query, k: String(k) });
  for (const f of filters.formats) p.append("formats", f);
  if (filters.after) p.set("after", filters.after);
  if (filters.before) p.set("before", filters.before);
  if (filters.folder.trim()) p.set("folder", filters.folder.trim());
  if (filters.undated && (filters.after || filters.before)) p.set("undated", "true");
  if (agent) p.set("agent", "true");
  return `${path}?${p}`;
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(url, { signal });
  const body = await r.json().catch(() => null);
  if (!r.ok) throw new ApiError(body?.detail ?? `the server answered ${r.status}`, r.status);
  return body as T;
}

export const fetchConfig = (signal?: AbortSignal) => getJson<ServerConfig>("/api/config", signal);

export const fetchPageText = (file: string, page: number | null, signal?: AbortSignal) =>
  getJson<PageText>(
    `/api/text?${new URLSearchParams({ file, ...(page === null ? {} : { page: String(page) }) })}`,
    signal,
  );

// Passages the page asks for: the reranked 30 (search.rerank_candidates), so more files show than the CLI's 10.
export const PAGE_K = 30;

// This tab's id: the server lets a new search stop only this tab's older search, never another tab's.
const TAB = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;

/** Calls onLine with each whole line of JSON in a stream, however the network splits the lines. */
export async function readLines(body: ReadableStream<Uint8Array>, onLine: (value: unknown) => void): Promise<void> {
  const reader = body.getReader();
  const text = new TextDecoder();
  let rest = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    const lines = (rest + text.decode(value, { stream: true })).split("\n");
    rest = lines.pop() ?? "";
    for (const line of lines) if (line.trim()) onLine(JSON.parse(line));
  }
  if (rest.trim()) onLine(JSON.parse(rest));
}

/** The search as it runs (GET /api/search/stream): early results and slide progress go to onEvent; the final
 * answer is returned. A 409 means a newer search from this tab replaced it. */
export async function streamSearch(
  query: string,
  filters: Filters,
  agent: boolean,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const url = searchUrl(query, filters, agent, PAGE_K, "/api/search/stream");
  const r = await fetch(url, { signal, headers: { "X-Docsearch-Tab": TAB } });
  if (!r.ok || !r.body) {
    const body = await r.json().catch(() => null);
    throw new ApiError(body?.detail ?? `the server answered ${r.status}`, r.status);
  }
  let final: SearchResponse | null = null;
  let failed: ApiError | null = null;
  await readLines(r.body, (v) => {
    const e = v as StreamEvent;
    if (e.event === "done") final = { results: e.results, notes: e.notes, terms: e.terms };
    else if (e.event === "error") failed = new ApiError(e.detail, e.status);
    else onEvent(e);
  });
  if (failed) throw failed;
  if (!final) throw new ApiError("the search stopped before it ended", 0);
  return final;
}

// Formats the server shows as a PDF (docx and pptx through their PDF copy), where #page=N opens that page.
const PDF_VIEW = new Set(["pdf", "docx", "pptx"]);

/** Where the browser shows a result's file: GET /api/file, at the page when there is one. */
export function fileUrl(file: string, format: string, page: number | null): string {
  const p = new URLSearchParams({ file });
  if (page !== null) p.set("page", String(page));
  return `/api/file?${p}${page !== null && PDF_VIEW.has(format) ? `#page=${page}` : ""}`;
}

// Formats docsearch render draws no page picture for.
const NO_PICTURE = new Set(["xlsx", "csv"]);

/** The picture of a result's page (GET /api/page); null for formats that have none. */
export function pageUrl(file: string, format: string, page: number | null): string | null {
  if (NO_PICTURE.has(format)) return null;
  return `/api/page?${new URLSearchParams({ file, page: String(page ?? 1) })}`;
}
