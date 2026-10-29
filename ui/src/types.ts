// The JSON of GET /api/search: `docsearch search --json` (display.as_json) plus notes and the query's terms.
export type Scorer = "rerank" | "page" | "fused";

export interface PageHit {
  page: number | null;
  rank: number;
  title: string;
  snippet: string;
  score: number | null;
  scorer: Scorer;
}

export interface ResultGroup {
  rank: number;
  file: string;
  format: string;
  name: string;
  date: string;
  period: string;
  pages: PageHit[];
}

export interface Answer {
  value: string;
  file: string;
  page: number | null;
}

export interface Results {
  query: string;
  confident: boolean | null;
  answer: Answer | null;
  groups: ResultGroup[];
  chunk_ids: string[];
}

export interface SearchResponse {
  results: Results;
  notes: string[];
  terms: string[];
}

export interface ServerConfig {
  agent: boolean;
  formats: string[];
  ready: boolean; // the server has loaded the search models
  examples: string[]; // searches the empty page offers (corpus-drafted, else web.examples)
  filters?: boolean; // file metadata exists (docsearch meta); filters need it
}

export interface Filters {
  formats: string[];
  after: string;
  before: string;
  folder: string;
  undated: boolean;
}

export const NO_FILTERS: Filters = { formats: [], after: "", before: "", folder: "", undated: false };

// GET /api/search/stream: lines of JSON (web/api.py search_stream).
export interface SlidePage {
  unit: string;
  file: string;
  page: number;
}

export type StreamEvent =
  | { event: "waiting"; why: "another search" | "loading" } // another search, or the start-up warm-up, runs first
  | { event: "stage"; stage: "looking" } // its turn came: retrieval runs
  | { event: "stage"; stage: "ranking"; files: Record<string, number> } // the candidates' files per format
  | { event: "early"; results: Results; terms: string[] } // the results that are not slides, in their final order
  | { event: "pages"; pages: SlidePage[]; done: string[] } // the slides the page reranker reads, and those read
  | ({ event: "done" } & SearchResponse)
  | { event: "error"; status: number; detail: string };

// GET /api/text: a page's text, for a preview where there is no page picture.
export interface PageText {
  format: string;
  parts: { title: string; body: string }[];
}
