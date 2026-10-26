import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, streamSearch } from "./lib/api";
import { type Filters, NO_FILTERS, type SearchResponse, type SlidePage } from "./types";

// Search as the user types: the full search after a short pause, or at once on Enter. A new search replaces the
// one running: the browser drops its request, and the server stops it between pages and answers 409. Answers are
// kept, so going back to a text is instant. While the page reranker reads slides (the slow step), the other
// results show in their final order, and `reading` says which slides are read.
export const MIN_CHARS = 3;
const PAUSE_MS = 350;

export type View =
  | { kind: "idle" }
  | { kind: "waiting" } // the first search for this text is running; nothing to show yet
  | { kind: "error"; message: string }
  | { kind: "results"; response: SearchResponse; seconds: number | null; early?: boolean };

export interface Reading {
  pages: SlidePage[];
  done: string[];
}

/** What the server is doing for the search running (GET /api/search/stream). */
export type Stage =
  | { step: "waiting"; why: "another search" | "loading" }
  | { step: "looking" }
  | { step: "ranking"; files: Record<string, number> };

const keyOf = (text: string, filters: Filters, agent: boolean) => JSON.stringify([text, filters, agent]);

export function useLiveSearch(agent: boolean) {
  const [query, setQueryState] = useState("");
  const [filters, setFiltersState] = useState<Filters>(NO_FILTERS);
  const [view, setView] = useState<View>({ kind: "idle" });
  const [busy, setBusy] = useState(false);
  const [reading, setReading] = useState<Reading | null>(null);
  const [stage, setStage] = useState<Stage | null>(null);
  const cache = useRef(new Map<string, SearchResponse>());
  const inflight = useRef<AbortController | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const run = useCallback(
    async (text: string, f: Filters) => {
      inflight.current?.abort();
      const ctl = new AbortController();
      inflight.current = ctl;
      setBusy(true);
      setReading(null);
      setStage(null);
      const t0 = performance.now();
      try {
        const response = await streamSearch(
          text,
          f,
          agent,
          (e) => {
            if (inflight.current !== ctl) return;
            if (e.event === "waiting") setStage({ step: "waiting", why: e.why });
            else if (e.event === "stage")
              setStage(e.stage === "ranking" ? { step: "ranking", files: e.files } : { step: "looking" });
            else if (e.event === "early" && e.results.groups.length > 0)
              setView({
                kind: "results",
                response: { results: e.results, notes: [], terms: e.terms },
                seconds: null,
                early: true,
              });
            else if (e.event === "pages") setReading({ pages: e.pages, done: e.done });
          },
          ctl.signal,
        );
        cache.current.set(keyOf(text, f, agent), response);
        if (inflight.current === ctl) {
          setReading(null);
          setStage(null);
          setView({ kind: "results", response, seconds: (performance.now() - t0) / 1000 });
        }
      } catch (e) {
        if (ctl.signal.aborted || (e instanceof ApiError && e.status === 409)) return;
        if (inflight.current === ctl) {
          setReading(null);
          setStage(null);
          setView({ kind: "error", message: e instanceof Error ? e.message : String(e) });
        }
      } finally {
        if (inflight.current === ctl) {
          inflight.current = null;
          setBusy(false);
        }
      }
    },
    [agent],
  );

  const schedule = useCallback(
    (raw: string, f: Filters, now: boolean) => {
      clearTimeout(timer.current);
      const text = raw.trim();
      if (text.length < MIN_CHARS) {
        inflight.current?.abort();
        inflight.current = null;
        setBusy(false);
        setReading(null);
        setStage(null);
        setView({ kind: "idle" });
        return;
      }
      const kept = cache.current.get(keyOf(text, f, agent));
      if (kept) {
        inflight.current?.abort();
        inflight.current = null;
        setBusy(false);
        setReading(null);
        setStage(null);
        setView({ kind: "results", response: kept, seconds: null });
        return;
      }
      setView((v) => (v.kind === "idle" || v.kind === "error" ? { kind: "waiting" } : v));
      setBusy(true);
      if (now) run(text, f);
      else timer.current = setTimeout(() => run(text, f), PAUSE_MS);
    },
    [agent, run],
  );

  useEffect(() => () => clearTimeout(timer.current), []);

  return {
    query,
    filters,
    view,
    busy,
    reading,
    stage,
    type: (q: string) => {
      setQueryState(q);
      schedule(q, filters, false);
    },
    submit: (q = query) => {
      setQueryState(q);
      schedule(q, filters, true);
    },
    setFilters: (f: Filters, now = true) => {
      setFiltersState(f);
      schedule(query, f, now);
    },
    clear: () => {
      setQueryState("");
      schedule("", filters, true);
    },
  };
}
