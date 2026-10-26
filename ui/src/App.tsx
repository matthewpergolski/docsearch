import { useEffect, useRef, useState } from "react";
import { AgentAnswerCard } from "./components/AgentAnswerCard";
import { Examples } from "./components/Examples";
import { FiltersPanel, filterCount } from "./components/FiltersPanel";
import { NoConfidentBanner } from "./components/NoConfidentBanner";
import { PagePreview } from "./components/PagePreview";
import { ResultCard } from "./components/ResultCard";
import { SearchBox } from "./components/SearchBox";
import { SearchProgress } from "./components/SearchProgress";
import { SlideProgress } from "./components/SlideProgress";
import { ThemeToggle } from "./components/ThemeToggle";
import { fetchConfig } from "./lib/api";
import { flip, loadChoice, resolveTheme, saveChoice, type Theme, type ThemeChoice } from "./lib/theme";
import type { SearchResponse, ServerConfig } from "./types";
import { useLiveSearch, type View } from "./useLiveSearch";

const READY_POLL_MS = 2000;

/** The theme on screen, and the one button that flips it. */
function useTheme(): [Theme, () => void] {
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const [choice, setChoice] = useState<ThemeChoice>(loadChoice);
  const [systemDark, setSystemDark] = useState(media.matches);
  useEffect(() => {
    const onChange = () => setSystemDark(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [media]);
  const shown = resolveTheme(choice, systemDark);
  useEffect(() => {
    document.documentElement.dataset.theme = shown;
  }, [shown]);
  return [
    shown,
    () => {
      const next = flip(shown, systemDark);
      saveChoice(next);
      setChoice(next);
    },
  ];
}

const WIDE = "(min-width: 1280px)";

/** Whether the screen has room for a page preview beside the results. */
function useWideScreen(): boolean {
  const [wide, setWide] = useState(() => window.matchMedia(WIDE).matches);
  useEffect(() => {
    const media = window.matchMedia(WIDE);
    const onChange = () => setWide(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);
  return wide;
}

/** The server's settings, asked again until it says the models are loaded. */
function useServerConfig(): ServerConfig | null {
  const [config, setConfig] = useState<ServerConfig | null>(null);
  useEffect(() => {
    const ctl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const ask = () =>
      fetchConfig(ctl.signal)
        .then((c) => {
          setConfig(c);
          if (c.ready === false) timer = setTimeout(ask, READY_POLL_MS);
        })
        .catch(() => {});
    ask();
    return () => {
      ctl.abort();
      clearTimeout(timer);
    };
  }, []);
  return config;
}

export function App() {
  const [theme, flipTheme] = useTheme();
  const config = useServerConfig();
  const [agent, setAgent] = useState(false);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const live = useLiveSearch(!!config?.agent && agent);
  const input = useRef<HTMLInputElement>(null);
  const { view } = live;
  const idle = view.kind === "idle";
  const roomy = useWideScreen(); // room for a page preview beside the results
  const [picked, setPicked] = useState<{ file: string; page: number | null } | null>(null);
  const groups = view.kind === "results" ? view.response.results.groups : [];
  const pickedGroup = picked ? groups.find((g) => g.file === picked.file) : undefined;
  const wide = roomy && !!pickedGroup; // results on the left, the clicked page on the right

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement;
      if (e.key === "Escape" && !typing) setPicked(null);
      if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        input.current?.focus();
        input.current?.select();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const startOver = () => {
    live.clear();
    input.current?.focus();
  };

  return (
    <div className={`mx-auto px-5 pb-24 sm:px-8 ${wide ? "max-w-[48rem] xl:max-w-[92rem]" : "max-w-[48rem]"}`}>
      <header className="flex items-center justify-between py-5">
        <a
          href="/"
          onClick={(e) => {
            e.preventDefault();
            startOver();
          }}
          className="text-[0.9375rem] font-semibold tracking-[0.01em]"
          title="New search"
        >
          docsearch
        </a>
        <ThemeToggle shown={theme} onFlip={flipTheme} />
      </header>
      <div className={wide ? "xl:grid xl:grid-cols-[minmax(0,46rem)_minmax(0,1fr)] xl:gap-14" : ""}>
        <div className={`stage flex min-w-0 flex-col gap-4 ${idle ? "pt-[12vh] sm:pt-[16vh]" : "pt-3 sm:pt-5"}`}>
          {idle && (
            <div className="grid gap-2 pb-4">
              <h1 className="text-[2.5rem] leading-[1.1] font-extralight tracking-[-0.025em] sm:text-[3.25rem]">
                Find the page.
              </h1>
              <p className="text-[1rem] text-muted">
                Search the files on this computer, and open the page that answers.
              </p>
            </div>
          )}
          <SearchBox
            inputRef={input}
            query={live.query}
            busy={live.busy || config?.ready === false}
            filters={filterCount(live.filters)}
            filtersOpen={filtersOpen}
            onType={live.type}
            onSubmit={(q) => live.submit(q)}
            onClear={startOver}
            onToggleFilters={() => setFiltersOpen((v) => !v)}
          />
          {filtersOpen && (
            <FiltersPanel
              formats={config?.formats ?? []}
              value={live.filters}
              onChange={live.setFilters}
              available={config?.filters !== false}
            />
          )}
          {config?.agent && (
            <label className="flex items-center gap-2 text-[0.875rem] text-muted">
              <input
                type="checkbox"
                className="size-4 accent-current"
                checked={agent}
                onChange={(e) => setAgent(e.target.checked)}
              />
              Ask the LLM to read the top pages
            </label>
          )}
          <StatusLine ready={config?.ready !== false} view={view} busy={live.busy} reading={!!live.reading} />
          {view.kind === "error" && (
            <p role="alert" className="text-bad">
              {view.message}
            </p>
          )}
          {idle && (
            <div className="pt-8">
              <Examples examples={config?.examples ?? []} onPick={(q) => live.submit(q)} />
            </div>
          )}
          {view.kind === "waiting" && <SearchProgress stage={live.stage} reading={live.reading} />}
          {view.kind === "results" && live.reading && (
            <div className="pt-2">
              <SlideProgress reading={live.reading} />
            </div>
          )}
          {view.kind === "results" && (
            <ResultsView
              response={view.response}
              picked={wide ? picked?.file : undefined}
              onPreview={roomy ? (file, page) => setPicked({ file, page }) : undefined}
            />
          )}
        </div>
        {wide && pickedGroup && picked && view.kind === "results" && (
          <aside className="hidden xl:block">
            <div className="sticky top-5 pt-5">
              <PagePreview
                group={pickedGroup}
                page={picked.page}
                terms={view.response.terms}
                onClose={() => setPicked(null)}
              />
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

function StatusLine({ ready, view, busy, reading }: { ready: boolean; view: View; busy: boolean; reading: boolean }) {
  let text = "";
  if (!ready) text = "Loading the search models. The first search waits for them, about half a minute.";
  else if (view.kind === "waiting")
    text = reading ? "" : "Searching…"; // the slide strip says it
  else if (view.kind === "results") {
    const n = view.response.results.groups.length;
    const files = `${n} ${n === 1 ? "file" : "files"}`;
    text = view.early
      ? `${files} so far`
      : busy
        ? `${files}; searching again…`
        : view.seconds === null
          ? files
          : `${files} in ${view.seconds.toFixed(1)} s`;
  }
  return (
    <p className="min-h-[1.25rem] text-[0.8125rem] text-muted tabular-nums" aria-live="polite">
      {text}
    </p>
  );
}

function ResultsView({
  response,
  picked,
  onPreview,
}: {
  response: SearchResponse;
  picked?: string;
  onPreview?: (file: string, page: number | null) => void;
}) {
  const { results, notes, terms } = response;
  const answerFormat = results.groups.find((g) => g.file === results.answer?.file)?.format;
  return (
    <main className="flex flex-col gap-6 pt-2">
      {notes.map((n) => (
        <p key={n} className="text-[0.875rem] text-muted">
          {n}
        </p>
      ))}
      {results.answer && <AgentAnswerCard answer={results.answer} format={answerFormat} />}
      {results.confident === false && <NoConfidentBanner />}
      <div className="flex flex-col divide-y divide-line">
        {results.groups.map((g) => (
          <div key={g.file} className="py-8 first:pt-2">
            <ResultCard
              group={g}
              terms={terms}
              selected={g.file === picked}
              onPreview={onPreview && ((page) => onPreview(g.file, page))}
            />
          </div>
        ))}
      </div>
    </main>
  );
}
