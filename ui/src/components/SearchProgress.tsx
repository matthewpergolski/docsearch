import type { CSSProperties } from "react";
import { byUse, formatLabel } from "../lib/labels";
import type { Reading, Stage } from "../useLiveSearch";
import { SlideProgress } from "./SlideProgress";

const GLYPH: Record<string, string> = {
  pdf: "glyph-lines",
  docx: "glyph-lines",
  xlsx: "glyph-grid",
  csv: "glyph-grid",
  pptx: "glyph-wide",
  slide_image: "glyph-wide",
  image: "glyph-image",
};

/** A file type as a tiny sheet drawn for its kind; a light passes over it while its step runs. */
export function TypeGlyph({ format, active, delay = 0 }: { format: string; active: boolean; delay?: number }) {
  return (
    <span
      aria-hidden="true"
      className={`glyph ${GLYPH[format] ?? "glyph-lines"} ${active ? "scanning" : ""}`}
      style={{ "--delay": `${delay}s` } as CSSProperties}
    />
  );
}

type State = "done" | "now" | "later";

function Step({ state, label, children }: { state: State; label: string; children?: React.ReactNode }) {
  return (
    <li className={`grid grid-cols-[1rem_minmax(0,1fr)] gap-x-3 gap-y-2 ${state === "later" ? "opacity-40" : ""}`}>
      <span
        aria-hidden="true"
        className={`mt-[0.4rem] size-2 rounded-full ${
          state === "now" ? "breathe bg-hl" : state === "done" ? "bg-fg/50" : "shadow-[0_0_0_1px_var(--muted)]"
        }`}
      />
      <span className={`text-[0.875rem] ${state === "now" ? "text-fg" : "text-muted"}`}>
        {label}
        {state === "done" && <span className="sr-only"> (done)</span>}
      </span>
      {children && <div className="col-start-2">{children}</div>}
    </li>
  );
}

/** The first search for a text, before any result: each step the server runs, as it runs. */
export function SearchProgress({ stage, reading }: { stage: Stage | null; reading: Reading | null }) {
  const step = reading ? "reading" : (stage?.step ?? "looking");
  const order = ["waiting", "looking", "ranking", "reading"];
  const at = (s: string): State => {
    const d = order.indexOf(s) - order.indexOf(step);
    return d < 0 ? "done" : d === 0 ? "now" : "later";
  };
  const files = stage?.step === "ranking" ? stage.files : null;
  const kinds = files ? byUse(Object.keys(files)) : [];
  return (
    <section aria-label="Search progress" className="pt-4" aria-live="polite">
      <ol className="grid gap-5">
        {stage?.step === "waiting" && (
          <Step
            state="now"
            label={
              stage.why === "loading" ? "Waiting for the search models to load" : "Waiting for another search to finish"
            }
          />
        )}
        <Step state={at("looking")} label="Looking through your files" />
        <Step
          state={at("ranking")}
          label={files ? `Ranking the best passages from ${count(files)}` : "Ranking the best passages"}
        >
          {kinds.length > 0 && (
            <ul className="flex flex-wrap gap-x-5 gap-y-2">
              {kinds.map((f, i) => (
                <li key={f} className="flex items-center gap-2 text-[0.8125rem] text-muted tabular-nums">
                  <TypeGlyph format={f} active={step === "ranking"} delay={i * 0.18} />
                  {formatLabel(f)} <span className="text-fg/80">{files?.[f]}</span>
                </li>
              ))}
            </ul>
          )}
        </Step>
        {reading && (
          <Step state="now" label={`Reading the slides: ${reading.done.length} of ${reading.pages.length}`}>
            <SlideProgress reading={reading} heading={false} />
          </Step>
        )}
      </ol>
    </section>
  );
}

const count = (files: Record<string, number>) => {
  const n = Object.values(files).reduce((a, b) => a + b, 0);
  return `${n} ${n === 1 ? "file" : "files"}`;
};
