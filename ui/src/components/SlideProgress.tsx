import { pageUrl } from "../lib/api";
import type { Reading } from "../useLiveSearch";

/** The slides the page reranker is reading for this search, each lit as it is read; the next one is being read. */
export function SlideProgress({ reading, heading = true }: { reading: Reading; heading?: boolean }) {
  const done = new Set(reading.done);
  const next = reading.pages.find((p) => !done.has(p.unit))?.unit;
  return (
    <section aria-label="Reading the slides" className="grid gap-3">
      <p className={heading ? "text-[0.8125rem] text-muted tabular-nums" : "sr-only"} aria-live="polite">
        Reading the slides: {done.size} of {reading.pages.length}
      </p>
      <ol className="grid grid-cols-5 gap-2 sm:grid-cols-10">
        {reading.pages.map((p) => (
          <li
            key={p.unit}
            className={`relative aspect-video overflow-hidden rounded-[3px] bg-sheet shadow-[0_0_0_1px_var(--line)] transition-opacity duration-500 ${
              done.has(p.unit) ? "opacity-100" : p.unit === next ? "opacity-80" : "opacity-30"
            } ${p.unit === next ? "ghost-sheet" : ""}`}
          >
            <img
              src={pageUrl(p.file, "slide_image", p.page) ?? ""}
              alt=""
              className="size-full object-cover object-top"
            />
          </li>
        ))}
      </ol>
    </section>
  );
}
