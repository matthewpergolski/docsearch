import { fileUrl } from "../lib/api";
import { capital, fileName, formatLabel, longDate, pageWord } from "../lib/labels";
import { tableRows } from "../lib/table";
import type { PageHit, ResultGroup } from "../types";
import { PageSheet } from "./PageSheet";
import { Snippet } from "./Snippet";
import { TableSnippet } from "./TableSnippet";

const SHOWN = 3;
const TABLES = new Set(["xlsx", "csv"]);

export function ResultCard({
  group,
  terms,
  onPreview,
  selected = false,
}: {
  group: ResultGroup;
  terms: string[];
  onPreview?: (page: number | null) => void; // on a wide screen: a click on a page shows it beside the results
  selected?: boolean; // its page shows in the preview
}) {
  const word = pageWord(group.format);
  const rest = group.pages
    .slice(SHOWN)
    .flatMap((p) => (p.page === null ? [] : [p.page]))
    .sort((a, b) => a - b);
  const top = group.pages[0]?.page ?? null;
  const link = (page: number | null) => fileUrl(group.file, group.format, page);
  // A click on a page link previews it when there is room for a preview (onPreview), and opens it otherwise.
  const open = (page: number | null) =>
    onPreview
      ? (e: React.MouseEvent) => {
          if (e.metaKey || e.ctrlKey || e.shiftKey) return;
          e.preventDefault();
          onPreview(page);
        }
      : undefined;
  return (
    <article
      className="grid grid-cols-[4rem_minmax(0,1fr)] gap-x-5 gap-y-5 sm:grid-cols-[6.5rem_minmax(0,1fr)] sm:gap-x-8"
      aria-label={`Result ${group.rank}`}
      aria-current={selected || undefined}
    >
      <div className="self-start sm:row-span-2">
        <PageSheet group={group} onClick={open(top)} />
      </div>
      <header className="min-w-0 self-center sm:self-start">
        <h2 className="text-[1.1875rem] leading-snug font-semibold tracking-[-0.01em]">
          <a
            href={link(top)}
            target="_blank"
            rel="noreferrer"
            title="Open the file"
            className="break-words decoration-1 underline-offset-[5px] hover:underline"
          >
            {group.name || fileName(group.file)}
          </a>
        </h2>
        <p className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[0.8125rem] text-muted">
          <span className="text-fg/80">{formatLabel(group.format)}</span>
          {group.period && <span>{group.period}</span>}
          {group.date && <span>{longDate(group.date)}</span>}
          <span className="min-w-0 break-all">{group.file}</span>
        </p>
      </header>
      <div className="col-span-2 grid min-w-0 gap-5 sm:col-span-1 sm:col-start-2">
        <ul className="grid gap-5">
          {group.pages.slice(0, SHOWN).map((p) => (
            <Passage
              key={`${p.page}-${p.rank}`}
              hit={p}
              word={word}
              terms={terms}
              table={TABLES.has(group.format)}
              href={link}
              onClick={open}
            />
          ))}
        </ul>
        {rest.length > 0 && (
          <p className="text-[0.8125rem] text-muted">
            Also on {rest.length === 1 ? word : `${word}s`}{" "}
            {rest.map((n, i) => (
              <span key={n}>
                {i > 0 && ", "}
                <a
                  href={link(n)}
                  target="_blank"
                  rel="noreferrer"
                  onClick={open(n)}
                  aria-label={`Open ${word} ${n}`}
                  className="text-fg/80 tabular-nums underline decoration-line underline-offset-4 hover:text-fg hover:decoration-fg"
                >
                  {n}
                </a>
              </span>
            ))}
          </p>
        )}
      </div>
    </article>
  );
}

function Passage(p: {
  hit: PageHit;
  word: string;
  terms: string[];
  table: boolean;
  href: (page: number | null) => string;
  onClick: (page: number | null) => ((e: React.MouseEvent) => void) | undefined;
}) {
  const { hit } = p;
  const rows = p.table ? tableRows(hit.snippet) : null;
  return (
    <li className="grid gap-1.5">
      {(hit.page !== null || hit.title) && (
        <p className="flex min-w-0 items-baseline gap-3 text-[0.8125rem]">
          {hit.page !== null && (
            <a
              href={p.href(hit.page)}
              target="_blank"
              rel="noreferrer"
              onClick={p.onClick(hit.page)}
              aria-label={`Open ${p.word} ${hit.page}`}
              className="shrink-0 text-muted tabular-nums underline decoration-line underline-offset-4 hover:text-fg hover:decoration-fg"
            >
              {capital(p.word)} {hit.page}
            </a>
          )}
          {hit.title && <span className="min-w-0 truncate font-medium text-fg/85">{hit.title}</span>}
        </p>
      )}
      {rows ? (
        <TableSnippet table={rows} terms={p.terms} />
      ) : (
        <p className="font-text text-[1rem] leading-[1.65] text-fg/90">
          <Snippet text={hit.snippet} terms={p.terms} />
        </p>
      )}
    </li>
  );
}
