import { useEffect, useState } from "react";
import { fetchPageText, fileUrl, pageUrl } from "../lib/api";
import { capital, fileName, pageWord } from "../lib/labels";
import { tableRows } from "../lib/table";
import type { PageText, ResultGroup } from "../types";
import { Snippet } from "./Snippet";
import { TableSnippet } from "./TableSnippet";

const TABLES = new Set(["xlsx", "csv"]);

/** On a wide screen, the page the user clicked, large enough to read: its picture, or, where there is none (a
 * spreadsheet, a page `render` did not draw), the text the index holds for it. */
export function PagePreview({
  group,
  page,
  terms,
  onClose,
}: {
  group: ResultGroup;
  page: number | null;
  terms: string[];
  onClose: () => void;
}) {
  const src = pageUrl(group.file, group.format, page);
  const [failed, setFailed] = useState<string | null>(null);
  const where = page === null ? "the whole file" : `${pageWord(group.format)} ${page}`;
  const showText = !src || failed === src;
  return (
    <section aria-label="Page preview" className="grid gap-3">
      <div className="flex min-w-0 items-center justify-between gap-4 text-[0.8125rem]">
        <p className="min-w-0 truncate">
          <span className="font-medium">{group.name || fileName(group.file)}</span>
          <span className="ml-3 text-muted">{capital(where)}</span>
        </p>
        <span className="flex shrink-0 items-center gap-3">
          <a
            href={fileUrl(group.file, group.format, page)}
            target="_blank"
            rel="noreferrer"
            className="text-muted underline decoration-line underline-offset-4 hover:text-fg hover:decoration-fg"
          >
            Open {where === "the whole file" ? "the file" : where}
          </a>
          <button
            type="button"
            aria-label="Close the preview"
            title="Close (Esc)"
            onClick={onClose}
            className="grid size-8 place-items-center rounded-full text-[1.125rem] text-muted hover:bg-fg/5 hover:text-fg"
          >
            ×
          </button>
        </span>
      </div>
      {showText ? (
        <TextPage key={`${group.file}#${page}`} group={group} page={page} terms={terms} />
      ) : (
        <img
          key={src}
          src={src ?? ""}
          alt=""
          onError={() => setFailed(src)}
          className="max-h-[calc(100dvh-7rem)] w-full rounded-[3px] bg-sheet object-contain object-top shadow-[0_0_0_1px_var(--line),0_18px_40px_-24px_var(--shadow)]"
        />
      )}
    </section>
  );
}

function TextPage({ group, page, terms }: { group: ResultGroup; page: number | null; terms: string[] }) {
  const [text, setText] = useState<PageText | "loading" | "none">("loading");
  useEffect(() => {
    const ctl = new AbortController();
    fetchPageText(group.file, page, ctl.signal)
      .then(setText)
      .catch(() => !ctl.signal.aborted && setText("none"));
    return () => ctl.abort();
  }, [group.file, page]);
  const box =
    "max-h-[calc(100dvh-7rem)] overflow-auto rounded-[3px] bg-sheet p-8 shadow-[0_0_0_1px_var(--line),0_18px_40px_-24px_var(--shadow)]";
  if (text === "loading") return <div className={`${box} text-[0.875rem] text-muted`}>Loading the page text…</div>;
  if (text === "none")
    return (
      <div className={`${box} text-[0.875rem] text-muted`}>
        This page has no picture or text to show. Open the file to see it.
      </div>
    );
  return (
    <article className={`${box} grid content-start gap-6`}>
      {text.parts.map((part, i) => {
        const rows = TABLES.has(text.format) ? tableRows(part.body) : null;
        return (
          // biome-ignore lint/suspicious/noArrayIndexKey: parts keep the page's order and have no other identity
          <div key={i} className="grid gap-2">
            {part.title && <p className="text-[0.8125rem] font-medium text-fg/85">{part.title}</p>}
            {rows ? (
              <TableSnippet table={rows} terms={terms} />
            ) : (
              <p className="font-text text-[1rem] leading-[1.65] text-fg/90">
                <Snippet text={part.body} terms={terms} />
              </p>
            )}
          </div>
        );
      })}
    </article>
  );
}
