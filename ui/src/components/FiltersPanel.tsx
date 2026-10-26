import { byUse, formatLabel } from "../lib/labels";
import { type Filters, NO_FILTERS } from "../types";

interface Props {
  formats: string[];
  value: Filters;
  /** now: search at once (a chip, a date); false: wait for a pause (typing a folder) */
  onChange: (f: Filters, now?: boolean) => void;
  available?: boolean; // false: no file metadata yet, so the server refuses every filter
}

const field =
  "rounded-[8px] bg-page px-3 py-1.5 text-fg outline-none shadow-[0_0_0_1px_var(--line)] focus:shadow-[0_0_0_1px_var(--fg)]";

export const filterCount = (f: Filters) =>
  f.formats.length + Number(!!f.after) + Number(!!f.before) + Number(!!f.folder);

/** The filters `search` has: file types, dates and folder. Opened by the Filters button in the search field. */
export function FiltersPanel({ formats, value, onChange, available = true }: Props) {
  const set = (patch: Partial<Filters>, now = true) => onChange({ ...value, ...patch }, now);
  const toggle = (f: string) =>
    set({ formats: value.formats.includes(f) ? value.formats.filter((x) => x !== f) : [...value.formats, f] });
  const row = "grid gap-2 sm:grid-cols-[4.5rem_minmax(0,1fr)] sm:items-center";
  return (
    <div className="grid gap-4 rounded-[14px] bg-sheet p-5 text-[0.875rem] shadow-[0_0_0_1px_var(--line)]">
      {!available && (
        <p role="note" className="border-warn border-l-2 pl-3 text-muted">
          Filters need the file details (type, date, folder), and they are not built yet. Run{" "}
          <code className="text-fg">uv run docsearch meta</code> once, then reload this page.
        </p>
      )}
      <fieldset disabled={!available} className="contents">
        <fieldset className={row}>
          <legend className="sr-only">File types</legend>
          <span aria-hidden="true" className="text-muted">
            Type
          </span>
          <div className="-ml-2.5 flex flex-wrap gap-1">
            {byUse(formats).map((f) => {
              const on = value.formats.includes(f);
              return (
                <label
                  key={f}
                  className={`cursor-pointer rounded-full px-2.5 py-1 transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-fg ${
                    on ? "bg-fg text-page" : "text-fg/80 hover:bg-fg/5 hover:text-fg"
                  }`}
                >
                  <input type="checkbox" className="sr-only" checked={on} onChange={() => toggle(f)} />
                  {formatLabel(f)}
                </label>
              );
            })}
          </div>
        </fieldset>
        <fieldset className={row}>
          <legend className="sr-only">Dates</legend>
          <span aria-hidden="true" className="text-muted">
            Date
          </span>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <input
              type="date"
              aria-label="From"
              className={field}
              value={value.after}
              onChange={(e) => set({ after: e.target.value })}
            />
            <span className="text-muted">to</span>
            <input
              type="date"
              aria-label="To"
              className={field}
              value={value.before}
              onChange={(e) => set({ before: e.target.value })}
            />
            <label className="flex items-center gap-2 text-muted">
              <input
                type="checkbox"
                className="accent-current"
                checked={value.undated}
                disabled={!value.after && !value.before}
                onChange={(e) => set({ undated: e.target.checked })}
              />
              Keep files with no date
            </label>
          </div>
        </fieldset>
        <label className={row}>
          <span className="text-muted">Folder</span>
          <input
            className={`${field} min-w-0`}
            placeholder="e.g. reports/2024"
            value={value.folder}
            onChange={(e) => set({ folder: e.target.value }, false)}
          />
        </label>
      </fieldset>
      {filterCount(value) > 0 && (
        <button
          type="button"
          onClick={() => onChange(NO_FILTERS)}
          className="justify-self-start text-muted underline decoration-line underline-offset-4 hover:text-fg sm:ml-[5rem]"
        >
          Clear filters
        </button>
      )}
    </div>
  );
}
