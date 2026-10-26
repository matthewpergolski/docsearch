import type { Ref } from "react";
import { FiltersIcon } from "./icons";

interface Props {
  query: string;
  busy?: boolean;
  filters: number; // filters set
  filtersOpen: boolean;
  inputRef?: Ref<HTMLInputElement>;
  onType: (query: string) => void;
  onSubmit: (query: string) => void;
  onClear: () => void;
  onToggleFilters: () => void;
}

/** Searches as the user types; Enter searches at once, Escape or the × starts over. While a search runs, a light
 * passes along the field's lower edge. */
export function SearchBox(p: Props) {
  return (
    <search>
      <form
        className="relative flex items-center gap-1 overflow-hidden rounded-[14px] bg-sheet p-2 pl-6 shadow-[0_0_0_1px_var(--line)] transition-shadow focus-within:shadow-[0_0_0_1px_color-mix(in_srgb,var(--fg)_55%,transparent)]"
        onSubmit={(e) => {
          e.preventDefault();
          p.onSubmit(p.query);
        }}
      >
        <input
          ref={p.inputRef}
          type="search"
          aria-label="Search your files"
          placeholder="Ask about your files"
          autoComplete="off"
          spellCheck={false}
          value={p.query}
          onChange={(e) => p.onType(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape" && p.query) {
              e.preventDefault();
              p.onClear();
            }
          }}
          className="min-w-0 flex-1 bg-transparent py-2.5 text-[1.25rem] text-fg outline-none placeholder:text-muted/70 focus-visible:outline-none sm:text-[1.375rem]"
        />
        {p.query && (
          <button
            type="button"
            aria-label="Clear the search"
            title="Clear (Esc)"
            onClick={p.onClear}
            className="grid size-9 shrink-0 place-items-center rounded-full text-[1.25rem] text-muted hover:bg-fg/5 hover:text-fg"
          >
            ×
          </button>
        )}
        <button
          type="button"
          aria-expanded={p.filtersOpen}
          onClick={p.onToggleFilters}
          className={`flex shrink-0 items-center gap-2 rounded-[9px] px-3 py-2 text-[0.875rem] transition-colors ${
            p.filtersOpen || p.filters ? "bg-fg/[0.06] text-fg" : "text-muted hover:bg-fg/5 hover:text-fg"
          }`}
        >
          <FiltersIcon />
          <span className="max-sm:sr-only">Filters</span>
          {p.filters > 0 && <span>{` (${p.filters})`}</span>}
        </button>
        {p.busy && <span aria-hidden="true" className="scan" />}
      </form>
    </search>
  );
}
