import { useLayoutEffect, useRef, useState } from "react";
import type { Table } from "../lib/table";
import { Snippet } from "./Snippet";

const cut = <td className="py-1.5 pr-6 text-muted">…</td>;

/** A spreadsheet snippet as the rows it came from; "…" where the snippet cut a row. A table wider than the column
 * scrolls, and its right edge fades to say so. */
export function TableSnippet({ table, terms }: { table: Table; terms: string[] }) {
  const box = useRef<HTMLDivElement>(null);
  const [wide, setWide] = useState(false);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const measure = () => setWide(el.scrollWidth > el.clientWidth + 1);
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const last = table.rows.length - 1;
  return (
    <div
      ref={box}
      className="max-w-full overflow-x-auto"
      style={wide ? { maskImage: "linear-gradient(to right, black calc(100% - 3rem), transparent)" } : undefined}
    >
      <table className="border-collapse font-text text-[0.9375rem] tabular-nums leading-snug text-fg/90">
        <tbody>
          {table.rows.map((row, r) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: rows have no identity beyond their place
            <tr key={r} className="border-line border-b last:border-b-0">
              {r === 0 && table.before && cut}
              {row.map((cell, c) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: cells have no identity beyond their place
                <td key={c} className={`py-1.5 pr-6 align-top whitespace-nowrap ${c > 0 ? "text-right" : ""}`}>
                  <Snippet text={cell} terms={terms} />
                </td>
              ))}
              {r === last && table.after && cut}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
