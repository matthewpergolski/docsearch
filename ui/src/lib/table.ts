export interface Table {
  before: boolean; // the snippet starts inside the table
  after: boolean; // the snippet ends inside it
  rows: string[][];
}

/** A spreadsheet snippet (display.title_and_body: rows joined by " / ", cells by " | ") as rows of cells; null when
 * no row has two cells, so prose that happens to hold " / " stays prose. */
export function tableRows(snippet: string): Table | null {
  const before = snippet.startsWith("…");
  const after = snippet.endsWith("…");
  const body = snippet.slice(before ? 1 : 0, after ? -1 : undefined);
  const rows = body.split(" / ").map((row) => {
    const cells = row.split("|").map((c) => c.trim());
    if (cells.length > 1 && cells[0] === "") cells.shift();
    if (cells.length > 1 && cells.at(-1) === "") cells.pop();
    return cells;
  });
  return rows.some((r) => r.length > 1) ? { before, after, rows } : null;
}
