import { expect, test } from "bun:test";
import { NO_FILTERS } from "../types";
import { fileUrl, pageUrl, readLines, searchUrl } from "./api";

test("a plain search sends the query and k only", () => {
  expect(searchUrl("q3 freight", NO_FILTERS, false)).toBe("/api/search?q=q3+freight&k=10");
});

test("filters and the agent toggle become query parameters", () => {
  const url = searchUrl(
    "x",
    { formats: ["pdf", "docx"], after: "2024-01-01", before: "", folder: " hr/2024 ", undated: true },
    true,
  );
  expect(url).toBe(
    "/api/search?q=x&k=10&formats=pdf&formats=docx&after=2024-01-01&folder=hr%2F2024&undated=true&agent=true",
  );
});

test("keep-undated is sent only with a date", () => {
  expect(searchUrl("x", { ...NO_FILTERS, undated: true }, false)).toBe("/api/search?q=x&k=10");
});

test("a file link names the file and opens a PDF at its page", () => {
  expect(fileUrl("reports/a b.pdf", "pdf", 4)).toBe("/api/file?file=reports%2Fa+b.pdf&page=4#page=4");
  expect(fileUrl("slides/deck", "slide_image", 2)).toBe("/api/file?file=slides%2Fdeck&page=2");
  expect(fileUrl("finance/costs.xlsx", "xlsx", null)).toBe("/api/file?file=finance%2Fcosts.xlsx");
});

test("a page picture is asked for only where docsearch render draws pages", () => {
  expect(pageUrl("reports/a b.pdf", "pdf", 4)).toBe("/api/page?file=reports%2Fa+b.pdf&page=4");
  expect(pageUrl("photos/scan.png", "image", null)).toBe("/api/page?file=photos%2Fscan.png&page=1");
  expect(pageUrl("finance/costs.xlsx", "xlsx", null)).toBeNull();
  expect(pageUrl("data/t.csv", "csv", null)).toBeNull();
});

test("the search stream's lines are read whole, however the network splits them", async () => {
  const parts = ['{"event":"ear', 'ly","n":1}\n{"event":"pages"', ',"n":2}\n{"event":"done","n":3}\n'];
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      for (const p of parts) c.enqueue(new TextEncoder().encode(p));
      c.close();
    },
  });
  const seen: unknown[] = [];
  await readLines(body, (e) => seen.push(e));
  expect(seen).toEqual([
    { event: "early", n: 1 },
    { event: "pages", n: 2 },
    { event: "done", n: 3 },
  ]);
});
