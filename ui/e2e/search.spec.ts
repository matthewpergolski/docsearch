import { expect, type Page, test } from "@playwright/test";
import { RESPONSE } from "../src/test/fixtures";

const EXAMPLES = ["What date had the most ridership?", "Q3 freight cost by region"];

const lines = (...events: object[]) => events.map((e) => `${JSON.stringify(e)}\n`).join("");

async function mockApi(
  page: Page,
  { agent = false, confident = true as boolean | null, status = 200, ready = true } = {},
) {
  const seen: string[] = [];
  await page.route("**/api/config", (r) =>
    r.fulfill({ json: { agent, ready, examples: EXAMPLES, formats: ["docx", "pdf", "pptx", "xlsx"], filters: true } }),
  );
  await page.route("**/api/search/stream?**", (r) => {
    seen.push(r.request().url());
    if (status !== 200) return r.fulfill({ status, json: { detail: "search filters need file metadata" } });
    const answer =
      agent && r.request().url().includes("agent=true")
        ? { value: "$4.2M", file: "reports/supplier-deck-q3.pptx", page: 4 }
        : null;
    return r.fulfill({
      body: lines({ event: "done", ...RESPONSE, results: { ...RESPONSE.results, confident, answer } }),
    });
  });
  return seen;
}

async function search(page: Page, q = "Q3 freight cost by region") {
  const box = page.getByRole("searchbox", { name: "Search your files" });
  await box.fill(q);
  await box.press("Enter");
}

test("a search shows results grouped by file, with the query's words marked", async ({ page }, info) => {
  await mockApi(page);
  await page.goto("/");
  await search(page);
  const cards = page.getByRole("article");
  await expect(cards).toHaveCount(2);
  await expect(cards.first().getByRole("heading")).toHaveText("Supplier review Q3");
  await expect(cards.first().locator("mark").first()).toHaveText("Freight");
  await expect(page.getByText("Also on slide 12")).toBeVisible();
  await expect(page.getByText("No confident match")).toHaveCount(0);
  await expect(page.getByText("2 files in")).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
  await page.screenshot({ path: info.outputPath("results.png"), fullPage: true });
});

test("a weak top result shows the No confident match banner above the results", async ({ page }) => {
  await mockApi(page, { confident: false });
  await page.goto("/");
  await search(page);
  await expect(page.getByRole("status")).toContainText("No confident match");
  await expect(page.getByRole("article")).toHaveCount(2);
});

test("the agent toggle shows only when agent mode is on, and sends agent=true", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await expect(page.getByText("Ask the LLM to read the top pages")).toHaveCount(0);
  await page.unrouteAll();
  const seen = await mockApi(page, { agent: true });
  await page.goto("/");
  await page.getByLabel("Ask the LLM to read the top pages").check();
  await search(page);
  await expect(page.getByRole("region", { name: "Answer" })).toContainText("$4.2M");
  expect(seen.at(-1)).toContain("agent=true");
});

test("filters reach the API", async ({ page }) => {
  const seen = await mockApi(page);
  await page.goto("/");
  await search(page);
  await page.getByRole("button", { name: "Filters" }).click();
  await page.getByText("PDF", { exact: true }).click();
  await expect(page.getByLabel("PDF")).toBeChecked();
  await expect.poll(() => seen.at(-1)).toContain("formats=pdf");
  await page.getByLabel("Folder").fill("finance");
  await expect.poll(() => seen.at(-1)).toContain("folder=finance");
  await expect(page.getByRole("button", { name: "Filters (2)" })).toBeVisible();
});

test("typing searches after a pause; Escape starts over; a text searched before comes back at once", async ({
  page,
}) => {
  const seen = await mockApi(page);
  await page.goto("/");
  const box = page.getByRole("searchbox", { name: "Search your files" });
  await box.pressSequentially("freight by region", { delay: 40 });
  await expect(page.getByRole("article")).toHaveCount(2);
  await expect.poll(() => seen.at(-1)).toMatch(/stream\?q=freight\+by\+region&k=30$/);
  expect(seen.length).toBeLessThanOrEqual(2);
  const asked = seen.length;
  await box.press("Escape");
  await expect(box).toHaveValue("");
  await expect(page.getByRole("heading", { name: "Find the page." })).toBeVisible();
  await expect(page.getByRole("article")).toHaveCount(0);
  await box.fill("freight by region");
  await expect(page.getByRole("article")).toHaveCount(2);
  await page.waitForTimeout(700);
  expect(seen.length).toBe(asked);
  await page.getByRole("button", { name: "Clear the search" }).click();
  await expect(page.getByRole("region", { name: "Searches to try" })).toBeVisible();
});

test("a search the server replaced (409) shows no error; each search names this tab", async ({ page }) => {
  await page.route("**/api/config", (r) =>
    r.fulfill({ json: { agent: false, ready: true, examples: [], formats: [] } }),
  );
  const tabs: (string | undefined)[] = [];
  await page.route("**/api/search/stream?**", (r) => {
    tabs.push(r.request().headers()["x-docsearch-tab"]);
    return r.fulfill({
      body: r.request().url().includes("old")
        ? lines({ event: "error", status: 409, detail: "a newer search replaced this one" })
        : lines({ event: "done", ...RESPONSE }),
    });
  });
  await page.goto("/");
  await search(page, "old text");
  await search(page, "new text");
  await expect(page.getByRole("article")).toHaveCount(2);
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(new Set(tabs).size).toBe(1);
  expect(tabs[0]).toBeTruthy();
});

test("the empty page offers example searches, and one click runs it", async ({ page }) => {
  const seen = await mockApi(page);
  await page.goto("/");
  await page.getByRole("button", { name: EXAMPLES[0] }).click();
  await expect(page.getByRole("article")).toHaveCount(2);
  await expect(page.getByRole("searchbox", { name: "Search your files" })).toHaveValue(EXAMPLES[0]);
  expect(seen.at(-1)).toContain("ridership");
  await expect(page.getByRole("region", { name: "Searches to try" })).toHaveCount(0);
});

test("while the server loads its models, the page says so", async ({ page }) => {
  await mockApi(page, { ready: false });
  await page.goto("/");
  await expect(page.getByText("Loading the search models")).toBeVisible();
});

test("a result's heading opens its file at the best page in a new tab", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await search(page);
  const link = page.getByRole("article").first().getByRole("link", { name: "Supplier review Q3" });
  await expect(link).toHaveAttribute("href", "/api/file?file=reports%2Fsupplier-deck-q3.pptx&page=4#page=4");
  await expect(link).toHaveAttribute("target", "_blank");
});

test("a result shows the picture of its best page, or a ruled sheet when there is none", async ({ page }) => {
  await mockApi(page);
  const png = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
    "base64",
  );
  const asked: string[] = [];
  await page.route("**/api/page?**", (r) => {
    asked.push(r.request().url());
    return r.fulfill({ contentType: "image/png", body: png });
  });
  await page.goto("/");
  await search(page);
  const cards = page.getByRole("article");
  await expect(cards.first().locator("img")).toHaveAttribute(
    "src",
    "/api/page?file=reports%2Fsupplier-deck-q3.pptx&page=4",
  );
  await expect(cards.nth(1).locator("img")).toHaveCount(0);
  await expect(cards.nth(1).locator(".sheet-grid")).toBeVisible();
  await expect
    .poll(() => [...new Set(asked.map((u) => new URL(u).search))])
    .toEqual(["?file=reports%2Fsupplier-deck-q3.pptx&page=4"]);
});

test("each step shows as it runs; the other results come before the slides; then the final results", async ({
  page,
}) => {
  // The test feeds the search stream one event at a time (route.fulfill sends a body all at once).
  await page.addInitScript(() => {
    const real = window.fetch;
    const w = window as unknown as { feed: (line: string) => void; end: () => void };
    window.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      if (!String(input).includes("/api/search/stream")) return real(input, init);
      const body = new ReadableStream<Uint8Array>({
        start(c) {
          w.feed = (line) => c.enqueue(new TextEncoder().encode(`${line}\n`));
          w.end = () => c.close();
        },
      });
      return new Response(body, { headers: { "Content-Type": "application/x-ndjson" } });
    }) as typeof fetch;
  });
  await page.route("**/api/config", (r) =>
    r.fulfill({ json: { agent: false, ready: true, examples: [], formats: [] } }),
  );
  await page.goto("/");
  await search(page);
  const feed = (e: object) =>
    page.evaluate((line) => (window as unknown as { feed: (l: string) => void }).feed(line), JSON.stringify(e));
  const pages = [1, 2].map((n) => ({ unit: `s/deck_page_${n}#p1`, file: `s/deck_page_${n}`, page: 1 }));
  await page.waitForFunction(() => "feed" in window);
  const progress = page.getByRole("region", { name: "Search progress" });
  await feed({ event: "waiting", why: "another search" });
  await expect(progress).toContainText("Waiting for another search to finish");
  await feed({ event: "stage", stage: "looking" });
  await feed({ event: "stage", stage: "ranking", files: { pptx: 1, xlsx: 1 } });
  await expect(progress).toContainText("Ranking the best passages from 2 files");
  await expect(progress.locator(".glyph.scanning")).toHaveCount(2);
  await feed({
    event: "early",
    results: { ...RESPONSE.results, groups: [RESPONSE.results.groups[1]] },
    terms: RESPONSE.terms,
  });
  await expect(page.getByRole("article")).toHaveCount(1);
  await expect(page.getByText("1 file so far")).toBeVisible();
  await feed({ event: "pages", pages, done: [pages[0].unit] });
  const strip = page.getByRole("region", { name: "Reading the slides" });
  await expect(strip).toContainText("Reading the slides: 1 of 2");
  await expect(strip.locator("img")).toHaveCount(2);
  await feed({ event: "done", ...RESPONSE });
  await page.evaluate(() => (window as unknown as { end: () => void }).end());
  await expect(page.getByRole("article")).toHaveCount(2);
  await expect(page.getByText("2 files in")).toBeVisible();
  await expect(strip).toHaveCount(0);
});

test("on a wide screen a click on a page previews it beside the results; a phone opens the file", async ({
  page,
}, info) => {
  await mockApi(page);
  await page.route("**/api/text?**", (r) =>
    r.fulfill({ json: { format: "xlsx", parts: [{ title: "Freight", body: "Region | Q3 freight / North | 4.2" }] } }),
  );
  await page.goto("/");
  await search(page);
  const preview = page.getByRole("region", { name: "Page preview" });
  const first = page.getByRole("article").first();
  await expect(page.getByRole("article")).toHaveCount(2);
  await expect(preview).toHaveCount(0);
  if (info.project.name.endsWith("phone")) {
    await expect(first.getByRole("link", { name: "Open slide 7" })).toHaveAttribute("target", "_blank");
    return;
  }
  await first.getByRole("link", { name: "Open slide 7" }).click();
  await expect(preview.getByRole("link", { name: "Open slide 7" })).toHaveAttribute(
    "href",
    "/api/file?file=reports%2Fsupplier-deck-q3.pptx&page=7#page=7",
  );
  await expect(first).toHaveAttribute("aria-current", "true");
  await page.getByRole("article").nth(1).getByRole("link", { name: "The whole file" }).click();
  await expect(preview).toContainText("costs-2024.xlsx");
  await expect(preview.locator("td")).toHaveText(["Region", "Q3 freight", "North", "4.2"]);
  await preview.getByRole("button", { name: "Close the preview" }).click();
  await expect(preview).toHaveCount(0);
});

test("with no file details built, the filters say how to build them", async ({ page }) => {
  await page.route("**/api/config", (r) =>
    r.fulfill({ json: { agent: false, ready: true, examples: [], formats: ["pdf"], filters: false } }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Filters" }).click();
  await expect(page.getByRole("note")).toContainText("uv run docsearch meta");
  await expect(page.getByLabel("Folder")).toBeDisabled();
});

test("an API error shows its reason", async ({ page }) => {
  await mockApi(page, { status: 400 });
  await page.goto("/");
  await search(page);
  await expect(page.getByRole("alert")).toHaveText("search filters need file metadata");
});

test("the theme follows the system; the one button flips it and keeps the choice", async ({ page }, info) => {
  await mockApi(page);
  await page.goto("/");
  const html = page.locator("html");
  const scheme = info.project.use.colorScheme as string;
  const other = scheme === "dark" ? "light" : "dark";
  await expect(html).toHaveAttribute("data-theme", scheme);
  const bg = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  const systemBg = await bg();
  await page.getByRole("button", { name: `Switch to the ${other} theme` }).click();
  await expect(html).toHaveAttribute("data-theme", other);
  expect(await bg()).not.toBe(systemBg);
  await page.reload();
  await expect(html).toHaveAttribute("data-theme", other);
  await page.getByRole("button", { name: `Switch to the ${scheme} theme` }).click();
  await expect(html).toHaveAttribute("data-theme", scheme);
  expect(await bg()).toBe(systemBg);
  expect(await page.evaluate(() => localStorage.getItem("docsearch-theme"))).toBeNull();
});
