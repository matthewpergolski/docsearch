import { afterEach, expect, test } from "bun:test";
import { cleanup, render } from "@testing-library/react";
import { RESPONSE } from "../test/fixtures";
import { ResultCard } from "./ResultCard";

afterEach(cleanup);

test("a file shows its name, format, period and date, three slides, and the rest by number", () => {
  const { container, getByRole } = render(<ResultCard group={RESPONSE.results.groups[0]} terms={RESPONSE.terms} />);
  expect(getByRole("heading").textContent).toBe("Supplier review Q3");
  expect(container.textContent).toContain("PowerPoint");
  expect(container.textContent).toContain("Q3 2024");
  expect(container.textContent).toContain("2 Oct 2024");
  expect(container.querySelectorAll("li").length).toBe(3);
  expect(container.textContent).toContain("Slide 4Freight by region");
  expect(container.textContent).toContain("Also on slide 12");
  expect([...container.querySelectorAll("mark")].map((m) => m.textContent)).toEqual([
    "Freight",
    "region",
    "Region",
    "freight",
  ]);
});

test("the heading opens the file at its best page, and each slide number opens that slide", () => {
  const { getByRole } = render(<ResultCard group={RESPONSE.results.groups[0]} terms={[]} />);
  const deck = "/api/file?file=reports%2Fsupplier-deck-q3.pptx";
  const heading = getByRole("link", { name: "Supplier review Q3" });
  expect(heading.getAttribute("href")).toBe(`${deck}&page=4#page=4`);
  expect(heading.getAttribute("target")).toBe("_blank");
  expect(getByRole("link", { name: "Open slide 7" }).getAttribute("href")).toBe(`${deck}&page=7#page=7`);
  expect(getByRole("link", { name: "Open slide 12" }).getAttribute("href")).toBe(`${deck}&page=12#page=12`);
});

test("a file without pages shows no page number and opens the whole file", () => {
  const { container, getByRole } = render(<ResultCard group={RESPONSE.results.groups[1]} terms={[]} />);
  expect(container.textContent).not.toMatch(/Sheet \d/);
  expect(container.textContent).toContain("costs-2024.xlsx");
  expect(getByRole("link", { name: "costs-2024.xlsx" }).getAttribute("href")).toBe(
    "/api/file?file=finance%2Fcosts-2024.xlsx",
  );
});

test("a spreadsheet passage shows as its table, and the file shows a ruled sheet for want of a page picture", () => {
  const { container } = render(<ResultCard group={RESPONSE.results.groups[1]} terms={["freight"]} />);
  expect([...container.querySelectorAll("td")].map((td) => td.textContent)).toEqual([
    "Region",
    "Q3 freight",
    "Q2 freight",
  ]);
  expect(container.querySelector("img")).toBeNull();
  expect(container.querySelector(".sheet-grid")).not.toBeNull();
});
