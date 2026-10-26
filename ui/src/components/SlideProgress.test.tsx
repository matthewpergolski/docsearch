import { afterEach, expect, test } from "bun:test";
import { cleanup, render } from "@testing-library/react";
import { SlideProgress } from "./SlideProgress";

afterEach(cleanup);

const pages = ["a", "b", "c"].map((d, i) => ({
  unit: `s/${d}_page_${i + 1}#p1`,
  file: `s/${d}_page_${i + 1}`,
  page: 1,
}));

test("it counts the slides read and shows each slide's picture", () => {
  const { container, getByRole } = render(<SlideProgress reading={{ pages, done: [pages[0].unit] }} />);
  expect(getByRole("region", { name: "Reading the slides" }).textContent).toContain("Reading the slides: 1 of 3");
  expect([...container.querySelectorAll("img")].map((i) => i.getAttribute("src"))).toEqual([
    "/api/page?file=s%2Fa_page_1&page=1",
    "/api/page?file=s%2Fb_page_2&page=1",
    "/api/page?file=s%2Fc_page_3&page=1",
  ]);
  expect(container.querySelectorAll(".ghost-sheet").length).toBe(1);
});
