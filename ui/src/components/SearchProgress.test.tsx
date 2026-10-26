import { afterEach, expect, test } from "bun:test";
import { cleanup, render } from "@testing-library/react";
import { SearchProgress } from "./SearchProgress";

afterEach(cleanup);

test("while passages are ranked, each file type found shows as a sheet that is being read", () => {
  const { getByRole, container } = render(
    <SearchProgress stage={{ step: "ranking", files: { slide_image: 4, pdf: 2, xlsx: 1 } }} reading={null} />,
  );
  const text = getByRole("region", { name: "Search progress" }).textContent ?? "";
  expect(text).toContain("Looking through your files (done)");
  expect(text).toContain("Ranking the best passages from 7 files");
  expect([...container.querySelectorAll("li li")].map((li) => li.textContent)).toEqual(["PDF 2", "Excel 1", "Slide 4"]);
  expect(container.querySelectorAll(".glyph.scanning").length).toBe(3);
});

test("a search waiting for another says so", () => {
  const { getByRole } = render(<SearchProgress stage={{ step: "waiting", why: "another search" }} reading={null} />);
  expect(getByRole("region", { name: "Search progress" }).textContent).toContain(
    "Waiting for another search to finish",
  );
});
