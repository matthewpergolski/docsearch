import { expect, test } from "bun:test";
import { tableRows } from "./table";

test("a table snippet's rows and cells", () => {
  expect(tableRows("Barista Annual Salary | £50,000 | / Cost inflation | 0.05 | *")).toEqual({
    before: false,
    after: false,
    rows: [
      ["Barista Annual Salary", "£50,000"],
      ["Cost inflation", "0.05", "*"],
    ],
  });
});

test("a cut snippet keeps its empty cells and says where it was cut", () => {
  expect(tableRows("…Counter | | 1 / Salaries | | £50,000 |…")).toEqual({
    before: true,
    after: true,
    rows: [
      ["Counter", "", "1"],
      ["Salaries", "", "£50,000"],
    ],
  });
});

test("text without cells is not a table", () => {
  expect(tableRows("Freight cost per region rose 4% / in the north.")).toBeNull();
  expect(tableRows("Region | Q3 freight | Q2 freight")).toEqual({
    before: false,
    after: false,
    rows: [["Region", "Q3 freight", "Q2 freight"]],
  });
});
