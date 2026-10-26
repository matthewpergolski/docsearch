import { expect, test } from "bun:test";
import { pageWord } from "./labels";

test("a page is a slide in a deck and a sheet in a workbook", () => {
  expect([pageWord("pdf"), pageWord("docx"), pageWord("pptx"), pageWord("slide_image"), pageWord("xlsx")]).toEqual([
    "page",
    "page",
    "slide",
    "slide",
    "sheet",
  ]);
});
