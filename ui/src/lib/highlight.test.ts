import { expect, test } from "bun:test";
import { highlight } from "./highlight";

test("marks whole-word matches of the terms, any case", () => {
  expect(highlight("Wages rose; wage growth in 2023 and 20234", ["wages", "2023"])).toEqual([
    { text: "Wages", hit: true },
    { text: " rose; wage growth in ", hit: false },
    { text: "2023", hit: true },
    { text: " and 20234", hit: false },
  ]);
});

test("no terms or no text gives the text as it is", () => {
  expect(highlight("plain", [])).toEqual([{ text: "plain", hit: false }]);
  expect(highlight("", ["x"])).toEqual([]);
});

test("terms with regex characters are matched literally", () => {
  expect(highlight("rate 32.7% vs 3207", ["32.7"])).toEqual([
    { text: "rate ", hit: false },
    { text: "32.7", hit: true },
    { text: "% vs 3207", hit: false },
  ]);
});
