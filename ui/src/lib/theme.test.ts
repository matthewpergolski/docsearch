import { beforeEach, expect, test } from "bun:test";
import { flip, loadChoice, resolveTheme, saveChoice } from "./theme";

beforeEach(() => localStorage.clear());

test("system follows the system; light and dark are fixed", () => {
  expect(resolveTheme("system", true)).toBe("dark");
  expect(resolveTheme("system", false)).toBe("light");
  expect(resolveTheme("light", true)).toBe("light");
  expect(resolveTheme("dark", false)).toBe("dark");
});

test("the choice is remembered; system is the default and is not stored", () => {
  expect(loadChoice()).toBe("system");
  saveChoice("dark");
  expect(loadChoice()).toBe("dark");
  saveChoice("system");
  expect(localStorage.getItem("docsearch-theme")).toBeNull();
});

test("the one button flips to the other theme, and back to following the system when that matches it", () => {
  expect(flip("light", false)).toBe("dark");
  expect(flip("dark", false)).toBe("system");
  expect(flip("dark", true)).toBe("light");
  expect(flip("light", true)).toBe("system");
});
