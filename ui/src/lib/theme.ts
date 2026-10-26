export type ThemeChoice = "system" | "light" | "dark";
export type Theme = "light" | "dark";

const KEY = "docsearch-theme";

export const resolveTheme = (choice: ThemeChoice, systemDark: boolean): Theme =>
  choice === "system" ? (systemDark ? "dark" : "light") : choice;

export function loadChoice(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

export function saveChoice(choice: ThemeChoice): void {
  try {
    if (choice === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, choice);
  } catch {
    // storage blocked: the choice lasts until the page reloads
  }
}

/** The one theme button: the other theme, stored only when it differs from the system's. */
export const flip = (shown: Theme, systemDark: boolean): ThemeChoice => {
  const next: Theme = shown === "dark" ? "light" : "dark";
  return next === (systemDark ? "dark" : "light") ? "system" : next;
};
