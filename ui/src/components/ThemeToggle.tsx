import type { Theme } from "../lib/theme";
import { MoonIcon, SunIcon } from "./icons";

/** One button: it shows the theme it switches to. */
export function ThemeToggle({ shown, onFlip }: { shown: Theme; onFlip: () => void }) {
  const to = shown === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      aria-label={`Switch to the ${to} theme`}
      title={`Switch to the ${to} theme`}
      onClick={onFlip}
      className="grid size-9 place-items-center rounded-full text-muted transition-colors hover:bg-fg/5 hover:text-fg"
    >
      {to === "dark" ? <MoonIcon /> : <SunIcon />}
    </button>
  );
}
