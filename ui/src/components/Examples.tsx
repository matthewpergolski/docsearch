export function Examples({ examples, onPick }: { examples: string[]; onPick: (q: string) => void }) {
  if (examples.length === 0) return null;
  return (
    <section aria-label="Searches to try" className="grid gap-3">
      <h2 className="text-[0.8125rem] text-muted">Try one of these</h2>
      <ul className="grid gap-1.5">
        {examples.map((q) => (
          <li key={q}>
            <button
              type="button"
              onClick={() => onPick(q)}
              className="text-left font-text text-[1.0625rem] leading-snug text-fg/75 decoration-1 underline-offset-4 hover:text-fg hover:underline"
            >
              {q}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
