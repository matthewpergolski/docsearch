import { highlight } from "../lib/highlight";

export function Snippet({ text, terms }: { text: string; terms: string[] }) {
  return (
    <span>
      {highlight(text, terms).map((p, i) =>
        p.hit ? (
          // biome-ignore lint/suspicious/noArrayIndexKey: pieces have no identity beyond their place
          <mark key={i} className="hl">
            {p.text}
          </mark>
        ) : (
          // biome-ignore lint/suspicious/noArrayIndexKey: pieces have no identity beyond their place
          <span key={i}>{p.text}</span>
        ),
      )}
    </span>
  );
}
