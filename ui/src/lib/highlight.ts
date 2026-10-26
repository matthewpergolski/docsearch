export interface Piece {
  text: string;
  hit: boolean;
}

const literal = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// Splits text into pieces, marking each whole-word match of a term (the API's query terms, lower case).
export function highlight(text: string, terms: string[]): Piece[] {
  const words = [...new Set(terms.filter((t) => t.length > 0))].sort((a, b) => b.length - a.length);
  if (!words.length || !text) return text ? [{ text, hit: false }] : [];
  const re = new RegExp(`(?<![\\p{L}\\p{N}])(${words.map(literal).join("|")})(?![\\p{L}\\p{N}])`, "giu");
  const out: Piece[] = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    const at = m.index ?? 0;
    if (at > last) out.push({ text: text.slice(last, at), hit: false });
    out.push({ text: m[0], hit: true });
    last = at + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last), hit: false });
  return out;
}
