export const FORMAT_LABEL: Record<string, string> = {
  csv: "CSV",
  docx: "Word",
  image: "Image",
  pdf: "PDF",
  pptx: "PowerPoint",
  slide_image: "Slide",
  xlsx: "Excel",
};

const FORMAT_COLOR: Record<string, string> = {
  csv: "var(--fmt-csv)",
  docx: "var(--fmt-docx)",
  image: "var(--fmt-image)",
  pdf: "var(--fmt-pdf)",
  pptx: "var(--fmt-pptx)",
  slide_image: "var(--fmt-slide)",
  xlsx: "var(--fmt-xlsx)",
};

const BY_USE = ["pdf", "docx", "xlsx", "pptx", "slide_image", "csv", "image"];

/** Formats in the order people reach for them; unknown ones last. */
export const byUse = (formats: string[]) =>
  [...formats].sort((a, b) => (BY_USE.indexOf(a) + 1 || 99) - (BY_USE.indexOf(b) + 1 || 99));

export const formatLabel = (f: string) => FORMAT_LABEL[f] ?? f;

export const formatColor = (f: string) => FORMAT_COLOR[f] ?? "var(--muted)";

export const fileName = (path: string) => path.split("/").pop() ?? path;

export const cite = (file: string, page: number | null) =>
  page === null ? fileName(file) : `${fileName(file)}, p. ${page}`;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "2024-10-02" as "2 Oct 2024"; anything else as it is. */
export function longDate(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  return m ? `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]} ${m[1]}` : iso;
}

/** What a page of the format is called: a slide in a deck, a sheet in a workbook. */
export const pageWord = (format: string) =>
  format === "pptx" || format === "slide_image" ? "slide" : format === "xlsx" ? "sheet" : "page";

export const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
