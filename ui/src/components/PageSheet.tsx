import { useState } from "react";
import { fileUrl, pageUrl } from "../lib/api";
import type { ResultGroup } from "../types";

const WIDE = new Set(["pptx", "slide_image"]);
const GRID = new Set(["xlsx", "csv"]);

/** The picture of the result's best page, linked to the file there; a ruled blank sheet when there is none. The
 * heading carries the same link, so this one is out of the tab order. onClick: on a wide screen, preview instead. */
export function PageSheet({ group, onClick }: { group: ResultGroup; onClick?: (e: React.MouseEvent) => void }) {
  const page = group.pages[0]?.page ?? null;
  const src = pageUrl(group.file, group.format, page);
  const [failed, setFailed] = useState(false);
  const blank = !src || failed;
  const ext = group.file.includes(".") ? group.file.split(".").pop() : group.format;
  return (
    <a
      href={fileUrl(group.file, group.format, page)}
      target="_blank"
      rel="noreferrer"
      tabIndex={-1}
      onClick={onClick}
      aria-label={page === null ? "The whole file" : `Page ${page}`}
      className={`sheet relative block overflow-hidden rounded-[3px] ${WIDE.has(group.format) ? "aspect-video" : "aspect-[17/22]"} ${
        blank ? (GRID.has(group.format) ? "sheet-grid" : "sheet-lines") : "bg-sheet"
      }`}
    >
      {blank ? (
        <span className="absolute inset-0 grid place-items-center text-[0.9375rem] font-light tracking-[0.04em] text-muted">
          .{ext}
        </span>
      ) : (
        <img
          src={src}
          alt=""
          loading="lazy"
          onError={() => setFailed(true)}
          className="size-full object-cover object-top"
        />
      )}
    </a>
  );
}
