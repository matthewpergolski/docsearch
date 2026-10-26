import { fileUrl } from "../lib/api";
import { cite } from "../lib/labels";
import type { Answer } from "../types";

export function AgentAnswerCard({ answer, format = "" }: { answer: Answer; format?: string }) {
  return (
    <section aria-label="Answer" className="grid gap-2 pb-2">
      <p className="text-[0.8125rem] text-muted">The LLM read this on the page below</p>
      <p className="font-text text-[2rem] leading-tight">
        <mark className="hl">{answer.value || "See the page"}</mark>
      </p>
      <a
        href={fileUrl(answer.file, format, answer.page)}
        target="_blank"
        rel="noreferrer"
        className="w-fit text-[0.875rem] text-muted underline-offset-4 hover:text-fg hover:underline"
      >
        {cite(answer.file, answer.page)}
      </a>
    </section>
  );
}
