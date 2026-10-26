export function NoConfidentBanner() {
  return (
    <p role="status" className="border-warn border-l-2 pl-4 text-[0.9375rem] text-muted">
      <strong className="font-medium text-warn">No confident match.</strong> These are the nearest files; check them
      before you rely on them.
    </p>
  );
}
