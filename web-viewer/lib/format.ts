export function clock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

export function duration(seconds: number | undefined | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds - m * 60);
  if (m < 60) return `${m} min ${String(s).padStart(2, "0")} s`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`;
}

export function since(iso: string | undefined, now = Date.now()): string {
  if (!iso) return "";
  const delta = Math.max(0, (now - new Date(iso).getTime()) / 1000);
  if (delta < 45) return "just now";
  if (delta < 3600) return `${Math.round(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)} h ago`;
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

export function elapsed(fromIso: string | undefined | null, toIso?: string | null, now = Date.now()): number | null {
  if (!fromIso) return null;
  const end = toIso ? new Date(toIso).getTime() : now;
  return Math.max(0, (end - new Date(fromIso).getTime()) / 1000);
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function number(value: number | undefined | null, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "–";
  return value.toFixed(digits);
}
