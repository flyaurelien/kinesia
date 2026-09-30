import { runCli } from "@/lib/server/worker";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export type GpuState = { ok: boolean; label: string; detail: string; checked_at: string };

let cache: { at: number; value: GpuState } | null = null;
const TTL_MS = 60_000;

/** Whether analyses can run: `kinesia doctor` (GPU and model files), cached for a minute. */
export async function GET(request: Request) {
  const force = new URL(request.url).searchParams.has("refresh");
  if (cache && !force && Date.now() - cache.at < TTL_MS) return Response.json(cache.value);
  const result = await runCli(["doctor", "--json"], 45_000);
  let value: GpuState;
  try {
    const report = JSON.parse(result.output.trim().split("\n").filter(Boolean).pop() ?? "{}");
    value = { ok: Boolean(report.ok), label: String(report.label ?? "GPU"), detail: String(report.detail ?? ""), checked_at: new Date().toISOString() };
  } catch {
    value = { ok: false, label: "GPU check failed", detail: result.output.slice(-400), checked_at: new Date().toISOString() };
  }
  cache = { at: Date.now(), value };
  return Response.json(value);
}
