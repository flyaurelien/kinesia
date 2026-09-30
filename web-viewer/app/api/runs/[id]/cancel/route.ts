import { readRun } from "@/lib/server/store";
import { CANCEL_TIMEOUT_MS, runCli } from "@/lib/server/worker";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Stop an analysis: `kinesia cancel` stops its worker and its GPU work. */
export async function POST(_request: Request, { params }: { params: { id: string } }) {
  const run = readRun(params.id);
  if (!run) return Response.json({ error: "Not found" }, { status: 404 });
  const result = await runCli(["cancel", params.id], CANCEL_TIMEOUT_MS);
  if (result.code !== 0) {
    return Response.json({ error: `Could not cancel the analysis:\n${result.output.slice(-800)}` }, { status: 502 });
  }
  return Response.json(readRun(params.id));
}
