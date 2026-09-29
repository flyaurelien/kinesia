import { readRun } from "@/lib/server/store";
import { startProcessing } from "@/lib/server/worker";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Start, resume or retry an analysis (the Python side is idempotent). */
export async function POST(_request: Request, { params }: { params: { id: string } }) {
  const run = readRun(params.id);
  if (!run) return Response.json({ error: "Not found" }, { status: 404 });
  if (run.worker_alive) return Response.json(run);
  startProcessing(params.id);
  return Response.json(readRun(params.id), { status: 202 });
}
