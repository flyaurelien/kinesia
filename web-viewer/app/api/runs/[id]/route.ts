import { isActive } from "@/lib/runs";
import { deleteRun, readRun, renameRun } from "@/lib/server/store";
import { CANCEL_TIMEOUT_MS, runCli } from "@/lib/server/worker";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type Context = { params: { id: string } };

export async function GET(_request: Request, { params }: Context) {
  const run = readRun(params.id);
  return run ? Response.json(run) : Response.json({ error: "Not found" }, { status: 404 });
}

export async function PATCH(request: Request, { params }: Context) {
  const body = (await request.json().catch(() => ({}))) as { name?: string };
  const name = body.name?.trim();
  if (!name) return Response.json({ error: "A name is required." }, { status: 400 });
  renameRun(params.id, name.slice(0, 120));
  return Response.json(readRun(params.id));
}

/** Delete an analysis; GPU work that may still be running is cancelled first. */
export async function DELETE(_request: Request, { params }: Context) {
  const run = readRun(params.id);
  if (!run) return Response.json({ error: "Not found" }, { status: 404 });
  const { state, gpu_started_at: started, gpu_finished_at: finished } = run.status;
  // A failure on this side (a lost connection, a crash) can leave GPU work running.
  if (isActive(state) || (state === "failed" && started && !finished)) {
    const result = await runCli(["cancel", params.id], CANCEL_TIMEOUT_MS);
    if (result.code !== 0) {
      return Response.json({ error: `Could not cancel the analysis:\n${result.output.slice(-800)}` }, { status: 502 });
    }
  }
  deleteRun(params.id);
  return new Response(null, { status: 204 });
}
