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

/** Delete an analysis; a job that may still be on the cluster is cancelled first (freeing its GPU). */
export async function DELETE(_request: Request, { params }: Context) {
  const run = readRun(params.id);
  if (!run) return Response.json({ error: "Not found" }, { status: 404 });
  const { state, job } = run.status;
  // A local failure (lost network, failed download) can leave the job running.
  if (isActive(state) || (state === "failed" && job)) {
    const result = await runCli(["cancel", params.id], CANCEL_TIMEOUT_MS);
    if (result.code !== 0) {
      return Response.json({ error: `Could not cancel the cluster job:\n${result.output.slice(-800)}` }, { status: 502 });
    }
  }
  deleteRun(params.id);
  return new Response(null, { status: 204 });
}
