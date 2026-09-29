import fs from "node:fs";

import { runFile } from "@/lib/server/paths";
import type { Edits } from "@/lib/scene/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type Context = { params: { id: string } };

function read(id: string): Edits {
  try {
    return JSON.parse(fs.readFileSync(runFile(id, "edits.json"), "utf8")) as Edits;
  } catch {
    return {};
  }
}

export async function GET(_request: Request, { params }: Context) {
  return Response.json(read(params.id));
}

/** Save the user's corrections (person names, hidden people). */
export async function PUT(request: Request, { params }: Context) {
  const body = (await request.json().catch(() => null)) as Edits | null;
  if (!body || typeof body !== "object") return Response.json({ error: "Invalid edits." }, { status: 400 });
  const edits: Edits = {
    labels: Object.fromEntries(
      Object.entries(body.labels ?? {})
        .filter(([key, value]) => /^\d+$/.test(key) && typeof value === "string")
        .map(([key, value]) => [key, value.trim().slice(0, 60)]),
    ),
    hidden: (body.hidden ?? []).filter((value) => Number.isInteger(value)),
  };
  fs.writeFileSync(runFile(params.id, "edits.json"), JSON.stringify(edits, null, 1));
  return Response.json(edits);
}
