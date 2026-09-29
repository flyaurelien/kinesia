"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { isActive, type RunSummary } from "@/lib/runs";
import { Back } from "../icons";
import { Topbar } from "../topbar";
import { Processing } from "./processing";

const Viewer = dynamic(() => import("../viewer/viewer").then((m) => m.Viewer), {
  ssr: false,
  loading: () => <div className="viewer-loading">Loading the 3D viewer…</div>,
});

export function RunView({ id }: { id: string }) {
  const router = useRouter();
  const [run, setRun] = useState<RunSummary | null>(null);
  const [missing, setMissing] = useState(false);

  const load = useCallback(async () => {
    const response = await fetch(`/api/runs/${id}`, { cache: "no-store" });
    if (response.status === 404) {
      setMissing(true);
      return null;
    }
    const body = (await response.json()) as RunSummary;
    setRun(body);
    return body;
  }, [id]);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      const current = await load().catch(() => null);
      if (!alive) return;
      const busy = current ? isActive(current.status.state) || !current.has_scene : true;
      if (busy) timer = setTimeout(tick, 2000);
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [load]);

  const act = useCallback(
    async (action: "process" | "cancel" | "delete") => {
      if (action === "delete") {
        if (!window.confirm("Delete this analysis and its files? This cannot be undone.")) return;
        const response = await fetch(`/api/runs/${id}`, { method: "DELETE" });
        if (response.ok) router.push("/");
        else window.alert((await response.json()).error ?? "Could not delete");
        return;
      }
      const response = await fetch(`/api/runs/${id}/${action}`, { method: "POST" });
      if (!response.ok) window.alert((await response.json().catch(() => ({}))).error ?? `Could not ${action}`);
      await load();
    },
    [id, load, router],
  );

  if (missing) {
    return (
      <>
        <Topbar />
        <main className="page">
          <div className="card empty">
            <h2>Analysis not found</h2>
            <Link href="/" className="btn">
              <Back /> Back to analyses
            </Link>
          </div>
        </main>
      </>
    );
  }

  if (run && run.has_scene && run.status.state === "done") {
    return <Viewer run={run} onRenamed={load} onDelete={() => act("delete")} />;
  }

  return (
    <>
      <Topbar>
        <Link href="/" className="crumb">
          <Back size={14} /> Analyses
        </Link>
        {run && <span className="crumb-current">{run.name}</span>}
      </Topbar>
      <main className="page">{run ? <Processing run={run} onAction={act} /> : <div className="viewer-loading">Loading…</div>}</main>
    </>
  );
}
