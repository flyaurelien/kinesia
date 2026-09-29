"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { duration, since } from "@/lib/format";
import { isActive, type RunSummary } from "@/lib/runs";
import { describe } from "@/lib/status";
import { Plus, User, Video } from "../icons";
import { Topbar } from "../topbar";
import { NewAnalysis } from "./new-analysis";

function RunCard({ run }: { run: RunSummary }) {
  const status = describe(run);
  const active = isActive(run.status.state) && run.worker_alive;
  return (
    <Link href={`/runs/${run.id}`} className="card run-card">
      <div className="run-thumb">
        {run.has_poster ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={`/api/runs/${run.id}/files/poster.jpg`} alt="" loading="lazy" />
        ) : (
          <Video size={28} />
        )}
        <span className={`chip ${status.tone} run-status`}>
          <span className={`dot${active ? " pulse" : ""}`} />
          {status.label}
        </span>
      </div>
      <div className="run-meta">
        <div className="run-name">{run.name}</div>
        <div className="run-facts faint">
          <span>{duration(run.video?.duration)}</span>
          {run.people != null && (
            <span className="run-people">
              <User size={13} /> {run.people}
            </span>
          )}
          <span className="run-when">{since(run.created_at)}</span>
        </div>
      </div>
    </Link>
  );
}

export function Library() {
  const router = useRouter();
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const body = (await fetch("/api/runs", { cache: "no-store" }).then((r) => r.json())) as { runs: RunSummary[] };
        if (!alive) return;
        setRuns(body.runs);
        const busy = body.runs.some((r) => isActive(r.status.state));
        timer = setTimeout(load, busy ? 3000 : 15000);
      } catch {
        timer = setTimeout(load, 5000);
      }
    };
    load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);

  return (
    <>
      <Topbar />
      <main className="page">
        <div className="page-header">
          <div>
            <h1 className="page-title">Analyses</h1>
            <p className="page-subtitle">Every person in a video, tracked and reconstructed in 3D.</p>
          </div>
          <button className="btn btn-primary" onClick={() => setCreating(true)}>
            <Plus /> New analysis
          </button>
        </div>

        {runs === null ? (
          <div className="run-grid">
            {[0, 1, 2].map((i) => (
              <div key={i} className="card run-card skeleton" />
            ))}
          </div>
        ) : runs.length === 0 ? (
          <div className="card empty">
            <Video size={28} />
            <h2>No analyses yet</h2>
            <p className="muted">Upload a video of people moving: a match, a training session, a dance.</p>
            <button className="btn btn-primary" onClick={() => setCreating(true)}>
              <Plus /> New analysis
            </button>
          </div>
        ) : (
          <div className="run-grid">
            {runs.map((run) => (
              <RunCard key={run.id} run={run} />
            ))}
          </div>
        )}
      </main>
      {creating && <NewAnalysis onClose={() => setCreating(false)} onCreated={(id) => router.push(`/runs/${id}`)} />}
    </>
  );
}
