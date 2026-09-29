"use client";

import Link from "next/link";
import { useEffect, useState, type ReactNode } from "react";

type ClusterState = { ok: boolean; user?: string; message: string };

export function ClusterBadge() {
  const [state, setState] = useState<ClusterState | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () =>
      fetch("/api/cluster")
        .then((r) => r.json())
        .then((value: ClusterState) => alive && setState(value))
        .catch(() => alive && setState({ ok: false, message: "Cluster status unavailable" }));
    load();
    const timer = setInterval(load, 60_000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);
  if (!state) {
    return (
      <span className="chip">
        <span className="dot pulse" /> Checking cluster
      </span>
    );
  }
  return (
    <span className={`chip ${state.ok ? "ok" : "warn"}`} title={state.message}>
      <span className="dot" />
      {state.ok ? "remote GPU server connected" : "Cluster offline"}
    </span>
  );
}

export function Topbar({ children }: { children?: ReactNode }) {
  return (
    <header className="topbar">
      <Link href="/" className="brand">
        <span className="brand-mark">K</span>
        Kinesia
      </Link>
      {children}
      <div className="topbar-spacer" />
      <ClusterBadge />
    </header>
  );
}
