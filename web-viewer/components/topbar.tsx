"use client";

import Link from "next/link";
import { useEffect, useState, type ReactNode } from "react";

type GpuState = { ok: boolean; label: string; detail: string };

/** Whether analyses can run here: the GPU (or the configured runner) and the model files. */
export function GpuBadge() {
  const [state, setState] = useState<GpuState | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () =>
      fetch("/api/gpu")
        .then((r) => r.json())
        .then((value: GpuState) => alive && setState(value))
        .catch(() => alive && setState({ ok: false, label: "GPU status unavailable", detail: "" }));
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
        <span className="dot pulse" /> Checking the GPU
      </span>
    );
  }
  return (
    <span className={`chip ${state.ok ? "ok" : "warn"}`} title={state.detail}>
      <span className="dot" />
      {state.label}
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
      <GpuBadge />
    </header>
  );
}
