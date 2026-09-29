import { useEffect, useState } from "react";

import { PersonTrack, meshData, type MeshData } from "./tracks";
import type { Edits, MetricsFile, OverlayFile, SceneFile } from "./types";

export type LoadedScene = {
  scene: SceneFile;
  mesh: MeshData;
  people: PersonTrack[];
  overlay: OverlayFile | null;
  metrics: MetricsFile | null;
  edits: Edits;
};

async function fetchBuffer(url: string, onBytes: (n: number) => void): Promise<ArrayBuffer> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok || !response.body) throw new Error(`${url}: ${response.status}`);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    size += value.byteLength;
    onBytes(value.byteLength);
  }
  const out = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}

async function fetchJson<T>(url: string): Promise<T | null> {
  const response = await fetch(url, { cache: "no-store" });
  return response.ok ? ((await response.json()) as T) : null;
}

export async function loadScene(runId: string, onProgress: (loadedBytes: number) => void): Promise<LoadedScene> {
  const base = `/api/runs/${runId}/files/scene`;
  const scene = await fetchJson<SceneFile>(`${base}/scene.json`);
  if (!scene) throw new Error("The 3D scene of this analysis is missing.");
  let loaded = 0;
  const bump = (n: number) => {
    loaded += n;
    onProgress(loaded);
  };
  const [meshBuffer, peopleBuffer, overlay, metrics, edits] = await Promise.all([
    fetchBuffer(`${base}/${scene.mesh.file}`, bump),
    fetchBuffer(`${base}/${scene.people_file}`, bump),
    fetchJson<OverlayFile>(`${base}/overlay.json`),
    fetchJson<MetricsFile>(`${base}/metrics.json`),
    fetchJson<Edits>(`/api/runs/${runId}/edits`),
  ]);
  const mesh = meshData(scene, meshBuffer);
  const people = scene.people.map((entry) => new PersonTrack(entry, peopleBuffer, scene, mesh.parents));
  return { scene, mesh, people, overlay, metrics, edits: edits ?? {} };
}

export function useScene(runId: string) {
  const [state, setState] = useState<{ data: LoadedScene | null; error: string | null; bytes: number }>({
    data: null,
    error: null,
    bytes: 0,
  });
  useEffect(() => {
    let alive = true;
    setState({ data: null, error: null, bytes: 0 });
    loadScene(runId, (bytes) => alive && setState((s) => ({ ...s, bytes })))
      .then((data) => alive && setState({ data, error: null, bytes: 0 }))
      .catch((error) => alive && setState({ data: null, error: String(error?.message ?? error), bytes: 0 }));
    return () => {
      alive = false;
    };
  }, [runId]);
  return state;
}
