"use client";

import { useRef, useState, type DragEvent } from "react";

import { bytes } from "@/lib/format";
import { Close, Upload } from "../icons";

const POOLS: { value: string; label: string; hint: string }[] = [
  { value: "b300,default", label: "B300, then A100", hint: "Fastest; falls back to the A100 pool when every B300 is busy." },
  { value: "default,b300", label: "A100 first", hint: "Your guaranteed quota; usually starts right away." },
  { value: "h100,h200,default", label: "H100 / H200", hint: "Pre-emptible pools; a pre-empted job resumes where it stopped." },
];

type Props = { onClose: () => void; onCreated: (id: string) => void };

export function NewAnalysis({ onClose, onCreated }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [maxPeople, setMaxPeople] = useState(64);
  const [pools, setPools] = useState(POOLS[0].value);
  const [over, setOver] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  function choose(next: File | undefined) {
    if (!next) return;
    setFile(next);
    setError(null);
    if (!name) setName(next.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " "));
  }

  function drop(event: DragEvent) {
    event.preventDefault();
    setOver(false);
    choose(event.dataTransfer.files?.[0]);
  }

  function start() {
    if (!file) return;
    const query = new URLSearchParams({ filename: file.name, name: name || file.name, max_people: String(maxPeople), pools });
    const request = new XMLHttpRequest();
    request.open("POST", `/api/runs?${query}`);
    request.setRequestHeader("content-type", file.type || "application/octet-stream");
    request.upload.onprogress = (event) => event.lengthComputable && setProgress(event.loaded / event.total);
    request.onload = () => {
      const body = JSON.parse(request.responseText || "{}");
      if (request.status >= 300) {
        setProgress(null);
        setError(body.error ?? `Upload failed (${request.status})`);
      } else {
        onCreated(body.id);
      }
    };
    request.onerror = () => {
      setProgress(null);
      setError("The upload was interrupted.");
    };
    setProgress(0);
    request.send(file);
  }

  const busy = progress !== null;
  return (
    <div className="backdrop" onMouseDown={(e) => e.target === e.currentTarget && !busy && onClose()}>
      <div className="dialog" role="dialog" aria-modal aria-labelledby="new-title">
        <div className="dialog-head" style={{ display: "flex", alignItems: "center" }}>
          <h2 id="new-title" style={{ flex: 1 }}>New analysis</h2>
          <button className="icon-btn" onClick={onClose} disabled={busy} aria-label="Close">
            <Close />
          </button>
        </div>
        <div className="dialog-body">
          <div
            className={`dropzone${over ? " over" : ""}`}
            onClick={() => input.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={drop}
          >
            <input
              ref={input}
              type="file"
              accept="video/*,.mp4,.mov,.m4v,.avi,.mkv,.webm"
              hidden
              onChange={(e) => choose(e.target.files?.[0])}
            />
            <div style={{ display: "grid", placeItems: "center", gap: 8 }}>
              <Upload size={22} />
              {file ? (
                <>
                  <strong>{file.name}</strong>
                  <span className="faint">{bytes(file.size)} · click to choose another</span>
                </>
              ) : (
                <>
                  <strong>Drop a video here, or click to choose</strong>
                  <span className="faint">From a fixed camera (tripod or stand) · MP4, MOV, MKV or WebM · up to about 3 minutes</span>
                </>
              )}
            </div>
          </div>

          <div className="field">
            <label htmlFor="run-name">Name</label>
            <input id="run-name" className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Sunday match" />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1.4fr", gap: 12 }}>
            <div className="field">
              <label htmlFor="max-people">People to track</label>
              <input
                id="max-people"
                className="input"
                type="number"
                min={4}
                max={128}
                value={maxPeople}
                onChange={(e) => setMaxPeople(Math.max(4, Math.min(128, Number(e.target.value) || 64)))}
              />
              <span className="hint">At most, at the same time</span>
            </div>
            <div className="field">
              <label htmlFor="pools">GPU</label>
              <select id="pools" className="input" value={pools} onChange={(e) => setPools(e.target.value)}>
                {POOLS.map((p) => (
                  <option key={p.value} value={p.value}>
                    {p.label}
                  </option>
                ))}
              </select>
              <span className="hint">{POOLS.find((p) => p.value === pools)?.hint}</span>
            </div>
          </div>

          <p className="faint" style={{ margin: 0, fontSize: 12.5 }}>
            Tracking and 3D reconstruction run on one remote GPU server GPU as a batch job. The job ends, and
            the GPU is released, as soon as the results are ready — or immediately if you cancel.
          </p>

          {busy && (
            <div className="progress">
              <div style={{ width: `${Math.round((progress ?? 0) * 100)}%` }} />
            </div>
          )}
          {error && <div className="error-box">{error}</div>}
        </div>
        <div className="dialog-foot">
          <button className="btn" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={start} disabled={!file || busy}>
            {busy ? `Uploading ${Math.round((progress ?? 0) * 100)}%` : "Start analysis"}
          </button>
        </div>
      </div>
    </div>
  );
}
