"use client";

import { useCallback, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { useWizard } from "../state";

const ALLOWED_EXTENSIONS = new Set(["mp4", "mov", "m4v", "avi", "mkv", "webm"]);
const MAX_BYTES = 8 * 1024 * 1024 * 1024; // 8 GB upper UI guard

// Return the lowercased file extension (without the dot), or "" if there is none.
function getExtension(name: string): string {
  const idx = name.lastIndexOf(".");
  return idx === -1 ? "" : name.slice(idx + 1).toLowerCase();
}

// First wizard step: pick a video by drag-drop or file browse, validate it locally,
// and hand the file (plus an object URL for preview) to the wizard state.
export function UploadStep() {
  const { dispatch } = useWizard();
  const [error, setError] = useState<string | null>(null);
  const [isHover, setIsHover] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // Validate a chosen file (extension + size guard) and, if OK, store it in wizard state.
  const acceptFile = useCallback(
    (file: File) => {
      const ext = getExtension(file.name);
      if (!ALLOWED_EXTENSIONS.has(ext)) {
        setError(`Unsupported format: .${ext || "?"}. Use .mp4 / .mov / .m4v / .avi / .mkv / .webm`);
        return;
      }
      if (file.size > MAX_BYTES) {
        setError(`File too large (${(file.size / 1e9).toFixed(2)} GB). Max 8 GB.`);
        return;
      }
      const url = URL.createObjectURL(file);
      setError(null);
      dispatch({ type: "set_file", file, url });
    },
    [dispatch],
  );

  // Handle a file picked via the hidden <input> (the click-to-browse path).
  const handleInput = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) acceptFile(file);
      // reset so the same file can be selected again later
      event.target.value = "";
    },
    [acceptFile],
  );

  // Handle a file dropped onto the drop zone (the drag-and-drop path).
  const handleDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsHover(false);
      const file = event.dataTransfer.files?.[0];
      if (file) acceptFile(file);
    },
    [acceptFile],
  );

  // Keep the drop zone highlighted while a file is dragged over it (preventDefault enables dropping).
  const handleDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsHover(true);
  }, []);

  // Clear the hover highlight once the drag leaves the drop zone.
  const handleDragLeave = useCallback(() => setIsHover(false), []);

  return (
    <div>
      <div className="of-step-header">
        <h2 className="of-step-title">Upload a video</h2>
        <p className="of-step-subtitle">
          The video stays in your browser until you launch the run. Common camera formats are supported.
        </p>
      </div>
      {error ? <div className="of-banner">{error}</div> : null}
      <div
        className={`of-upload-drop ${isHover ? "is-hover" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <div className="of-upload-icon" aria-hidden="true">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 3v12" />
            <path d="m7 8 5-5 5 5" />
            <path d="M5 21h14" />
          </svg>
        </div>
        <div>
          <p className="of-upload-title">Drop a video here or click to browse</p>
          <p className="of-upload-hint">A single file at a time. You will trim and review before processing.</p>
        </div>
        <div className="of-upload-formats">
          <span className="of-format-chip">mp4</span>
          <span className="of-format-chip">mov</span>
          <span className="of-format-chip">m4v</span>
          <span className="of-format-chip">avi</span>
          <span className="of-format-chip">mkv</span>
          <span className="of-format-chip">webm</span>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          hidden
          onChange={handleInput}
        />
      </div>
    </div>
  );
}
