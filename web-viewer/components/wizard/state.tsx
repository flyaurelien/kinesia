"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useReducer,
  type ReactNode,
} from "react";
import { makeInitialSegments, type Segment } from "../../lib/video-timeline";

// The four ordered stages of the processing wizard.
export type WizardStep = "upload" | "detect" | "run";

// One persistent tracked identity surfaced by the streaming detector.
export type DetectTrack = {
  id: number;
  color: string;
  firstFrame: number;
  lastFrame: number;
  frameCount: number;
  repFrame: number;
};

// One detection on a frame: normalized [x, y, w, h] box tagged with its track id,
// plus an optional segmentation polygon (m: [[x,y], ...] normalized).
export type DetectDet = { id: number; b: [number, number, number, number]; m?: number[][] };

export type DetectStatus = "starting" | "loading" | "running" | "completed" | "stopped" | "error";

// The full-video streaming detection session shown in the Detect step.
export type DetectSession = {
  id: string;
  prompt: string;
  status: DetectStatus;
  processed: number;
  totalToProcess: number;
  totalFrames: number;
  lastFrame: number; // highest processed frame index — clamps preview playback
  videoW: number;
  videoH: number;
  fps: number;
  stride: number;
  tracks: DetectTrack[];
  framesById: Record<number, DetectDet[]>; // processed frame index -> detections
  // Identity grouping: each raw track id maps to a subject id. Default = each
  // track is its own subject; merging maps several tracks onto one subject id
  // (the smallest of the merged set), so re-ID splits can be fixed by hand.
  subjectOf: Record<number, number>;
  // Subject ids the user selected to reconstruct (multi-select).
  selectedSubjects: number[];
  error: string | null;
};

// Full wizard state, grouped by the step that owns each field.
export type WizardState = {
  step: WizardStep;
  file: File | null;
  fileUrl: string | null;
  videoDurationSec: number;
  // Timeline state: `segments` carries the keep/mask/delete model forwarded to the
  // backend; trim bounds + undo history back the in-step scrub timeline.
  trimStartSec: number;
  trimEndSec: number;
  segments: Segment[];
  cutHistory: CutHistoryEntry[];
  cutHistoryIndex: number;
  // Detection step
  subjectPrompt: string;
  // Staged once here, reused by the run so the video uploads a single time.
  stagedUpload: { stagedUploadId: string; fileName: string } | null;
  detect: DetectSession | null;
  // Path (server-side) of the chosen subject's dense per-frame box track, fed to
  // the run via --subject-track-file.
  detectTrackFile: string | null;
  // Run step
  activeJobRunId: string | null;
};

// One undo/redo snapshot of the cut-step segment list.
type CutHistoryEntry = {
  segments: Segment[];
};

// Canonical step sequence; drives go_next/go_back and gating.
const STEP_ORDER: WizardStep[] = ["upload", "detect", "run"];

// Human-readable labels for each step, shown in the wizard UI.
export const STEP_LABELS: Record<WizardStep, string> = {
  upload: "Upload",
  detect: "Detect subject",
  run: "Run & monitor",
};

// Every state transition the reducer understands.
type Action =
  | { type: "set_file"; file: File; url: string }
  | { type: "reset_file" }
  | { type: "set_duration"; durationSec: number }
  | { type: "set_step"; step: WizardStep }
  | { type: "go_next" }
  | { type: "go_back" }
  | { type: "set_trim_start"; sec: number }
  | { type: "set_trim_end"; sec: number }
  | { type: "set_segments"; segments: Segment[] }
  | { type: "reset_edits" }
  | { type: "commit_history" }
  | { type: "undo" }
  | { type: "redo" }
  | { type: "set_subject_prompt"; prompt: string }
  | { type: "set_staged_upload"; staged: { stagedUploadId: string; fileName: string } | null }
  | { type: "detect_start"; id: string; prompt: string; stride: number }
  | {
      type: "detect_progress";
      patch: Partial<Omit<DetectSession, "framesById" | "subjectOf" | "selectedSubjects">>;
      frames?: DetectDet2[];
    }
  | { type: "detect_toggle_subject"; subjectId: number } // select/deselect for reconstruction
  | { type: "detect_merge_selected" } // merge the selected subjects into one
  | { type: "detect_unmerge_subject"; subjectId: number } // split a merged subject back to its tracks
  | { type: "set_detect_track_file"; path: string | null }
  | { type: "detect_reset" }
  | { type: "set_active_run_id"; runId: string | null };

// One streamed frame line merged into the session (frame index + its dets).
export type DetectDet2 = { f: number; dets: DetectDet[] };

// Starting state: fresh wizard at the upload step with no file loaded.
const INITIAL_STATE: WizardState = {
  step: "upload",
  file: null,
  fileUrl: null,
  videoDurationSec: 0,
  trimStartSec: 0,
  trimEndSec: 0,
  segments: [],
  cutHistory: [{ segments: [] }],
  cutHistoryIndex: 0,
  subjectPrompt: "person",
  stagedUpload: null,
  detect: null,
  detectTrackFile: null,
  activeJobRunId: null,
};

// Deep-copy the current segments into a standalone history entry.
function snapshot(state: WizardState): CutHistoryEntry {
  return { segments: state.segments.map((s) => ({ ...s })) };
}

// Pure reducer mapping (state, action) to the next wizard state.
function reducer(state: WizardState, action: Action): WizardState {
  switch (action.type) {
    case "set_file":
      return {
        ...state,
        file: action.file,
        fileUrl: action.url,
        step: "detect",
        // reset edits when a new file is loaded
        trimStartSec: 0,
        trimEndSec: 0,
        segments: [],
        cutHistory: [{ segments: [] }],
        cutHistoryIndex: 0,
        videoDurationSec: 0,
        stagedUpload: null,
        detect: null,
        detectTrackFile: null,
      };
    case "reset_file":
      return { ...INITIAL_STATE };
    case "set_duration": {
      const durationSec = action.durationSec;
      if (durationSec <= 0 || !Number.isFinite(durationSec)) {
        return state;
      }
      // Initialize the segment list the first time we learn the duration.
      // If segments already exist (came back to this step), keep them — and
      // preserve the cut undo/redo history, which is only seeded on a fresh
      // load. A set_duration re-fired on remount must not wipe accumulated edits.
      const isFresh = state.segments.length === 0;
      const segments = isFresh ? makeInitialSegments(durationSec) : state.segments;
      return {
        ...state,
        videoDurationSec: durationSec,
        trimEndSec: state.trimEndSec > 0 ? state.trimEndSec : durationSec,
        segments,
        ...(isFresh
          ? { cutHistory: [{ segments: segments.map((s) => ({ ...s })) }], cutHistoryIndex: 0 }
          : {}),
      };
    }
    case "set_step":
      return { ...state, step: action.step };
    case "go_next": {
      const idx = STEP_ORDER.indexOf(state.step);
      const next = STEP_ORDER[Math.min(idx + 1, STEP_ORDER.length - 1)];
      return { ...state, step: next };
    }
    case "go_back": {
      const idx = STEP_ORDER.indexOf(state.step);
      const prev = STEP_ORDER[Math.max(0, idx - 1)];
      return { ...state, step: prev };
    }
    case "set_trim_start":
      return { ...state, trimStartSec: action.sec };
    case "set_trim_end":
      return { ...state, trimEndSec: action.sec };
    case "set_segments":
      return { ...state, segments: action.segments };
    case "reset_edits":
      return {
        ...state,
        trimStartSec: 0,
        trimEndSec: state.videoDurationSec,
        segments: makeInitialSegments(state.videoDurationSec),
      };
    case "commit_history": {
      const snap = snapshot(state);
      const truncated = state.cutHistory.slice(0, state.cutHistoryIndex + 1);
      const last = truncated[truncated.length - 1];
      if (last && historiesEqual(last, snap)) {
        return state;
      }
      const history = [...truncated, snap].slice(-50);
      return {
        ...state,
        cutHistory: history,
        cutHistoryIndex: history.length - 1,
      };
    }
    case "undo": {
      if (state.cutHistoryIndex <= 0) return state;
      const nextIndex = state.cutHistoryIndex - 1;
      const entry = state.cutHistory[nextIndex];
      return {
        ...state,
        ...entry,
        cutHistoryIndex: nextIndex,
      };
    }
    case "redo": {
      if (state.cutHistoryIndex >= state.cutHistory.length - 1) return state;
      const nextIndex = state.cutHistoryIndex + 1;
      const entry = state.cutHistory[nextIndex];
      return {
        ...state,
        ...entry,
        cutHistoryIndex: nextIndex,
      };
    }
    case "set_subject_prompt":
      return { ...state, subjectPrompt: action.prompt };
    case "set_staged_upload":
      return { ...state, stagedUpload: action.staged };
    case "detect_start":
      return {
        ...state,
        detectTrackFile: null,
        detect: {
          id: action.id,
          prompt: action.prompt,
          status: "starting",
          processed: 0,
          totalToProcess: 0,
          totalFrames: 0,
          lastFrame: -1,
          videoW: 0,
          videoH: 0,
          fps: 30,
          stride: action.stride,
          tracks: [],
          framesById: {},
          subjectOf: {},
          selectedSubjects: [],
          error: null,
        },
      };
    case "detect_progress": {
      if (!state.detect) return state;
      const framesById = { ...state.detect.framesById };
      for (const line of action.frames ?? []) {
        framesById[line.f] = line.dets;
      }
      const patch = action.patch;
      // Reconcile identity grouping: every surfaced track id needs a subject;
      // new tracks default to their own subject, existing merges are preserved.
      const subjectOf = { ...state.detect.subjectOf };
      const liveTracks = patch.tracks ?? state.detect.tracks;
      for (const t of liveTracks) {
        if (subjectOf[t.id] === undefined) subjectOf[t.id] = t.id;
      }
      // Drop selections whose subject no longer has any live track.
      const liveSubjectIds = new Set(liveTracks.map((t) => subjectOf[t.id]));
      const selectedSubjects = state.detect.selectedSubjects.filter((s) => liveSubjectIds.has(s));
      return {
        ...state,
        detect: { ...state.detect, ...patch, framesById, subjectOf, selectedSubjects },
      };
    }
    case "detect_toggle_subject": {
      if (!state.detect) return state;
      const sel = state.detect.selectedSubjects;
      const next = sel.includes(action.subjectId)
        ? sel.filter((s) => s !== action.subjectId)
        : [...sel, action.subjectId];
      return { ...state, detect: { ...state.detect, selectedSubjects: next }, detectTrackFile: null };
    }
    case "detect_merge_selected": {
      if (!state.detect || state.detect.selectedSubjects.length < 2) return state;
      const target = Math.min(...state.detect.selectedSubjects);
      const mergeSet = new Set(state.detect.selectedSubjects);
      const subjectOf = { ...state.detect.subjectOf };
      for (const [tid, sid] of Object.entries(subjectOf)) {
        if (mergeSet.has(sid)) subjectOf[Number(tid)] = target;
      }
      return {
        ...state,
        detect: { ...state.detect, subjectOf, selectedSubjects: [target] },
        detectTrackFile: null,
      };
    }
    case "detect_unmerge_subject": {
      if (!state.detect) return state;
      const subjectOf = { ...state.detect.subjectOf };
      for (const [tid, sid] of Object.entries(subjectOf)) {
        if (sid === action.subjectId) subjectOf[Number(tid)] = Number(tid); // back to singletons
      }
      return {
        ...state,
        detect: { ...state.detect, subjectOf, selectedSubjects: [] },
        detectTrackFile: null,
      };
    }
    case "set_detect_track_file":
      return { ...state, detectTrackFile: action.path };
    case "detect_reset":
      return { ...state, detect: null, detectTrackFile: null };
    case "set_active_run_id":
      return { ...state, activeJobRunId: action.runId };
    default:
      return state;
  }
}

// True when two history entries hold the same segments (with float tolerance),
// so commit_history can skip recording a no-op edit.
function historiesEqual(a: CutHistoryEntry, b: CutHistoryEntry): boolean {
  if (a.segments.length !== b.segments.length) return false;
  for (let i = 0; i < a.segments.length; i += 1) {
    const x = a.segments[i];
    const y = b.segments[i];
    if (x.mode !== y.mode) return false;
    if (Math.abs(x.startSec - y.startSec) > 0.0001) return false;
    if (Math.abs(x.endSec - y.endSec) > 0.0001) return false;
  }
  return true;
}

// Wizard state plus the navigation helpers exposed to consumers via context.
type WizardContextValue = {
  state: WizardState;
  dispatch: React.Dispatch<Action>;
  goNext: () => void;
  goBack: () => void;
  goTo: (step: WizardStep) => void;
  canGoTo: (step: WizardStep) => boolean;
};

const WizardContext = createContext<WizardContextValue | null>(null);

// Owns the wizard reducer and provides state + navigation helpers to children.
export function WizardProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);

  // Whether a step is reachable: earlier steps are always open; later steps are
  // gated on having a file and a known video duration.
  const canGoTo = useCallback(
    (step: WizardStep): boolean => {
      const targetIdx = STEP_ORDER.indexOf(step);
      const currentIdx = STEP_ORDER.indexOf(state.step);
      if (targetIdx <= currentIdx) return true;
      // gate: cut needs file, detect needs cut done, etc.
      if (targetIdx >= 1 && !state.file) return false;
      if (targetIdx >= 2 && state.videoDurationSec <= 0) return false;
      return true;
    },
    [state.file, state.step, state.videoDurationSec],
  );

  const goNext = useCallback(() => dispatch({ type: "go_next" }), []);
  const goBack = useCallback(() => dispatch({ type: "go_back" }), []);
  const goTo = useCallback(
    (step: WizardStep) => dispatch({ type: "set_step", step }),
    [],
  );

  const value = useMemo<WizardContextValue>(
    () => ({ state, dispatch, goNext, goBack, goTo, canGoTo }),
    [canGoTo, goBack, goNext, goTo, state],
  );

  return <WizardContext.Provider value={value}>{children}</WizardContext.Provider>;
}

// Access the wizard context; throws if used outside <WizardProvider>.
export function useWizard(): WizardContextValue {
  const ctx = useContext(WizardContext);
  if (!ctx) {
    throw new Error("useWizard must be used inside <WizardProvider>");
  }
  return ctx;
}

/* App-level actions wizard steps can trigger (e.g. "open this finished run in the
 * viewer"). Provided by the host (wizard-shell / viewer-shell) via WizardActionsProvider. */
export type WizardActions = {
  onViewResults?: (runId: string) => void;
};

const WizardActionsContext = createContext<WizardActions>({});

// Supplies the app-level action callbacks down to the wizard steps.
export function WizardActionsProvider({
  actions,
  children,
}: {
  actions: WizardActions;
  children: ReactNode;
}) {
  return (
    <WizardActionsContext.Provider value={actions}>{children}</WizardActionsContext.Provider>
  );
}

// Access the app-level wizard actions (empty object when no provider is present).
export function useWizardActions(): WizardActions {
  return useContext(WizardActionsContext);
}

export { STEP_ORDER };
