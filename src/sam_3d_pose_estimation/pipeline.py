from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .artifacts import build_run_manifest, write_json
from .sam3d_runtime import (
    add_sam3d_repo_to_path,
    infer_single_person_from_bbox,
    load_estimator,
    patch_sam3d_cuda_assumptions,
    select_device,
    try_build_human_detector,
)


@dataclass
class PipelineConfig:
    """All knobs for a single ``run_pipeline`` invocation (one video)."""

    video_input: Path
    output_dir: Path
    sam3d_code_root: Path
    checkpoint_path: Path
    mhr_path: Path
    prompt_bbox: np.ndarray | None
    prompt_bbox_frame: int | None = None
    start_frame: int = 0
    frame_step: int = 1
    max_frames: int | None = None
    # Spans (seconds, in the edited input video) to keep in the output at their
    # original timing but skip inference on ("masking"). Recorded as
    # subject-absent with reason "masked".
    mask_time_ranges: tuple[tuple[float, float], ...] = ()
    force_cpu: bool = False
    cpu_threads: int = 0
    render_preview: bool = True
    face_stride_overlay: int = 1
    face_stride_3d: int = 1
    output_codec: str = "h264"
    live_preview: bool = False
    live_preview_panel: str = "top-left"  # top-left|full
    live_preview_chunk_frames: int = 60  # loop length in frames for rolling preview
    live_preview_refresh_every: int = 1
    live_preview_codec: str = "h264"
    export_meshes: bool = True
    export_joint_timeseries: bool = True
    inference_precision: str = "float32"  # float32|float16
    sam3_code_root: Path | None = None
    inference_target: str = "body"  # body|hand
    auto_init_mode: str = "sam3"  # off|smart|sam3 (smart and sam3 are both SAM3-prompt-only)
    auto_detector_threshold: float = 0.5
    sam3_text_prompts: tuple[str, ...] = ("person",)
    auto_select_strategy: str = "patient"  # patient|largest|leftmost|rightmost|center
    enforce_ground_contact: bool = True
    ground_contact_auto: bool = True
    ground_contact_auto_calib_frames: int = 45
    ground_contact_quantile: float = 1.5
    ground_contact_smoothing: float = 0.35
    bbox_smoothing_alpha_slow: float = 0.55
    bbox_smoothing_alpha_fast: float = 0.95
    bbox_smoothing_fast_motion_ratio: float = 0.10
    identity_lock_enabled: bool = True
    identity_warmup_frames: int = 10
    identity_max_center_jump_ratio: float = 0.35
    identity_min_appearance_similarity: float = 0.32
    identity_reacquire_min_similarity: float = 0.42
    identity_reacquire_every_n: int = 4
    # While the subject is off-screen, only run the heavy SAM3 detector every N
    # frames instead of every frame — avoids thousands of wasted detections on
    # videos where the patient is absent for long stretches.
    identity_reacquire_when_lost_every_n: int = 6
    identity_max_hold_frames: int = 240
    # Independent-evidence + fixed-gallery controls (anti-drift / absence / re-entry).
    # Run the promptable person detector every N frames as identity ground truth,
    # gate the self-fed box against a frozen appearance gallery, and declare the
    # subject absent (no mesh) when no trusted detection supports it.
    # N=1 detects on EVERY frame (most accurate, no tracking-gap flicker) — heavy
    # off CUDA but the intended behaviour for clinical accuracy. Raise for speed.
    identity_detect_every_n: int = 1
    identity_gallery_floor: float = 0.30
    # Coasting (propagating the self-fed box without a fresh detection) must
    # clear this gallery similarity against the FROZEN patient gallery. Kept high
    # so the box is declared absent rather than coasting onto a different person
    # when the patient leaves the frame — a low floor let the box walk onto a
    # bystander whose colour histogram still scored ~0.6-0.7.
    identity_coast_gallery_floor: float = 0.70
    identity_absence_patience: int = 6
    identity_gallery_max_size: int = 8
    # Offline (non-causal) identity resolution: after the forward pass, replay the
    # logged detector candidates through a Viterbi that looks at past AND future
    # frames, and suppress frames where the greedy tracker briefly rode a
    # bystander (the "teleport"). It only ever flips a wrong-person frame to
    # subject-absent (never resurrects a frame — that needs a second inference
    # pass fed the exported resolved boxes). See resolve_identity_track.
    identity_offline_resolve: bool = True
    hand_temporal_enabled: bool = True
    hand_occlusion_hold_frames: int = 16
    hand_interpolation_max_gap: int = 8
    hand_reentry_blend_frames: int = 6
    hand_drift_max_center_jump_ratio: float = 0.82
    hand_drift_min_iou: float = 0.03
    hand_drift_max_area_ratio: float = 2.6
    hand_bbox_smoothing_alpha: float = 0.72
    hand_hold_follow_alpha: float = 0.22
    hand_mesh_smoothing_alpha: float = 0.58
    # Optional multi-anchor list (provided by web UI Manual mode). Currently
    # not consumed by the inner tracking loop; the median anchor is promoted
    # to prompt_bbox/prompt_bbox_frame at config-build time. Full list is
    # persisted to tracking_anchors.json for future re-anchoring work.
    tracking_anchors: list[dict[str, Any]] | None = None
    # Optional path to a dense per-frame box track of the chosen subject (from
    # the detect-step streaming preview). When set, it seeds manual_subject_bboxes
    # directly so the run reconstructs exactly that person.
    subject_track_file: str | None = None


@dataclass
class PipelineRuntime:
    """Reusable SAM3D runtime kept alive across several videos."""

    device: torch.device
    inference_dtype: torch.dtype
    mps_mhr_mode: str
    estimator: Any
    mhr_backend: str
    faces: Any


def write_progress_metadata(
    output_dir: Path,
    *,
    video_input: Path,
    mesh_dir: Path,
    records: list[dict[str, Any]],
    fps_output: float,
    video_width: int,
    video_height: int,
    inference_target: str,
    space_view: dict[str, Any] | None = None,
    processing_status: str = "running",
    total_frames_target: int | None = None,
) -> None:
    """Write an in-progress run_metadata.json + manifest so the viewer can poll mid-run."""
    metadata = {
        "video_input": str(video_input),
        "output_video": None,
        "mesh_dir": str(mesh_dir),
        "mesh_export_enabled": True,
        "joint_timeseries_export_enabled": True,
        "inference_target": inference_target,
        "fps_output": float(fps_output),
        "video_width": int(video_width),
        "video_height": int(video_height),
        "total_frames_processed": len(records),
        # Total frames this run will process — lets the viewer show a real %
        # mid-run (None when the source frame count is unknown).
        "total_frames_target": (int(total_frames_target) if total_frames_target else None),
        "processing_status": processing_status,
        "space_view": space_view,
        "records": records,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, separators=(",", ":")),
        encoding="utf-8",
    )
    manifest = build_run_manifest(
        run_id=output_dir.name,
        run_directory=output_dir,
        metadata=metadata,
    )
    manifest["processing_status"] = processing_status
    write_json(output_dir / "run_manifest.json", manifest)


def frame_is_masked(
    frame_idx: int, fps: float, mask_time_ranges: tuple[tuple[float, float], ...]
) -> bool:
    """True when the frame's timestamp falls inside a user-defined mask span."""
    if not mask_time_ranges:
        return False
    t = frame_idx / fps if fps > 0 else 0.0
    return any(start <= t < end for start, end in mask_time_ranges)


def draw_masked_overlay_panel(frame_bgr: np.ndarray) -> np.ndarray:
    """Dimmed raw frame with a MASKED label, used for masked-span preview frames."""
    panel = (frame_bgr.astype(np.float32) * 0.4).astype(np.uint8)
    cv2.putText(
        panel,
        "MASKED - not processed",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def append_subject_absent_record(
    records: list[dict[str, Any]],
    *,
    frame_idx: int,
    patient_bbox: np.ndarray | None,
    reason: str,
    identity_info: dict[str, Any] | None = None,
) -> None:
    """Append a frame record for intervals where the target subject is absent."""

    identity = identity_info or {}
    records.append(
        {
            "video_frame": int(frame_idx),
            "mesh_path": None,
            "bbox_xyxy": (
                [float(v) for v in patient_bbox.tolist()]
                if patient_bbox is not None
                else None
            ),
            "subject_present": False,
            "inference_status": reason,
            "subject_tracking_status": reason,
            "identity_lock_status": str(identity.get("status", reason)),
            "identity_is_lost": bool(identity.get("is_lost", True)),
            "identity_lost_frames": int(identity.get("lost_frames", 0)),
            "identity_appearance_similarity": (
                float(identity["appearance_similarity"])
                if identity.get("appearance_similarity") is not None
                else None
            ),
            "identity_stability_score": (
                float(identity["stability_score"])
                if identity.get("stability_score") is not None
                else None
            ),
            "identity_reacquire_scanned": bool(identity.get("reacquire_scanned", False)),
            "identity_reacquire_candidates": int(identity.get("reacquire_candidates", 0)),
        }
    )


def _configure_torch_runtime(cfg: PipelineConfig) -> None:
    """Wire the SAM3D repo onto sys.path, apply CUDA-compat shims, and set thread/precision."""
    add_sam3d_repo_to_path(cfg.sam3d_code_root)
    patch_sam3d_cuda_assumptions()

    if cfg.cpu_threads > 0:
        torch.set_num_threads(cfg.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    torch.set_float32_matmul_precision("high")


def build_pipeline_runtime(cfg: PipelineConfig) -> PipelineRuntime:
    """Load the model once so batch processing can reuse it video after video."""

    _configure_torch_runtime(cfg)

    device = select_device(force_cpu=cfg.force_cpu)
    precision = cfg.inference_precision.strip().lower()
    if precision not in {"float32", "float16"}:
        precision = "float32"
    if precision == "float16" and device.type not in {"mps", "cuda"}:
        print("float16 requested but GPU backend unavailable. Falling back to float32.")
        precision = "float32"
    inference_dtype = torch.float16 if precision == "float16" else torch.float32

    default_mps_mode = "native" if device.type == "mps" else "auto"
    mps_mhr_mode = os.environ.get("SAM3D_MHR_MODE", default_mps_mode).strip().lower()
    if mps_mhr_mode not in {"auto", "native", "wrapper"}:
        mps_mhr_mode = default_mps_mode

    estimator = load_estimator(
        checkpoint_path=cfg.checkpoint_path,
        mhr_path=cfg.mhr_path,
        device=device,
        mps_mhr_mode=mps_mhr_mode,
    )
    mhr_backend = getattr(estimator, "mhr_backend", "unknown")
    if device.type == "mps":
        print(
            "MPS fallback (CPU) enabled:"
            f" {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '1') == '1'}"
        )
        if mhr_backend == "native_mps_patched":
            print("MHR backend: native MPS (TorchScript float32 patch active).")
        elif mhr_backend == "cpu_wrapper_on_mps":
            print("MHR backend: CPU wrapper on MPS (TorchScript MHR not fully MPS compatible).")

    return PipelineRuntime(
        device=device,
        inference_dtype=inference_dtype,
        mps_mhr_mode=mps_mhr_mode,
        estimator=estimator,
        mhr_backend=mhr_backend,
        faces=estimator.faces,
    )


def parse_bbox(text: str) -> np.ndarray:
    """Parse a "x1,y1,x2,y2" CLI string into a validated [4] float32 bbox."""
    parts = [float(p.strip()) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("prompt_bbox must contain 4 values: x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bbox: expected x2>x1 and y2>y1")
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def default_center_bbox(frame_shape: tuple[int, int, int]) -> np.ndarray:
    """A generous centre box used as the last-resort prompt when auto-init is off."""
    h, w = frame_shape[:2]
    return np.array([w * 0.25, h * 0.1, w * 0.75, h * 0.98], dtype=np.float32)


def clip_bbox(bbox: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
    """Clamp an xyxy bbox to lie inside the frame while keeping x2>x1 and y2>y1."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox.astype(np.float32)
    x1 = float(np.clip(x1, 0, w - 2))
    y1 = float(np.clip(y1, 0, h - 2))
    x2 = float(np.clip(x2, x1 + 1, w - 1))
    y2 = float(np.clip(y2, y1 + 1, h - 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def normalize_sam3_text_prompts(prompts: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Canonicalise, de-duplicate and cap the user's text prompts (defaulting to "person")."""
    if prompts is None:
        return ("person",)

    out: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        p = str(prompt).strip()
        if not p:
            continue
        key_raw = p.casefold()
        canonical_map = {
            "hand only": "hand",
            "only hand": "hand",
            "just hand": "hand",
            "person only": "person",
            "only person": "person",
            "just person": "person",
            "main": "hand",
            "la main": "hand",
        }
        p = canonical_map.get(key_raw, p)
        key = p.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= 8:
            break
    if len(out) == 0:
        return ("person",)
    return tuple(out)


def normalize_inference_target(target: str | None) -> str:
    """Collapse the various "partial" target spellings to either "hand" or "body"."""
    value = str(target or "").strip().lower()
    if value in {"hand", "partial", "part", "non_full", "non-full"}:
        return "hand"
    return "body"


LEFT_HAND_KEYPOINT_IDXS_70 = np.arange(23, 43, dtype=np.int32)
RIGHT_HAND_KEYPOINT_IDXS_70 = np.arange(43, 63, dtype=np.int32)


def expand_bbox(
    bbox: np.ndarray,
    frame_shape: tuple[int, int, int],
    scale_x: float = 1.12,
    scale_y: float = 1.12,
) -> np.ndarray:
    """Grow a bbox about its centre by per-axis scale factors, clipped to the frame."""
    x1, y1, x2, y2 = bbox.astype(np.float32)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(1.0, (x2 - x1) * scale_x)
    h = max(1.0, (y2 - y1) * scale_y)
    expanded = np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)
    return clip_bbox(expanded, frame_shape)


def _candidate_score_patient(
    bbox: np.ndarray,
    frame_shape: tuple[int, int, int],
    detector_score: float | None,
) -> float:
    """Heuristic 0..1 "is this the patient" score from box geometry + detector confidence.

    Tuned for the clinical lane setup: tall, slender, low-standing, slightly
    centre-left boxes score highest.
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    area = bw * bh
    aspect = bw / bh
    bottom = float(y2) / max(1.0, h)
    center_x = float((x1 + x2) * 0.5) / max(1.0, w)

    height_score = np.clip(bh / max(1.0, h), 0.0, 1.0)
    area_score = np.clip(area / max(1.0, w * h), 0.0, 1.0)
    slender_score = float(max(0.0, 1.0 - abs(aspect - 0.45) / 0.45))
    bottom_score = np.clip(bottom, 0.0, 1.0)
    # Slight bias to center-left for this clinical setup where patient walks on the lane.
    center_score = float(max(0.0, 1.0 - abs(center_x - 0.42) / 0.42))
    conf_score = (
        float(np.clip(detector_score / 3.0, 0.0, 1.0))
        if detector_score is not None
        else 0.5
    )

    return (
        0.30 * height_score
        + 0.18 * area_score
        + 0.20 * slender_score
        + 0.12 * bottom_score
        + 0.10 * center_score
        + 0.10 * conf_score
    )


def select_patient_bbox_candidate(
    candidates: list[dict[str, Any]],
    frame_shape: tuple[int, int, int],
    strategy: str = "patient",
) -> dict[str, Any] | None:
    """Pick one candidate box per the requested selection strategy ("patient" heuristic by default)."""
    if not candidates:
        return None

    strat = strategy.strip().lower()
    if strat == "largest":
        return max(candidates, key=lambda c: (c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]))
    if strat == "leftmost":
        return min(candidates, key=lambda c: c["bbox"][0])
    if strat == "rightmost":
        return max(candidates, key=lambda c: c["bbox"][2])
    if strat == "center":
        w = frame_shape[1]
        return min(candidates, key=lambda c: abs((c["bbox"][0] + c["bbox"][2]) * 0.5 - (w * 0.5)))
    if strat == "tightest":
        return min(
            candidates,
            key=lambda c: (c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]),
        )

    # Default: "patient" heuristic.
    return max(
        candidates,
        key=lambda c: _candidate_score_patient(c["bbox"], frame_shape, c.get("score")),
    )


# Words that introduce a descriptive clause we can shed when the full prompt
# matches nobody. Order in the prompt doesn't matter; we always cut the
# rightmost one first so the most specific shorter phrase is tried next.
_PROMPT_CLAUSE_WORDS = frozenset(
    {
        "with",
        "holding",
        "carrying",
        "wearing",
        "in",
        "on",
        "near",
        "behind",
        "beside",
        "that",
        "who",
        "having",
        "and",
    }
)


def simplify_prompt_chain(prompt: str) -> list[str]:
    """Ordered prompt variants, most specific first, ending at "person".

    SAM3's open-vocabulary detector reliably matches short noun phrases
    ("person", "person in blue") but frequently returns *nothing* for compound
    descriptions like "person in blue with a bag". When the caller's specific
    prompt finds nobody we retry with progressively shorter phrases — dropping
    one trailing clause at a time — and finally the bare head noun "person", so
    a clearly-visible subject is never missed just because the description was
    over-specified. When the original prompt already matches, the chain's first
    entry hits and behavior is unchanged.
    """
    base = " ".join(str(prompt).split()).strip()
    if not base:
        return ["person"]

    variants: list[str] = [base]
    tokens = base.split()
    # Progressively truncate at the rightmost clause word (never index 0 — that
    # is the head noun we want to keep).
    while True:
        cut = None
        for i in range(len(tokens) - 1, 0, -1):
            if tokens[i].lower().strip(",.;") in _PROMPT_CLAUSE_WORDS:
                cut = i
                break
        if cut is None:
            break
        tokens = tokens[:cut]
        candidate = " ".join(tokens).strip()
        if candidate and candidate not in variants:
            variants.append(candidate)

    if not any(v.lower() == "person" for v in variants):
        variants.append("person")
    return variants


def _run_sam3_prompt(
    frame_bgr: np.ndarray,
    sam3_detector: Any,
    auto_detector_threshold: float,
    text_prompt: str,
) -> list[np.ndarray] | None:
    """Run SAM3 for one text prompt; return a list of [4] boxes, or None on error."""
    try:
        boxes = sam3_detector.run_human_detection(
            frame_bgr,
            bbox_thr=auto_detector_threshold,
            det_cat_id=0,
            default_to_full_image=False,
            text_prompt=text_prompt,
        )
    except Exception as exc:
        print(
            f"SAM3 detection failed for prompt '{text_prompt}': "
            f"{exc.__class__.__name__}: {exc}"
        )
        return None
    boxes_arr = np.asarray(boxes, dtype=np.float32)
    if boxes_arr.ndim == 1 and boxes_arr.shape[0] == 4:
        boxes_arr = boxes_arr.reshape(1, 4)
    if boxes_arr.size == 0:
        return []
    return [box for box in boxes_arr if box.shape[0] == 4]


def detect_sam3_prompt_candidates(
    frame_bgr: np.ndarray,
    *,
    sam3_detector: Any,
    auto_detector_threshold: float,
    sam3_text_prompts: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Detect subject candidates via SAM3, falling back from specific prompts to bare "person"."""

    def _mk(boxes: list[np.ndarray], source: str) -> list[dict[str, Any]]:
        """Wrap raw boxes as candidate dicts tagged with their detection source."""
        return [
            {
                "bbox": clip_bbox(box.astype(np.float32), frame_bgr.shape),
                "score": None,
                "source": source,
            }
            for box in boxes
        ]

    # Cheap presence gate. "person" is the most general human prompt, so if it
    # matches nobody the frame has no subject — bail after a single detector
    # pass. This bounds cost on the long empty stretches of a clinical corridor
    # video (and during tracking re-acquisition), where running every prompt
    # variant would otherwise triple the work for no gain.
    person_boxes = _run_sam3_prompt(frame_bgr, sam3_detector, auto_detector_threshold, "person")
    if not person_boxes:
        return []

    user_prompts_cf = {p.strip().casefold() for p in sam3_text_prompts}
    candidates: list[dict[str, Any]] = []
    matched_specific = False
    for text_prompt in sam3_text_prompts:
        # Try the full prompt first, then progressively simpler phrasings, and
        # stop at the first that matches. "person" is handled by the gate above.
        for variant in simplify_prompt_chain(text_prompt):
            if variant.casefold() == "person":
                continue
            boxes = _run_sam3_prompt(frame_bgr, sam3_detector, auto_detector_threshold, variant)
            if not boxes:
                continue
            is_fallback = variant.casefold() != text_prompt.strip().casefold()
            source = (
                f"sam3_prompt_fallback:{variant}" if is_fallback else f"sam3_prompt:{variant}"
            )
            candidates.extend(_mk(boxes, source))
            matched_specific = True
            break

    if not matched_specific:
        # Someone is in frame but nobody matches the description — fall back to
        # the generic person boxes (already detected by the gate, no extra pass).
        source = "sam3_prompt:person" if "person" in user_prompts_cf else "sam3_prompt_fallback:person"
        candidates.extend(_mk(person_boxes, source))
    return candidates


def auto_initialize_patient_bbox(
    frame_bgr: np.ndarray,
    auto_init_mode: str,
    auto_select_strategy: str,
    auto_detector_threshold: float,
    sam3_text_prompts: tuple[str, ...],
    sam3_detector: Any | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """First-frame patient bbox from the SAM3 prompt detector; (None, info) when nobody matches."""
    mode = auto_init_mode.strip().lower()
    info: dict[str, Any] = {
        "mode": mode,
        "selected_source": None,
        "selected_score": None,
        "num_candidates": 0,
        "fallback_used": False,
    }
    candidates: list[dict[str, Any]] = []

    # Auto initialization is strictly the SAM3 promptable detector with the
    # caller's text prompt — no HOG/heuristic fallback. If SAM3 finds nobody
    # matching the prompt, the subject is reported absent rather than guessed.
    if mode in {"smart", "sam3"} and sam3_detector is not None:
        candidates.extend(
            detect_sam3_prompt_candidates(
                frame_bgr,
                sam3_detector=sam3_detector,
                auto_detector_threshold=auto_detector_threshold,
                sam3_text_prompts=sam3_text_prompts,
            )
        )

    info["num_candidates"] = len(candidates)
    selected = select_patient_bbox_candidate(
        candidates,
        frame_shape=frame_bgr.shape,
        strategy=auto_select_strategy,
    )
    if selected is None:
        return None, info

    selected_bbox = expand_bbox(selected["bbox"], frame_bgr.shape, scale_x=1.10, scale_y=1.12)
    info["selected_source"] = selected.get("source")
    info["selected_score"] = selected.get("score")
    return selected_bbox, info


def bbox_from_keypoints(
    keypoints_2d: np.ndarray,
    frame_shape: tuple[int, int, int],
    expand_ratio: float = 0.20,
    min_size: int = 60,
    in_view_only: bool = False,
) -> np.ndarray | None:
    """Tight padded bbox around the valid 2D keypoints, or None if too few/too small."""
    if keypoints_2d.size == 0:
        return None
    valid = np.isfinite(keypoints_2d).all(axis=1)
    pts = keypoints_2d[valid]
    if in_view_only and pts.shape[0] > 0:
        h, w = frame_shape[:2]
        in_view = (
            (pts[:, 0] >= 0.0)
            & (pts[:, 0] <= float(max(0, w - 1)))
            & (pts[:, 1] >= 0.0)
            & (pts[:, 1] <= float(max(0, h - 1)))
        )
        pts = pts[in_view]
    if pts.shape[0] < 5:
        return None

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bw, bh = x2 - x1, y2 - y1
    if bw < 1 or bh < 1:
        return None

    pad_x = bw * expand_ratio
    pad_y = bh * expand_ratio
    cand = np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], dtype=np.float32)
    cand = clip_bbox(cand, frame_shape)
    if (cand[2] - cand[0]) < min_size or (cand[3] - cand[1]) < min_size:
        return None
    return cand


def bbox_from_projected_mesh(
    vertices_cam: np.ndarray,
    focal_length: float,
    frame_shape: tuple[int, int, int],
    *,
    expand_ratio: float = 0.10,
    min_size: int = 28,
    quantile_low: float = 0.03,
    quantile_high: float = 0.97,
) -> np.ndarray | None:
    """Bbox around the projected, in-view mesh vertices (quantile-trimmed against outliers)."""
    if vertices_cam.size == 0:
        return None
    u, v, valid = project_mesh_to_image(vertices_cam, focal_length, frame_shape)
    if not np.any(valid):
        return None

    h, w = frame_shape[:2]
    in_view = (
        valid
        & (u >= 0.0)
        & (u <= float(max(0, w - 1)))
        & (v >= 0.0)
        & (v <= float(max(0, h - 1)))
    )
    if int(np.count_nonzero(in_view)) < 12:
        return None

    uu = u[in_view]
    vv = v[in_view]
    q_lo = float(np.clip(quantile_low, 0.0, 0.49))
    q_hi = float(np.clip(quantile_high, 0.51, 1.0))
    x1 = float(np.quantile(uu, q_lo))
    y1 = float(np.quantile(vv, q_lo))
    x2 = float(np.quantile(uu, q_hi))
    y2 = float(np.quantile(vv, q_hi))
    if x2 <= x1 or y2 <= y1:
        x1 = float(np.min(uu))
        y1 = float(np.min(vv))
        x2 = float(np.max(uu))
        y2 = float(np.max(vv))
        if x2 <= x1 or y2 <= y1:
            return None

    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * float(max(0.0, expand_ratio))
    pad_y = bh * float(max(0.0, expand_ratio))
    cand = np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], dtype=np.float32)
    cand = clip_bbox(cand, frame_shape)
    if (cand[2] - cand[0]) < min_size and (cand[3] - cand[1]) < min_size:
        return None
    return cand


def smooth_bbox(prev_bbox: np.ndarray, new_bbox: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Exponential blend toward ``new_bbox`` (alpha=1 keeps the new box, 0 keeps the old)."""
    return (alpha * new_bbox + (1.0 - alpha) * prev_bbox).astype(np.float32)


def lerp_bbox(start_bbox: np.ndarray, end_bbox: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two boxes at clamped fraction ``t``."""
    weight = float(np.clip(t, 0.0, 1.0))
    return ((1.0 - weight) * start_bbox + weight * end_bbox).astype(np.float32)


def adaptive_bbox_smoothing_alpha(
    prev_bbox: np.ndarray,
    new_bbox: np.ndarray,
    *,
    alpha_slow: float = 0.55,
    alpha_fast: float = 0.95,
    fast_motion_ratio: float = 0.10,
) -> float:
    """
    Motion-adaptive smoothing:
    - small motion -> stronger smoothing (alpha closer to alpha_slow),
    - fast motion/impulse -> minimal lag (alpha closer to alpha_fast).
    """
    prev_cx = 0.5 * float(prev_bbox[0] + prev_bbox[2])
    prev_cy = 0.5 * float(prev_bbox[1] + prev_bbox[3])
    new_cx = 0.5 * float(new_bbox[0] + new_bbox[2])
    new_cy = 0.5 * float(new_bbox[1] + new_bbox[3])
    motion_px = float(np.hypot(new_cx - prev_cx, new_cy - prev_cy))

    bw = max(1.0, float(new_bbox[2] - new_bbox[0]))
    bh = max(1.0, float(new_bbox[3] - new_bbox[1]))
    box_diag = float(np.hypot(bw, bh))
    motion_ratio = motion_px / box_diag

    t = float(np.clip(motion_ratio / max(1e-6, fast_motion_ratio), 0.0, 1.0))
    return float(alpha_slow + (alpha_fast - alpha_slow) * t)


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Centre point [cx, cy] of an xyxy box."""
    b = bbox.astype(np.float32)
    return np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float32)


def bbox_size(bbox: np.ndarray) -> np.ndarray:
    """[width, height] of an xyxy box, floored at 1px per side."""
    b = bbox.astype(np.float32)
    return np.array([max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])], dtype=np.float32)


def bbox_area(bbox: np.ndarray) -> float:
    """Pixel area of an xyxy box."""
    s = bbox_size(bbox)
    return float(s[0] * s[1])


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two xyxy boxes (0 when disjoint)."""
    ax1, ay1, ax2, ay2 = a.astype(np.float32)
    bx1, by1, bx2, by2 = b.astype(np.float32)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def _scale_bbox_xyxy(bbox: np.ndarray, scale: float) -> np.ndarray:
    """Scale all four xyxy coordinates by ``scale`` (for moving between full/downscaled frames)."""
    return (bbox.astype(np.float32) * float(scale)).astype(np.float32)


def _clip_bbox_xyxy_to_hw(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp an xyxy box to a width/height (variant of clip_bbox taking explicit dims)."""
    x1, y1, x2, y2 = bbox.astype(np.float32)
    x1 = float(np.clip(x1, 0, max(0, width - 2)))
    y1 = float(np.clip(y1, 0, max(0, height - 2)))
    x2 = float(np.clip(x2, x1 + 1, max(1, width - 1)))
    y2 = float(np.clip(y2, y1 + 1, max(1, height - 1)))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _detect_tracking_points(gray: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray | None:
    """Good-features-to-track points inside a box, falling back to a regular grid."""
    h, w = gray.shape[:2]
    bbox = _clip_bbox_xyxy_to_hw(bbox_xyxy, w, h)
    x1, y1, x2, y2 = bbox.astype(np.int32).tolist()
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    mask = np.zeros_like(gray, dtype=np.uint8)
    pad_x = int(max(2, 0.05 * (x2 - x1)))
    pad_y = int(max(2, 0.05 * (y2 - y1)))
    roi_x1 = int(np.clip(x1 + pad_x, 0, w - 1))
    roi_y1 = int(np.clip(y1 + pad_y, 0, h - 1))
    roi_x2 = int(np.clip(x2 - pad_x, roi_x1 + 1, w))
    roi_y2 = int(np.clip(y2 - pad_y, roi_y1 + 1, h))
    mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255

    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=96,
        qualityLevel=0.01,
        minDistance=6,
        mask=mask,
        blockSize=7,
    )
    if points is not None and len(points) >= 6:
        return points.astype(np.float32)

    xs = np.linspace(roi_x1, roi_x2 - 1, num=4, dtype=np.float32)
    ys = np.linspace(roi_y1, roi_y2 - 1, num=6, dtype=np.float32)
    grid = np.array([[[x, y]] for y in ys for x in xs], dtype=np.float32)
    return grid if len(grid) > 0 else None


def _track_bbox_direction(
    gray_frames: list[np.ndarray],
    *,
    anchor_frame: int,
    anchor_bbox: np.ndarray,
    direction: int,
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """Translate a box frame-by-frame from an anchor via LK optical flow (forward or backward).

    Uses median point motion with forward-backward consistency filtering, capping
    per-frame steps so a flow blow-up cannot teleport the box. Returns the box per
    visited frame index plus tracked/flow-failure counts.
    """
    if direction not in {-1, 1} or not gray_frames:
        return {}, {"tracked": 0, "flow_failures": 0}

    h, w = gray_frames[0].shape[:2]
    current_bbox = _clip_bbox_xyxy_to_hw(anchor_bbox, w, h)
    current_points = _detect_tracking_points(gray_frames[anchor_frame], current_bbox)
    prev_gray = gray_frames[anchor_frame]
    out: dict[int, np.ndarray] = {}
    tracked = 0
    flow_failures = 0

    idx = anchor_frame + direction
    while 0 <= idx < len(gray_frames):
        next_gray = gray_frames[idx]
        moved = False
        if current_points is not None and len(current_points) >= 4:
            next_points, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                next_gray,
                current_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.03),
            )
            if next_points is not None and status is not None:
                good = status.reshape(-1) == 1
                if err is not None:
                    err_values = err.reshape(-1)
                    finite_err = err_values[np.isfinite(err_values)]
                    if finite_err.size > 0:
                        err_limit = float(np.quantile(finite_err, 0.80) + 1e-6)
                        good = good & (err_values <= err_limit)
                prev_pts = current_points.reshape(-1, 2)[good]
                next_pts = next_points.reshape(-1, 2)[good]
                if len(prev_pts) >= 4:
                    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                        next_gray,
                        prev_gray,
                        next_pts.reshape(-1, 1, 2).astype(np.float32),
                        None,
                        winSize=(21, 21),
                        maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 16, 0.03),
                    )
                    if back_points is not None and back_status is not None:
                        fb_dist = np.linalg.norm(
                            back_points.reshape(-1, 2) - prev_pts,
                            axis=1,
                        )
                        fb_good = (back_status.reshape(-1) == 1) & (fb_dist <= 3.0)
                        prev_pts = prev_pts[fb_good]
                        next_pts = next_pts[fb_good]
                if len(prev_pts) >= 4:
                    deltas = next_pts - prev_pts
                    delta = np.median(deltas, axis=0).astype(np.float32)
                    size = bbox_size(current_bbox)
                    max_step = max(4.0, 0.35 * float(np.hypot(size[0], size[1])))
                    delta_norm = float(np.linalg.norm(delta))
                    if np.isfinite(delta_norm) and delta_norm > max_step:
                        delta *= max_step / max(1e-6, delta_norm)
                    if np.isfinite(delta).all():
                        current_bbox = _clip_bbox_xyxy_to_hw(
                            current_bbox + np.array([delta[0], delta[1], delta[0], delta[1]], dtype=np.float32),
                            w,
                            h,
                        )
                        current_points = next_pts.reshape(-1, 1, 2).astype(np.float32)
                        moved = True
        if not moved:
            flow_failures += 1
            current_points = _detect_tracking_points(next_gray, current_bbox)
        elif len(current_points) < 16 or tracked % 12 == 11:
            refreshed = _detect_tracking_points(next_gray, current_bbox)
            if refreshed is not None and len(refreshed) >= len(current_points):
                current_points = refreshed

        out[idx] = current_bbox.copy()
        tracked += 1
        prev_gray = next_gray
        idx += direction

    return out, {"tracked": tracked, "flow_failures": flow_failures}


def build_manual_subject_bbox_track(
    video_input: Path,
    *,
    anchor_frame: int,
    anchor_bbox: np.ndarray,
    total_frames: int,
    width: int,
    height: int,
    max_long_side: int = 360,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Build a per-frame manual subject box track by LK-tracking a single user anchor both ways.

    Frames are pre-decoded to grayscale and downscaled (``max_long_side``) for
    speed, then tracked backward and forward from the anchor and rescaled back to
    original resolution. Returns {frame_idx: box} plus a tracking-info dict.
    """
    if total_frames <= 0:
        return {}, {"enabled": False, "reason": "empty_video"}

    anchor = int(np.clip(anchor_frame, 0, max(0, total_frames - 1)))
    scale = min(1.0, float(max_long_side) / float(max(width, height, 1)))
    gray_frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_input))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if scale < 0.999:
                gray = cv2.resize(
                    gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            gray_frames.append(gray)
    finally:
        cap.release()

    if not gray_frames:
        return {}, {"enabled": False, "reason": "read_failed"}
    if anchor >= len(gray_frames):
        anchor = len(gray_frames) - 1

    scaled_anchor_bbox = _scale_bbox_xyxy(anchor_bbox, scale)
    small_h, small_w = gray_frames[0].shape[:2]
    scaled_anchor_bbox = _clip_bbox_xyxy_to_hw(scaled_anchor_bbox, small_w, small_h)

    backward, backward_info = _track_bbox_direction(
        gray_frames,
        anchor_frame=anchor,
        anchor_bbox=scaled_anchor_bbox,
        direction=-1,
    )
    forward, forward_info = _track_bbox_direction(
        gray_frames,
        anchor_frame=anchor,
        anchor_bbox=scaled_anchor_bbox,
        direction=1,
    )

    small_bboxes: dict[int, np.ndarray] = {
        **backward,
        anchor: scaled_anchor_bbox,
        **forward,
    }
    inv_scale = 1.0 / max(scale, 1e-6)
    original_shape = (height, width, 3)
    bboxes = {
        idx: clip_bbox(_scale_bbox_xyxy(bbox, inv_scale), original_shape)
        for idx, bbox in small_bboxes.items()
    }
    return bboxes, {
        "enabled": True,
        "method": "bidirectional_lk_optical_flow",
        "anchor_frame_requested": int(anchor_frame),
        "anchor_frame_effective": int(anchor),
        "tracked_frames": int(len(bboxes)),
        "scale": float(scale),
        "backward": backward_info,
        "forward": forward_info,
    }


def _normalize_hist(hist: np.ndarray) -> np.ndarray:
    """Flatten a histogram and L1-normalise it (unchanged if its mass is ~0)."""
    h = hist.astype(np.float32).reshape(-1)
    s = float(h.sum())
    if s <= 1e-12:
        return h
    return h / s


def extract_bbox_appearance_hist(
    frame_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
) -> np.ndarray | None:
    """Identity appearance feature for a box: HS colour histogram (clothing) + down-weighted brightness."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox_xyxy, frame_bgr.shape).astype(np.int32).tolist()
    if x2 - x1 < 8 or y2 - y1 < 12:
        return None

    roi_bgr = frame_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return None

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat = roi_hsv[:, :, 1]
    val = roi_hsv[:, :, 2]

    # Keep informative cloth/body pixels, reduce background and dark shadows.
    valid_mask = ((sat > 20) & (val > 28)).astype(np.uint8) * 255
    if cv2.countNonZero(valid_mask) < max(25, int(0.02 * roi_bgr.shape[0] * roi_bgr.shape[1])):
        valid_mask = None

    hs_hist = cv2.calcHist(
        [roi_hsv],
        [0, 1],
        valid_mask,
        [24, 16],
        [0, 180, 0, 256],
    )
    v_hist = cv2.calcHist(
        [roi_hsv],
        [2],
        valid_mask,
        [8],
        [0, 256],
    )
    if hs_hist is None or v_hist is None:
        return None
    # Hue/saturation (clothing colour) is the identity-bearing cue; brightness is
    # lighting-dependent and barely distinguishes people, so weight it down hard.
    # Otherwise two people of similar overall brightness but different colours
    # score a misleadingly high similarity (~0.5) and absence/discrimination fail.
    hs = _normalize_hist(hs_hist.reshape(-1))
    v = _normalize_hist(v_hist.reshape(-1))
    feat = np.concatenate([0.85 * hs, 0.15 * v], axis=0).astype(np.float32)
    feat = _normalize_hist(feat)
    if not np.isfinite(feat).all() or float(feat.sum()) <= 1e-12:
        return None
    return feat


def appearance_similarity_score(
    reference_hist: np.ndarray | None,
    candidate_hist: np.ndarray | None,
) -> float | None:
    """Blend of Bhattacharyya/correlation/intersection histogram similarities, 0..1 (None if missing)."""
    if reference_hist is None or candidate_hist is None:
        return None
    ref = _normalize_hist(reference_hist)
    cand = _normalize_hist(candidate_hist)
    if ref.size != cand.size:
        return None

    corr = float(cv2.compareHist(ref, cand, cv2.HISTCMP_CORREL))
    corr_score = float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))

    bhatta = float(cv2.compareHist(ref, cand, cv2.HISTCMP_BHATTACHARYYA))
    bhatta_score = float(np.clip(1.0 - bhatta, 0.0, 1.0))

    inter = float(cv2.compareHist(ref, cand, cv2.HISTCMP_INTERSECT))
    inter_score = float(np.clip(inter, 0.0, 1.0))

    return float(0.40 * bhatta_score + 0.35 * corr_score + 0.25 * inter_score)


def gallery_match_score(
    gallery: "list[np.ndarray] | None",
    candidate_hist: np.ndarray | None,
) -> float | None:
    """Best appearance similarity of a candidate against a fixed identity gallery.

    The gallery holds several *frozen* views of the locked subject (manual
    anchors + early detector-supported frames), so back/profile views during a
    turning-in-place task still match at least one stored view. Unlike the
    adaptive appearance reference, the gallery never drifts, which is what stops
    the tracker from silently sliding onto a passer-by or the background.
    """
    if not gallery or candidate_hist is None:
        return None
    best: float | None = None
    for ref in gallery:
        score = appearance_similarity_score(ref, candidate_hist)
        if score is None:
            continue
        if best is None or score > best:
            best = score
    return best


def _candidate_stability_score(
    candidate_bbox: np.ndarray,
    predicted_bbox: np.ndarray,
    appearance_score: float | None,
    max_center_jump_ratio: float,
) -> dict[str, float]:
    """Score a candidate box against a motion prediction (appearance + motion + IoU + scale)."""
    cand_center = bbox_center(candidate_bbox)
    pred_center = bbox_center(predicted_bbox)
    pred_size = bbox_size(predicted_bbox)
    cand_size = bbox_size(candidate_bbox)

    dist = float(np.linalg.norm(cand_center - pred_center))
    pred_diag = float(np.hypot(float(pred_size[0]), float(pred_size[1])))
    motion_ratio = dist / max(1e-6, pred_diag)
    motion_score = float(np.clip(1.0 - (motion_ratio / max(1e-6, max_center_jump_ratio)), 0.0, 1.0))

    iou = bbox_iou(candidate_bbox, predicted_bbox)
    scale_ratio = float(max(cand_size[0] * cand_size[1], 1.0) / max(pred_size[0] * pred_size[1], 1.0))
    scale_score = float(np.exp(-abs(np.log(max(scale_ratio, 1e-6)))))
    app = 0.5 if appearance_score is None else float(np.clip(appearance_score, 0.0, 1.0))

    combined = float(
        0.44 * app
        + 0.28 * motion_score
        + 0.18 * iou
        + 0.10 * scale_score
    )
    return {
        "combined": combined,
        "appearance": app,
        "motion_score": motion_score,
        "motion_ratio": motion_ratio,
        "iou": iou,
        "scale_score": scale_score,
    }


def collect_person_candidates_for_reacquire(
    frame_bgr: np.ndarray,
    *,
    sam3_detector: Any | None,
    auto_detector_threshold: float,
    sam3_text_prompts: tuple[str, ...],
) -> list[dict[str, Any]]:
    """SAM3 person candidates (with light NMS) used as independent re-acquisition evidence."""
    # Re-acquisition evidence is strictly the SAM3 promptable detector — no HOG
    # fallback. With no detector or no prompt match, there are no candidates and
    # the subject stays absent until the prompt detector sees them again.
    candidates: list[dict[str, Any]] = []
    if sam3_detector is not None:
        for candidate in detect_sam3_prompt_candidates(
            frame_bgr,
            sam3_detector=sam3_detector,
            auto_detector_threshold=auto_detector_threshold,
            sam3_text_prompts=sam3_text_prompts,
        ):
            candidate["score"] = 1.0
            candidates.append(candidate)

    # Light NMS to avoid duplicate boxes.
    if len(candidates) <= 1:
        return candidates
    candidates_sorted = sorted(
        candidates,
        key=lambda c: float(c.get("score", 0.0)),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    for cand in candidates_sorted:
        if all(bbox_iou(cand["bbox"], keep["bbox"]) < 0.55 for keep in selected):
            selected.append(cand)
    return selected


class IdentityLockedBboxTracker:
    """
    Single-patient identity lock with a *fixed* appearance gallery, independent
    per-frame detector evidence, distractor memory, and explicit absence.

    Design goals (why this is not a plain appearance-EMA tracker):

    - The proposed bbox each frame is derived from the pose model's own fit on
      the previous box, so it is self-reinforcing. We therefore gate it against
      a frozen identity gallery (anchors + early supported views) that cannot
      drift, and against independent person detections when available.
    - When another person is in frame we keep a short-lived ``distractors``
      memory so a passer-by is actively rejected instead of silently adopted.
    - When the subject leaves the frame the detector finds no trusted patient,
      so we declare ``absent`` (subject_present=False) instead of hallucinating
      a body, and only re-acquire on a confident gallery match when they return.
    """

    def __init__(
        self,
        *,
        initial_bbox: np.ndarray,
        frame_shape: tuple[int, int, int],
        enabled: bool = True,
        warmup_frames: int = 10,
        max_center_jump_ratio: float = 0.35,
        min_appearance_similarity: float = 0.32,
        reacquire_min_similarity: float = 0.42,
        reacquire_every_n: int = 4,
        reacquire_when_lost_every_n: int = 1,
        max_hold_frames: int = 240,
        gallery_floor: float = 0.30,
        coast_gallery_floor: float = 0.70,
        absence_patience: int = 6,
        gallery_max_size: int = 8,
        support_min_iou: float = 0.20,
        distractor_ttl: int = 90,
        gallery_seeds: "list[np.ndarray] | None" = None,
    ) -> None:
        """Initialise the locked box, gate thresholds, fixed gallery seeds and counters."""
        self.enabled = bool(enabled)
        self.frame_shape = frame_shape
        self.current_bbox = clip_bbox(initial_bbox, frame_shape)
        self.prev_bbox = self.current_bbox.copy()
        self.velocity_xy = np.zeros(2, dtype=np.float32)

        self.warmup_frames = max(1, int(warmup_frames))
        self.warmup_left = self.warmup_frames
        self.max_center_jump_ratio = float(max(0.08, max_center_jump_ratio))
        self.min_appearance_similarity = float(np.clip(min_appearance_similarity, 0.0, 1.0))
        self.reacquire_min_similarity = float(np.clip(reacquire_min_similarity, 0.0, 1.0))
        self.reacquire_every_n = max(1, int(reacquire_every_n))
        self.reacquire_when_lost_every_n = max(1, int(reacquire_when_lost_every_n))
        self.max_hold_frames = max(1, int(max_hold_frames))

        # Fixed identity gallery + anti-drift gates.
        self.gallery_floor = float(np.clip(gallery_floor, 0.0, 1.0))
        self.coast_gallery_floor = float(np.clip(coast_gallery_floor, 0.0, 1.0))
        self.absence_patience = max(0, int(absence_patience))
        self.gallery_max_size = max(1, int(gallery_max_size))
        self.support_min_iou = float(np.clip(support_min_iou, 0.0, 1.0))
        self.distractor_ttl = max(1, int(distractor_ttl))
        self.fixed_gallery: list[np.ndarray] = []
        for seed in gallery_seeds or []:
            self._add_gallery_view(seed)
        # Distractors: list of [hist, ttl] for non-patient people seen recently.
        self.distractors: list[list[Any]] = []

        self.appearance_ref: np.ndarray | None = None
        self.is_lost = False
        self.lost_frames = 0
        self.frames_since_reacquire_scan = 0
        self.support_misses = 0

        self.total_blocked_switches = 0
        self.total_reacquired = 0
        self.total_lost_events = 0
        self.total_absent_frames = 0
        self.total_distractor_blocks = 0
        self.last_status = "boot"

        # Occlusion-aware hold: when two detections overlap (a crossing) the
        # appearance crops blend and a greedy single-target pick can flip onto
        # the other person. Instead, carry the locked subject straight through on
        # constant-velocity motion — no switch, no appearance/gallery update —
        # and let the normal gallery-dominant selection re-anchor to the right
        # person once the detections separate. Main defence against ID swaps.
        self.occlusion_hold_enabled = True
        self.occlusion_iou_thresh = 0.30
        self.max_occlusion_frames = 30
        self.occluded_frames = 0
        self.total_occlusion_holds = 0
        self.last_occlusion_iou = 0.0
        # Last accepted *detection* centre — velocity is driven from the detector
        # (the reliable subject position) rather than the smoother self-fed box,
        # so the occlusion hold carries the true motion straight through a cross.
        self.last_det_center: np.ndarray | None = None

    def _predict_bbox(self) -> np.ndarray:
        """Constant-velocity prediction of next box: current centre shifted by velocity."""
        center = bbox_center(self.current_bbox) + self.velocity_xy
        size = bbox_size(self.current_bbox)
        pred = np.array(
            [
                center[0] - 0.5 * size[0],
                center[1] - 0.5 * size[1],
                center[0] + 0.5 * size[0],
                center[1] + 0.5 * size[1],
            ],
            dtype=np.float32,
        )
        return clip_bbox(pred, self.frame_shape)

    def _update_motion(self, accepted_bbox: np.ndarray) -> None:
        """EMA-update the velocity estimate from the accepted centre displacement."""
        old_center = bbox_center(self.current_bbox)
        new_center = bbox_center(accepted_bbox)
        delta = (new_center - old_center).astype(np.float32)
        vel_alpha = 0.35
        self.velocity_xy = (1.0 - vel_alpha) * self.velocity_xy + vel_alpha * delta

    def _update_appearance(self, new_hist: np.ndarray | None) -> None:
        """Blend the (drift-prone) adaptive appearance reference toward a new view."""
        if new_hist is None:
            return
        if self.appearance_ref is None:
            self.appearance_ref = new_hist.astype(np.float32)
            return
        # Slower update after warmup so we keep long-term patient identity.
        update_rate = 0.28 if self.warmup_left > 0 else 0.08
        blended = (1.0 - update_rate) * self.appearance_ref + update_rate * new_hist
        self.appearance_ref = _normalize_hist(blended)

    def _add_gallery_view(self, new_hist: np.ndarray | None) -> None:
        """Store a frozen identity view, keeping the gallery small but diverse.

        Only frames we *trust* (anchors, detector-supported boxes) feed this, so
        it captures the patient's front/back/profile across a turn without ever
        absorbing a distractor.
        """
        if new_hist is None:
            return
        hist = _normalize_hist(new_hist.astype(np.float32))
        if not np.isfinite(hist).all() or float(hist.sum()) <= 1e-12:
            return
        if not self.fixed_gallery:
            self.fixed_gallery.append(hist)
            return
        # Skip near-duplicates; replace the most redundant entry once full.
        sims = [appearance_similarity_score(ref, hist) or 0.0 for ref in self.fixed_gallery]
        if max(sims) >= 0.93:
            return
        if len(self.fixed_gallery) < self.gallery_max_size:
            self.fixed_gallery.append(hist)
            return
        # Gallery full: replace the entry most similar to the rest (least informative).
        redundancy = []
        for i, ref in enumerate(self.fixed_gallery):
            others = [
                appearance_similarity_score(ref, other) or 0.0
                for j, other in enumerate(self.fixed_gallery)
                if j != i
            ]
            redundancy.append(max(others) if others else 0.0)
        self.fixed_gallery[int(np.argmax(redundancy))] = hist

    def _decay_distractors(self) -> None:
        """Age out the short-lived distractor memory by one frame, dropping expired entries."""
        for entry in self.distractors:
            entry[1] -= 1
        self.distractors = [e for e in self.distractors if e[1] > 0]

    def _remember_distractor(self, hist: np.ndarray | None) -> None:
        """Record a non-patient appearance (merging near-duplicates) so it can be rejected later."""
        if hist is None:
            return
        hist = _normalize_hist(hist.astype(np.float32))
        for entry in self.distractors:
            if (appearance_similarity_score(entry[0], hist) or 0.0) >= 0.90:
                entry[0] = _normalize_hist(0.5 * entry[0] + 0.5 * hist)
                entry[1] = self.distractor_ttl
                return
        self.distractors.append([hist, self.distractor_ttl])

    def _distractor_match(self, hist: np.ndarray | None) -> float:
        """Best similarity of an appearance to any remembered distractor (0 if none)."""
        if hist is None or not self.distractors:
            return 0.0
        return max(
            (appearance_similarity_score(e[0], hist) or 0.0) for e in self.distractors
        )

    def _accept_candidate(
        self,
        candidate_bbox: np.ndarray,
        candidate_hist: np.ndarray | None,
        *,
        status: str,
        reacquired: bool = False,
        supported: bool = False,
        update_ref: bool = True,
        grow_gallery: bool = False,
    ) -> None:
        """Commit a box as the patient: update motion/appearance/gallery and clear the lost state."""
        accepted = clip_bbox(candidate_bbox, self.frame_shape)
        self._update_motion(accepted)
        self.prev_bbox = self.current_bbox.copy()
        self.current_bbox = accepted
        # Only adapt the (drift-prone) reference on frames we trust, never on a
        # bare self-fed coast — that is exactly how template drift creeps in.
        if update_ref:
            self._update_appearance(candidate_hist)
        if grow_gallery:
            self._add_gallery_view(candidate_hist)
        was_lost = self.is_lost
        self.is_lost = False
        self.lost_frames = 0
        self.support_misses = 0
        self.frames_since_reacquire_scan = 0
        self.last_status = status
        if self.warmup_left > 0:
            self.warmup_left -= 1
        if reacquired and was_lost:
            self.total_reacquired += 1

    def _hold_prediction(self) -> np.ndarray:
        """Advance the box to its motion prediction (used while lost), damping velocity."""
        pred = self._predict_bbox()
        # Damp velocity while lost to avoid drifting forever.
        self.velocity_xy *= 0.92
        self.prev_bbox = self.current_bbox.copy()
        self.current_bbox = pred
        return pred

    def _detections_crossing(self, detections: list[dict[str, Any]]) -> float:
        """Max pairwise IoU among the detections; high means two people overlap (a crossing)."""
        boxes = [
            clip_bbox(d["bbox"], self.frame_shape)
            for d in detections
            if d.get("bbox") is not None
        ]
        best = 0.0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                best = max(best, bbox_iou(boxes[i], boxes[j]))
        return float(best)

    def _update_motion_from_detection(self, det_bbox: np.ndarray) -> None:
        """EMA the velocity from consecutive detection centres.

        The self-fed box can lag the subject (and then a constant-velocity hold
        barely moves), so we prefer the detector — the most reliable measure of
        where the subject actually is — to drive motion through a crossing.
        """
        center = bbox_center(det_bbox).astype(np.float32)
        if self.last_det_center is not None:
            delta = center - self.last_det_center
            self.velocity_xy = (0.6 * self.velocity_xy + 0.4 * delta).astype(np.float32)
        self.last_det_center = center

    def _coast_through_occlusion(self) -> np.ndarray:
        """Advance the box by its (undamped) constant velocity.

        Used to carry the locked subject straight through a crossing without
        switching identity — unlike ``_hold_prediction`` it does NOT damp the
        velocity, because the subject is still moving normally, just occluded.
        """
        pred = self._predict_bbox()
        self.prev_bbox = self.current_bbox.copy()
        self.current_bbox = pred
        return pred

    def _score_detection(
        self,
        frame_bgr: np.ndarray,
        det_bbox: np.ndarray,
        predicted_bbox: np.ndarray,
        *,
        relaxed: bool,
    ) -> dict[str, Any]:
        """Identity-first score for a detection: fixed-gallery affinity dominant, motion as tie-breaker."""
        det_hist = extract_bbox_appearance_hist(frame_bgr, det_bbox)
        gallery = gallery_match_score(self.fixed_gallery, det_hist)
        adaptive = appearance_similarity_score(self.appearance_ref, det_hist)
        distractor = self._distractor_match(det_hist)
        metrics = _candidate_stability_score(
            det_bbox,
            predicted_bbox,
            gallery if gallery is not None else adaptive,
            self.max_center_jump_ratio * 1.8,
        )
        # Identity-first score: appearance vs the fixed gallery dominates, with
        # motion continuity as tie-breaker. Penalise boxes that look like a
        # known distractor more than like the patient.
        identity = gallery if gallery is not None else (adaptive if adaptive is not None else 0.5)
        penalty = max(0.0, distractor - (gallery if gallery is not None else 0.0))
        combined = float(
            0.62 * identity + 0.30 * metrics["motion_score"] + 0.08 * metrics["iou"] - 0.5 * penalty
        )
        return {
            "bbox": det_bbox,
            "hist": det_hist,
            "gallery": gallery,
            "adaptive": adaptive,
            "distractor": distractor,
            "motion_score": metrics["motion_score"],
            "combined": combined,
        }

    def _select_patient_detection(
        self,
        frame_bgr: np.ndarray,
        detections: list[dict[str, Any]],
        predicted_bbox: np.ndarray,
    ) -> "dict[str, Any] | None":
        """Pick the detection that is the locked patient; flag the rest as distractors.

        During warmup (or before any gallery exists) we trust motion continuity;
        afterwards a detection must clear the fixed-gallery floor and not look
        more like a remembered distractor than like the patient.
        """
        relaxed = self.warmup_left > 0 or not self.fixed_gallery
        scored = [
            self._score_detection(
                frame_bgr, clip_bbox(d["bbox"], self.frame_shape), predicted_bbox, relaxed=relaxed
            )
            for d in detections
            if d.get("bbox") is not None
        ]
        if not scored:
            return None
        scored.sort(key=lambda s: s["combined"], reverse=True)
        best = scored[0]

        accept = False
        if relaxed:
            # Trust geometry while we still learn identity; avoid obvious distractors.
            accept = best["motion_score"] >= 0.25 and best["distractor"] <= 0.85
        else:
            gallery_ok = best["gallery"] is not None and best["gallery"] >= self.gallery_floor
            not_distractor = best["gallery"] is None or best["distractor"] <= best["gallery"] + 0.05
            accept = gallery_ok and not_distractor

        if not accept:
            for s in scored:
                self._remember_distractor(s["hist"])
            if scored:
                self.total_distractor_blocks += 1
            return None
        # Everyone except the chosen patient is a distractor to remember.
        for s in scored[1:]:
            self._remember_distractor(s["hist"])
        # Record how much the chosen patient box overlaps the nearest other
        # detection — a crossing/occlusion signal for downstream logic + QA.
        self.last_occlusion_iou = max(
            (bbox_iou(best["bbox"], s["bbox"]) for s in scored[1:]),
            default=0.0,
        )
        return best

    def _info(
        self,
        status: str,
        *,
        present: bool,
        supported: bool,
        appearance: float | None,
        gallery: float | None,
        stability: float | None,
        reacquire_scanned: bool = False,
        reacquire_candidates: int = 0,
    ) -> dict[str, Any]:
        """Build the per-frame status dict returned by ``update`` for the caller's record."""
        return {
            "status": status,
            "present": bool(present),
            "supported": bool(supported),
            "is_lost": bool(not present),
            "lost_frames": int(self.lost_frames),
            "appearance_similarity": appearance,
            "gallery_similarity": gallery,
            "stability_score": stability,
            "reacquire_scanned": bool(reacquire_scanned),
            "reacquire_candidates": int(reacquire_candidates),
            "distractors_tracked": len(self.distractors),
            "gallery_size": len(self.fixed_gallery),
            "blocked_switches_total": self.total_blocked_switches,
            "reacquired_total": self.total_reacquired,
            "occlusion_iou": float(getattr(self, "last_occlusion_iou", 0.0)),
        }

    def _enter_lost(
        self,
        *,
        status: str,
        appearance: float | None,
        gallery: float | None,
        stability: float | None,
        reacquire_scanned: bool = False,
        reacquire_candidates: int = 0,
    ) -> "tuple[np.ndarray, dict[str, Any]]":
        """Transition into the lost/absent state: bump counters, hold the prediction, report absent."""
        if not self.is_lost:
            self.is_lost = True
            self.total_lost_events += 1
        self.lost_frames += 1
        self.total_blocked_switches += 1
        self.total_absent_frames += 1
        self.last_status = status
        held = self._hold_prediction()
        if self.lost_frames > self.max_hold_frames:
            self.velocity_xy *= 0.0
        return held.copy(), self._info(
            status,
            present=False,
            supported=False,
            appearance=appearance,
            gallery=gallery,
            stability=stability,
            reacquire_scanned=reacquire_scanned,
            reacquire_candidates=reacquire_candidates,
        )

    def update(
        self,
        frame_bgr: np.ndarray,
        proposed_bbox: np.ndarray | None,
        *,
        detections: list[dict[str, Any]] | None = None,
        anchor_bbox: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Advance one frame.

        ``proposed_bbox`` is the self-fed box from the pose model's own fit.
        ``detections`` is the list of independent person detections for THIS
        frame (``None`` = detector was not run this frame; ``[]`` = it ran and
        found nobody trusted). ``anchor_bbox`` re-asserts user ground truth.
        """
        if not self.enabled:
            if proposed_bbox is not None:
                self.current_bbox = clip_bbox(proposed_bbox, self.frame_shape)
            return self.current_bbox.copy(), self._info(
                "disabled", present=True, supported=False,
                appearance=None, gallery=None, stability=None,
            )

        self._decay_distractors()
        # Per-frame crossing/occlusion signal (max IoU of the chosen box vs other
        # detections); reset each frame, set when a detection is selected.
        self.last_occlusion_iou = 0.0
        predicted_bbox = self._predict_bbox()

        # (0) Hard anchor re-assertion — user ground truth overrides everything.
        if anchor_bbox is not None:
            acc = clip_bbox(anchor_bbox, self.frame_shape)
            hist = extract_bbox_appearance_hist(frame_bgr, acc)
            self._accept_candidate(
                acc, hist, status="anchor", reacquired=self.is_lost,
                supported=True, update_ref=True, grow_gallery=True,
            )
            return self.current_bbox.copy(), self._info(
                "anchor", present=True, supported=True,
                appearance=appearance_similarity_score(self.appearance_ref, hist),
                gallery=gallery_match_score(self.fixed_gallery, hist),
                stability=1.0,
            )

        candidate_bbox = (
            clip_bbox(proposed_bbox, self.frame_shape) if proposed_bbox is not None else None
        )
        candidate_hist = (
            extract_bbox_appearance_hist(frame_bgr, candidate_bbox)
            if candidate_bbox is not None else None
        )
        cand_gallery = gallery_match_score(self.fixed_gallery, candidate_hist)
        cand_adaptive = appearance_similarity_score(self.appearance_ref, candidate_hist)
        cand_metrics = (
            _candidate_stability_score(
                candidate_bbox,
                predicted_bbox,
                cand_adaptive if cand_adaptive is not None else cand_gallery,
                self.max_center_jump_ratio,
            )
            if candidate_bbox is not None else None
        )

        in_warmup = self.warmup_left > 0
        relaxed = in_warmup or not self.fixed_gallery
        score_thr = 0.36 if in_warmup else 0.50

        # (A) Independent detector evidence is available this frame.
        if detections is not None:
            self.frames_since_reacquire_scan = 0
            # Occlusion-aware hold: two overlapping detections = a crossing, where
            # blended crops make a greedy pick flip identity. While we hold a
            # confident lock, carry the subject through on constant-velocity
            # motion (no switch, no appearance/gallery update); the gallery-
            # dominant selection below re-anchors once the people separate.
            crossing_iou = self._detections_crossing(detections)
            self.last_occlusion_iou = crossing_iou
            if (
                self.occlusion_hold_enabled
                and not relaxed
                and not self.is_lost
                and self.fixed_gallery
                and crossing_iou >= self.occlusion_iou_thresh
                and self.occluded_frames < self.max_occlusion_frames
            ):
                self.occluded_frames += 1
                self.total_occlusion_holds += 1
                held = self._coast_through_occlusion()
                return held.copy(), self._info(
                    "occluded", present=True, supported=False,
                    appearance=cand_adaptive, gallery=cand_gallery,
                    stability=(cand_metrics["combined"] if cand_metrics else None),
                    reacquire_scanned=True, reacquire_candidates=len(detections),
                )
            self.occluded_frames = 0
            best_det = self._select_patient_detection(frame_bgr, detections, predicted_bbox)
            if best_det is not None:
                was_lost = self.is_lost
                status = "reacquired_detector" if was_lost else "tracked"
                supported_candidate = (
                    candidate_bbox is not None
                    and bbox_iou(candidate_bbox, best_det["bbox"]) >= self.support_min_iou
                    and (relaxed or (cand_gallery is not None and cand_gallery >= self.gallery_floor))
                )
                if supported_candidate:
                    # Detection confirms the smoother self-fed box: keep it, but
                    # take the velocity from the detection so a later occlusion
                    # hold carries the subject's true motion (not the lagging box).
                    self._accept_candidate(
                        candidate_bbox, candidate_hist, status=status, reacquired=was_lost,
                        supported=True, update_ref=True, grow_gallery=True,
                    )
                    self._update_motion_from_detection(best_det["bbox"])
                    return self.current_bbox.copy(), self._info(
                        status, present=True, supported=True,
                        appearance=cand_adaptive, gallery=cand_gallery,
                        stability=(cand_metrics["combined"] if cand_metrics else best_det["combined"]),
                        reacquire_scanned=True, reacquire_candidates=len(detections),
                    )
                # Self-fed box drifted off the patient: snap back to the detection.
                self._accept_candidate(
                    best_det["bbox"], best_det["hist"], status=status, reacquired=was_lost,
                    supported=True, update_ref=True, grow_gallery=True,
                )
                self._update_motion_from_detection(best_det["bbox"])
                return self.current_bbox.copy(), self._info(
                    status, present=True, supported=True,
                    appearance=best_det["adaptive"], gallery=best_det["gallery"],
                    stability=best_det["combined"],
                    reacquire_scanned=True, reacquire_candidates=len(detections),
                )

            # Detector ran but found no trusted patient -> the subject is not here.
            self.support_misses += 1
            coast_ok = (
                candidate_bbox is not None
                and not relaxed
                and not self.is_lost
                and cand_gallery is not None
                and cand_gallery >= self.coast_gallery_floor
                and cand_metrics is not None
                and cand_metrics["motion_score"] >= 0.2
                and self.support_misses <= self.absence_patience
            )
            if coast_ok:
                self._accept_candidate(
                    candidate_bbox, candidate_hist, status="tracked_coast",
                    supported=False, update_ref=False, grow_gallery=False,
                )
                return self.current_bbox.copy(), self._info(
                    "tracked_coast", present=True, supported=False,
                    appearance=cand_adaptive, gallery=cand_gallery,
                    stability=(cand_metrics["combined"] if cand_metrics else None),
                    reacquire_scanned=True, reacquire_candidates=len(detections),
                )
            return self._enter_lost(
                status="absent",
                appearance=cand_adaptive, gallery=cand_gallery,
                stability=(cand_metrics["combined"] if cand_metrics else None),
                reacquire_scanned=True, reacquire_candidates=len(detections),
            )

        # (B) No detector this frame: coast on the self-fed box only while it
        #     still clears the FIXED identity floor, and never recover from a
        #     lost state without detector evidence (prevents re-latching drift).
        self.frames_since_reacquire_scan += 1
        accept_coast = False
        if cand_metrics is not None and not self.is_lost:
            if relaxed:
                accept_coast = cand_metrics["combined"] >= score_thr
            else:
                # Coast only while the self-fed box still clears the (stricter)
                # coast floor against the FROZEN gallery. The adaptive ref can
                # drift onto a nearby bystander, so it must not justify a coast —
                # that is exactly how the box used to walk onto a second person
                # once the patient left the frame.
                identity_ok = (
                    cand_gallery is not None and cand_gallery >= self.coast_gallery_floor
                )
                accept_coast = identity_ok and cand_metrics["combined"] >= score_thr
        if accept_coast and candidate_bbox is not None:
            status = "tracked" if in_warmup else "tracked_coast"
            self._accept_candidate(
                candidate_bbox, candidate_hist, status=status, supported=False,
                update_ref=(cand_gallery is not None and cand_gallery >= self.gallery_floor),
                grow_gallery=False,
            )
            return self.current_bbox.copy(), self._info(
                status, present=True, supported=False,
                appearance=cand_adaptive, gallery=cand_gallery,
                stability=cand_metrics["combined"],
            )
        # Temporal hysteresis: right after confirmed support, a single self-fed
        # dip (mid-turn blur, brief partial box) shouldn't flip the subject to
        # lost between detector-cadence frames. Hold the prediction as present
        # for a bounded grace; the next detection frame re-confirms or, if the
        # subject really left, escalates to absence within ~one cadence interval.
        # Only grace a *benign* dip: the box must be geometrically continuous
        # with the motion prediction (a mid-turn identity dip), never a jump
        # onto a distractor or the background — that distinction is what keeps
        # the hold from silently re-introducing drift.
        benign_dip = cand_metrics is not None and cand_metrics["motion_score"] >= 0.5
        if not self.is_lost and self.support_misses < self.absence_patience and benign_dip:
            self.support_misses += 1
            held = self._hold_prediction()
            self.last_status = "coasting"
            return held.copy(), self._info(
                "coasting", present=True, supported=False,
                appearance=cand_adaptive, gallery=cand_gallery,
                stability=(cand_metrics["combined"] if cand_metrics else None),
            )
        return self._enter_lost(
            status="lost_hold",
            appearance=cand_adaptive, gallery=cand_gallery,
            stability=(cand_metrics["combined"] if cand_metrics else None),
        )

    def summary(self) -> dict[str, Any]:
        """Final tracker config + lifetime counters for the run manifest."""
        return {
            "enabled": self.enabled,
            "warmup_frames": self.warmup_frames,
            "max_center_jump_ratio": self.max_center_jump_ratio,
            "min_appearance_similarity": self.min_appearance_similarity,
            "reacquire_min_similarity": self.reacquire_min_similarity,
            "reacquire_every_n": self.reacquire_every_n,
            "max_hold_frames": self.max_hold_frames,
            "gallery_floor": self.gallery_floor,
            "coast_gallery_floor": self.coast_gallery_floor,
            "absence_patience": self.absence_patience,
            "gallery_size": len(self.fixed_gallery),
            "lost_events_total": self.total_lost_events,
            "absent_frames_total": self.total_absent_frames,
            "blocked_switches_total": self.total_blocked_switches,
            "distractor_blocks_total": self.total_distractor_blocks,
            "reacquired_total": self.total_reacquired,
            "occlusion_holds_total": self.total_occlusion_holds,
            "last_status": self.last_status,
            "currently_lost": self.is_lost,
            "lost_frames": self.lost_frames,
        }

    def needs_reacquire_candidates(self) -> bool:
        """Whether the caller should run the independent detector this frame.

        While LOST we used to demand a detection on *every* frame — on a video
        where the subject is off-screen for long stretches (e.g. a clinical
        corridor) that runs the heavy SAM3 detector thousands of needless times.
        Throttle the lost-state scan to ``reacquire_when_lost_every_n`` frames;
        the subject is still re-found within a fraction of a second.
        """
        if not self.enabled:
            return False
        if self.warmup_left > 0:
            return True
        if self.is_lost:
            return self.frames_since_reacquire_scan + 1 >= self.reacquire_when_lost_every_n
        return self.frames_since_reacquire_scan + 1 >= self.reacquire_every_n


def resolve_identity_track(
    detection_frames: list[dict[str, Any]],
    *,
    frame_diag: float,
    affinity_baseline: float = 0.62,
    hard_floor: float = 0.45,
    switch_penalty: float = 0.45,
    jump_weight: float = 0.6,
    jump_switch_ratio: float = 0.12,
    reentry_cost: float = 0.10,
) -> list[dict[str, Any]]:
    """Offline (non-causal) identity resolution by Viterbi over detection frames.

    This is the future-aware counterpart to the greedy, per-frame choice in
    :class:`IdentityLockedBboxTracker`. A single frame where a bystander beats
    the patient on motion can flip the greedy tracker; here the whole window is
    optimised jointly, so a brief, low-affinity distractor cannot win — the
    future frames where the real patient reappears pull the optimum back.

    Each entry in ``detection_frames`` is::

        {"frame_idx": int,
         "candidates": [{"bbox": np.ndarray, "gallery": float | None}, ...]}

    ``gallery`` is the affinity to the FROZEN identity gallery — the drift-free
    signal. A candidate's reward for being the patient is ``gallery -
    affinity_baseline`` (so a weak match scores near or below the ABSENT reward
    of 0), gated by ``hard_floor``. The DP maximises
    ``Σ reward − Σ transition_cost``; present→present transitions across a large
    centre jump pay ``switch_penalty`` so the box never teleports between people.
    States per frame: every qualifying candidate, plus ABSENT (index ``-1``).
    """
    n_frames = len(detection_frames)
    if n_frames == 0:
        return []
    neg_inf = -1e9
    diag = max(1.0, float(frame_diag))

    def reward(frame: dict[str, Any], sidx: int) -> float:
        """Per-state reward: gallery affinity above baseline (0 for ABSENT, -inf below the floor)."""
        if sidx == -1:
            return 0.0
        gallery = frame["candidates"][sidx].get("gallery")
        if gallery is None or gallery < hard_floor:
            return neg_inf
        return float(gallery) - affinity_baseline

    def center_of(frame: dict[str, Any], sidx: int) -> np.ndarray | None:
        if sidx == -1:
            return None
        return bbox_center(frame["candidates"][sidx]["bbox"])

    def transition(pf: dict, ps: int, cf: dict, cs: int) -> float:
        """State-to-state cost: centre-jump distance plus a switch penalty for identity teleports."""
        if ps == -1 and cs == -1:
            return 0.0
        if ps == -1 or cs == -1:
            return reentry_cost
        prev_c, cur_c = center_of(pf, ps), center_of(cf, cs)
        jump = float(np.linalg.norm(cur_c - prev_c)) / diag
        cost = jump_weight * jump
        if jump > jump_switch_ratio:  # a jump this large is an identity switch
            cost += switch_penalty
        return cost

    # Forward Viterbi. dp[t] maps state index -> (best_score, back_pointer).
    def states_of(t: int) -> list[int]:
        return list(range(len(detection_frames[t]["candidates"]))) + [-1]

    dp: list[dict[int, tuple[float, int | None]]] = []
    first = {s: (reward(detection_frames[0], s), None) for s in states_of(0)}
    dp.append({s: v for s, v in first.items() if v[0] > neg_inf / 2})
    for t in range(1, n_frames):
        frame, pframe = detection_frames[t], detection_frames[t - 1]
        cur: dict[int, tuple[float, int | None]] = {}
        for s in states_of(t):
            r = reward(frame, s)
            if r <= neg_inf / 2:
                continue
            best_val: float | None = None
            best_prev: int | None = None
            for ps, (psc, _) in dp[t - 1].items():
                val = psc + r - transition(pframe, ps, frame, s)
                if best_val is None or val > best_val:
                    best_val, best_prev = val, ps
            cur[s] = (r if best_val is None else best_val, best_prev)
        if not cur:  # everyone disqualified -> force ABSENT
            cur[-1] = (max((v[0] for v in dp[t - 1].values()), default=0.0), None)
        dp.append(cur)

    # Backtrack the highest-scoring path.
    end_state = max(dp[-1].items(), key=lambda kv: kv[1][0])[0]
    path = [end_state]
    for t in range(n_frames - 1, 0, -1):
        end_state = dp[t][end_state][1]
        if end_state is None:
            end_state = -1
        path.append(end_state)
    path.reverse()

    out: list[dict[str, Any]] = []
    for t, s in enumerate(path):
        frame = detection_frames[t]
        present = s is not None and s != -1
        out.append({
            "frame_idx": frame["frame_idx"],
            "state": "present" if present else "absent",
            "bbox": frame["candidates"][s]["bbox"] if present else None,
            "cand_idx": s if present else -1,
        })
    return out


def apply_offline_identity_resolution(
    records: list[dict[str, Any]],
    identity_trace: list[dict[str, Any]],
    *,
    frame_shape: tuple[int, int, int],
    mesh_dir: Path,
    gallery_floor: float,
    affinity_baseline: float = 0.62,
) -> dict[str, Any]:
    """Second pass: suppress frames where the greedy tracker rode a bystander.

    ``identity_trace`` carries, for every detector-cadence frame, the candidate
    boxes + their frozen-gallery affinity and the index of that frame's record.
    We resolve a temporally-consistent patient track over the whole video
    (:func:`resolve_identity_track`) and, where the resolver says "absent" yet
    the forward pass emitted a *low-affinity* box, flip the record to
    subject-absent and delete the wrong-person mesh. It never resurrects a frame
    (that needs a 2nd inference pass fed ``resolved_subject_bboxes``); it only
    removes teleports. Returns a summary for the run manifest.
    """
    summary: dict[str, Any] = {
        "applied": False,
        "detection_frames": 0,
        "suppressed_frames": 0,
    }
    if not identity_trace:
        summary["reason"] = "no_trace"
        return summary

    trace = sorted(identity_trace, key=lambda e: e["record_index"])
    detection_frames = [
        {"frame_idx": e["frame_idx"], "candidates": e["candidates"]} for e in trace
    ]
    frame_diag = float(np.hypot(frame_shape[1], frame_shape[0]))
    resolved = resolve_identity_track(
        detection_frames,
        frame_diag=frame_diag,
        affinity_baseline=affinity_baseline,
        hard_floor=max(0.0, gallery_floor - 0.25),
    )
    summary["detection_frames"] = len(detection_frames)

    # Map each detector-frame resolution onto its record, then carry it forward
    # over the coast frames until the next detector frame.
    resolved_by_record: dict[int, dict[str, Any]] = {
        e["record_index"]: r for e, r in zip(trace, resolved)
    }
    resolved_bboxes: dict[int, list[float]] = {
        r["frame_idx"]: [float(v) for v in r["bbox"].tolist()]
        for r in resolved
        if r["state"] == "present" and r["bbox"] is not None
    }

    suppressed = 0
    current_state: str | None = None
    for idx, record in enumerate(records):
        if idx in resolved_by_record:
            current_state = resolved_by_record[idx]["state"]
        if current_state != "absent":
            continue
        if not record.get("subject_present"):
            continue
        # Only suppress a *low-affinity* box — never a confidently-tracked
        # patient frame the resolver merely lacked a detection for.
        used_gallery = record.get("identity_gallery_similarity")
        if used_gallery is not None and used_gallery >= gallery_floor:
            continue
        mesh_path = record.get("mesh_path")
        if mesh_path:
            try:
                Path(mesh_path).unlink(missing_ok=True)
            except OSError:
                pass
        record["mesh_path"] = None
        record["subject_present"] = False
        record["inference_status"] = "subject_absent"
        record["subject_tracking_status"] = "distractor_suppressed"
        record["identity_lock_status"] = "distractor_suppressed"
        record["identity_offline_suppressed"] = True
        suppressed += 1

    summary["applied"] = True
    summary["suppressed_frames"] = suppressed
    summary["resolved_present_frames"] = len(resolved_bboxes)
    summary["resolved_subject_bboxes"] = resolved_bboxes
    return summary


class HandTemporalPostprocessor:
    """
    Stabilize hand/partial outputs over time:
    - reject bbox outliers (anti-drift),
    - hold last good state during short occlusions,
    - blend re-entry after temporary disappearance,
    - smooth mesh when topology stays stable.
    """

    def __init__(
        self,
        *,
        frame_shape: tuple[int, int, int],
        enabled: bool = True,
        occlusion_hold_frames: int = 16,
        interpolation_max_gap: int = 8,
        reentry_blend_frames: int = 6,
        max_center_jump_ratio: float = 0.82,
        min_iou: float = 0.03,
        max_area_ratio: float = 2.6,
        bbox_smoothing_alpha: float = 0.72,
        hold_follow_alpha: float = 0.22,
        mesh_smoothing_alpha: float = 0.58,
    ) -> None:
        """Initialise drift/occlusion/re-entry thresholds and the last-good mesh state."""
        self.enabled = bool(enabled)
        self.frame_shape = frame_shape
        self.occlusion_hold_frames = max(1, int(occlusion_hold_frames))
        self.interpolation_max_gap = max(0, int(interpolation_max_gap))
        self.reentry_blend_frames = max(0, int(reentry_blend_frames))
        self.max_center_jump_ratio = float(max(0.10, max_center_jump_ratio))
        self.min_iou = float(np.clip(min_iou, 0.0, 1.0))
        self.max_area_ratio = float(max(1.10, max_area_ratio))
        self.bbox_smoothing_alpha = float(np.clip(bbox_smoothing_alpha, 0.0, 1.0))
        self.hold_follow_alpha = float(np.clip(hold_follow_alpha, 0.0, 1.0))
        self.mesh_smoothing_alpha = float(np.clip(mesh_smoothing_alpha, 0.0, 1.0))

        self.current_bbox: np.ndarray | None = None
        self.missing_frames = 0

        self.last_good_vertices_cam: np.ndarray | None = None
        self.last_good_vertices_space_cam: np.ndarray | None = None
        self.last_good_faces: np.ndarray | None = None

        self.reentry_left = 0
        self.reentry_anchor_bbox: np.ndarray | None = None
        self.reentry_anchor_vertices_cam: np.ndarray | None = None
        self.reentry_anchor_vertices_space_cam: np.ndarray | None = None

        self.lost_events_total = 0
        self.drift_rejects_total = 0
        self.reentry_events_total = 0
        self.last_status = "boot"

    @staticmethod
    def _mesh_valid(vertices_cam: np.ndarray, faces: np.ndarray) -> bool:
        """True if the mesh has enough vertices/faces to be worth keeping."""
        return bool(
            vertices_cam.ndim == 2
            and faces.ndim == 2
            and vertices_cam.shape[0] >= 16
            and faces.shape[0] >= 12
        )

    @staticmethod
    def _same_topology(
        faces_a: np.ndarray | None,
        vertices_a: np.ndarray | None,
        faces_b: np.ndarray,
        vertices_b: np.ndarray,
    ) -> bool:
        """True when two meshes share identical topology, so EMA vertex blending is valid."""
        if faces_a is None or vertices_a is None:
            return False
        if faces_a.shape != faces_b.shape:
            return False
        if vertices_a.shape[0] != vertices_b.shape[0]:
            return False
        return bool(np.array_equal(faces_a, faces_b))

    @staticmethod
    def _blend_vertices(
        previous_vertices: np.ndarray,
        current_vertices: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Per-vertex blend toward the current mesh (alpha=1 keeps current, 0 keeps previous)."""
        weight = float(np.clip(alpha, 0.0, 1.0))
        return (
            weight * current_vertices.astype(np.float32)
            + (1.0 - weight) * previous_vertices.astype(np.float32)
        ).astype(np.float32)

    def update(
        self,
        *,
        prior_bbox: np.ndarray | None,
        measured_bbox: np.ndarray | None,
        measured_vertices_cam: np.ndarray,
        measured_vertices_space_cam: np.ndarray,
        measured_faces: np.ndarray,
        bbox_source: str,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Advance one frame: reject drift, hold/interp through occlusions, EMA-smooth the mesh.

        Returns the stabilised (bbox, vertices_cam, vertices_space_cam, faces, info).
        """
        empty_vertices = np.empty((0, 3), dtype=np.float32)
        empty_faces = np.empty((0, 3), dtype=np.int32)

        mesh_valid = self._mesh_valid(measured_vertices_cam, measured_faces)
        measured_bbox_clipped = (
            clip_bbox(measured_bbox, self.frame_shape)
            if measured_bbox is not None
            else None
        )
        prior_clipped = clip_bbox(prior_bbox, self.frame_shape) if prior_bbox is not None else None

        if not self.enabled:
            passthrough_bbox = measured_bbox_clipped if measured_bbox_clipped is not None else prior_clipped
            info = {
                "enabled": False,
                "status": "disabled_passthrough",
                "bbox_source": bbox_source,
                "mesh_source": "measured" if mesh_valid else "empty",
                "missing_frames": 0,
                "drift_rejected": False,
                "center_jump_ratio": None,
                "iou_to_ref": None,
                "area_ratio_to_ref": None,
                "hold_active": False,
                "bbox_available": passthrough_bbox is not None,
                "mesh_available": mesh_valid,
                "lost_events_total": 0,
                "drift_rejects_total": 0,
                "reentry_events_total": 0,
            }
            return (
                passthrough_bbox,
                measured_vertices_cam if mesh_valid else empty_vertices,
                measured_vertices_space_cam if mesh_valid else empty_vertices,
                measured_faces if mesh_valid else empty_faces,
                info,
            )

        ref_bbox = self.current_bbox.copy() if self.current_bbox is not None else (
            prior_clipped.copy() if prior_clipped is not None else None
        )

        drift_rejected = False
        center_jump_ratio: float | None = None
        iou_to_ref: float | None = None
        area_ratio_to_ref: float | None = None
        candidate_bbox = measured_bbox_clipped.copy() if measured_bbox_clipped is not None else None
        if candidate_bbox is not None and ref_bbox is not None:
            dist = float(np.linalg.norm(bbox_center(candidate_bbox) - bbox_center(ref_bbox)))
            ref_diag = float(np.hypot(*bbox_size(ref_bbox)))
            center_jump_ratio = dist / max(1e-6, ref_diag)
            iou_to_ref = bbox_iou(candidate_bbox, ref_bbox)
            area_ratio_to_ref = float(
                max(
                    bbox_area(candidate_bbox) / max(1.0, bbox_area(ref_bbox)),
                    bbox_area(ref_bbox) / max(1.0, bbox_area(candidate_bbox)),
                )
            )

            source = str(bbox_source).strip().lower()
            strict_source = source in {"raw_output", "fallback_output"}
            jump_thr = self.max_center_jump_ratio * (0.82 if strict_source else 1.0)
            area_thr = self.max_area_ratio * (0.85 if strict_source else 1.0)
            if (
                (center_jump_ratio > jump_thr and iou_to_ref < self.min_iou)
                or (area_ratio_to_ref > area_thr and iou_to_ref < 0.18)
            ):
                drift_rejected = True
                self.drift_rejects_total += 1
                candidate_bbox = None

        was_missing = int(self.missing_frames)
        if candidate_bbox is not None:
            if self.current_bbox is None:
                self.current_bbox = candidate_bbox.copy()
            else:
                smoothed_bbox = smooth_bbox(
                    self.current_bbox,
                    candidate_bbox,
                    alpha=self.bbox_smoothing_alpha,
                )
                can_interp_reentry = (
                    was_missing > 0
                    and was_missing <= self.interpolation_max_gap
                    and self.reentry_blend_frames > 0
                )
                if can_interp_reentry:
                    if self.reentry_left <= 0:
                        self.reentry_left = self.reentry_blend_frames
                        self.reentry_events_total += 1
                        self.reentry_anchor_bbox = self.current_bbox.copy()
                        self.reentry_anchor_vertices_cam = (
                            self.last_good_vertices_cam.copy()
                            if self.last_good_vertices_cam is not None
                            else None
                        )
                        self.reentry_anchor_vertices_space_cam = (
                            self.last_good_vertices_space_cam.copy()
                            if self.last_good_vertices_space_cam is not None
                            else None
                        )
                    if self.reentry_anchor_bbox is not None and self.reentry_left > 0:
                        blend_t = 1.0 - (
                            self.reentry_left / float(self.reentry_blend_frames + 1)
                        )
                        interp_bbox = lerp_bbox(
                            self.reentry_anchor_bbox,
                            candidate_bbox,
                            blend_t,
                        )
                        smoothed_bbox = smooth_bbox(smoothed_bbox, interp_bbox, alpha=0.72)
                        self.reentry_left -= 1
                        if self.reentry_left <= 0:
                            self.reentry_anchor_bbox = None
                else:
                    self.reentry_left = 0
                    self.reentry_anchor_bbox = None
                    self.reentry_anchor_vertices_cam = None
                    self.reentry_anchor_vertices_space_cam = None
                self.current_bbox = clip_bbox(smoothed_bbox, self.frame_shape)
            self.missing_frames = 0
        else:
            if self.missing_frames == 0:
                self.lost_events_total += 1
            self.missing_frames += 1

            if self.current_bbox is None and prior_clipped is not None:
                self.current_bbox = prior_clipped.copy()
            elif self.current_bbox is not None and prior_clipped is not None and self.hold_follow_alpha > 0.0:
                self.current_bbox = smooth_bbox(
                    self.current_bbox,
                    prior_clipped,
                    alpha=self.hold_follow_alpha,
                )

            if self.missing_frames > self.interpolation_max_gap:
                self.reentry_left = 0
                self.reentry_anchor_bbox = None
                self.reentry_anchor_vertices_cam = None
                self.reentry_anchor_vertices_space_cam = None

        bbox_out = self.current_bbox.copy() if self.current_bbox is not None else None

        mesh_out_cam = empty_vertices
        mesh_out_space_cam = empty_vertices
        mesh_out_faces = empty_faces
        mesh_source = "empty"
        if mesh_valid:
            mesh_candidate_cam = measured_vertices_cam.astype(np.float32, copy=False)
            mesh_candidate_space = measured_vertices_space_cam.astype(np.float32, copy=False)
            mesh_candidate_faces = measured_faces.astype(np.int32, copy=False)
            mesh_source = "measured"

            can_ema = self._same_topology(
                self.last_good_faces,
                self.last_good_vertices_cam,
                mesh_candidate_faces,
                mesh_candidate_cam,
            )
            if can_ema and self.mesh_smoothing_alpha > 0.0:
                if (
                    was_missing > 0
                    and self.reentry_blend_frames > 0
                    and self.reentry_anchor_vertices_cam is not None
                    and self.reentry_anchor_vertices_space_cam is not None
                    and self.reentry_anchor_vertices_cam.shape == mesh_candidate_cam.shape
                    and self.reentry_anchor_vertices_space_cam.shape == mesh_candidate_space.shape
                ):
                    reentry_denom = max(1.0, float(self.reentry_blend_frames))
                    reentry_t = float(np.clip(1.0 - (self.reentry_left / reentry_denom), 0.0, 1.0))
                    mesh_candidate_cam = self._blend_vertices(
                        self.reentry_anchor_vertices_cam,
                        mesh_candidate_cam,
                        reentry_t,
                    )
                    mesh_candidate_space = self._blend_vertices(
                        self.reentry_anchor_vertices_space_cam,
                        mesh_candidate_space,
                        reentry_t,
                    )
                    mesh_source = "reentry_interp"
                else:
                    assert self.last_good_vertices_cam is not None
                    assert self.last_good_vertices_space_cam is not None
                    mesh_candidate_cam = self._blend_vertices(
                        self.last_good_vertices_cam,
                        mesh_candidate_cam,
                        self.mesh_smoothing_alpha,
                    )
                    mesh_candidate_space = self._blend_vertices(
                        self.last_good_vertices_space_cam,
                        mesh_candidate_space,
                        self.mesh_smoothing_alpha,
                    )
                    mesh_source = "ema"

            mesh_out_cam = mesh_candidate_cam
            mesh_out_space_cam = mesh_candidate_space
            mesh_out_faces = mesh_candidate_faces
            self.last_good_vertices_cam = mesh_out_cam.copy()
            self.last_good_vertices_space_cam = mesh_out_space_cam.copy()
            self.last_good_faces = mesh_out_faces.copy()
        elif (
            self.last_good_vertices_cam is not None
            and self.last_good_vertices_space_cam is not None
            and self.last_good_faces is not None
            and self.missing_frames <= self.occlusion_hold_frames
        ):
            mesh_out_cam = self.last_good_vertices_cam.copy()
            mesh_out_space_cam = self.last_good_vertices_space_cam.copy()
            mesh_out_faces = self.last_good_faces.copy()
            mesh_source = "hold"
        elif self.missing_frames > (self.occlusion_hold_frames + self.interpolation_max_gap + 2):
            self.last_good_vertices_cam = None
            self.last_good_vertices_space_cam = None
            self.last_good_faces = None

        if candidate_bbox is not None and mesh_valid:
            status = "reentry_blend" if was_missing > 0 else "tracked"
        elif candidate_bbox is not None:
            status = "bbox_only"
        elif drift_rejected and bbox_out is not None and self.missing_frames <= self.occlusion_hold_frames:
            status = "drift_reject_hold"
        elif bbox_out is not None and self.missing_frames <= self.occlusion_hold_frames:
            status = "occluded_hold"
        else:
            status = "lost_empty"

        self.last_status = status
        info = {
            "enabled": True,
            "status": status,
            "bbox_source": bbox_source,
            "mesh_source": mesh_source,
            "missing_frames": int(self.missing_frames),
            "drift_rejected": bool(drift_rejected),
            "center_jump_ratio": center_jump_ratio,
            "iou_to_ref": iou_to_ref,
            "area_ratio_to_ref": area_ratio_to_ref,
            "hold_active": bool(self.missing_frames > 0 and self.missing_frames <= self.occlusion_hold_frames),
            "bbox_available": bbox_out is not None,
            "mesh_available": bool(self._mesh_valid(mesh_out_cam, mesh_out_faces)),
            "lost_events_total": int(self.lost_events_total),
            "drift_rejects_total": int(self.drift_rejects_total),
            "reentry_events_total": int(self.reentry_events_total),
        }
        return bbox_out, mesh_out_cam, mesh_out_space_cam, mesh_out_faces, info

    def summary(self) -> dict[str, Any]:
        """Final hand-postprocess config + lifetime counters for the run manifest."""
        return {
            "enabled": self.enabled,
            "occlusion_hold_frames": self.occlusion_hold_frames,
            "interpolation_max_gap": self.interpolation_max_gap,
            "reentry_blend_frames": self.reentry_blend_frames,
            "max_center_jump_ratio": self.max_center_jump_ratio,
            "min_iou": self.min_iou,
            "max_area_ratio": self.max_area_ratio,
            "bbox_smoothing_alpha": self.bbox_smoothing_alpha,
            "hold_follow_alpha": self.hold_follow_alpha,
            "mesh_smoothing_alpha": self.mesh_smoothing_alpha,
            "lost_events_total": self.lost_events_total,
            "drift_rejects_total": self.drift_rejects_total,
            "reentry_events_total": self.reentry_events_total,
            "missing_frames": self.missing_frames,
            "last_status": self.last_status,
        }


def export_mesh(vertices_cam: np.ndarray, faces: np.ndarray, output_path: Path) -> None:
    """Write a triangle mesh as a binary little-endian PLY.

    Hand-rolled rather than via trimesh: faster for per-frame exports and keeps
    deterministic topology.
    """
    vertices = np.asarray(vertices_cam, dtype=np.float32)
    tri_faces = np.asarray(faces, dtype=np.int32)
    face_dtype = np.dtype(
        [
            ("count", np.uint8),
            ("idx", np.int32, (3,)),
        ]
    )
    face_data = np.empty(tri_faces.shape[0], dtype=face_dtype)
    face_data["count"] = 3
    face_data["idx"] = tri_faces

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {tri_faces.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    with output_path.open("wb") as f:
        f.write(header)
        f.write(vertices.astype("<f4", copy=False).tobytes())
        f.write(face_data.astype(face_dtype.newbyteorder("<"), copy=False).tobytes())


def project_mesh_to_image(
    vertices_cam: np.ndarray,
    focal_length: float,
    image_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole-project camera-space vertices to pixel (u, v); ``valid`` flags points in front."""
    h, w = image_shape[:2]
    x = vertices_cam[:, 0]
    y = vertices_cam[:, 1]
    z = vertices_cam[:, 2]
    valid = z > 1e-6
    u = np.full_like(x, -1.0, dtype=np.float32)
    v = np.full_like(y, -1.0, dtype=np.float32)
    u[valid] = focal_length * (x[valid] / z[valid]) + (w / 2.0)
    v[valid] = focal_length * (y[valid] / z[valid]) + (h / 2.0)
    return u, v, valid


def _compact_mesh_from_vertex_mask(
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    vertex_mask: np.ndarray,
    *,
    min_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Sub-mesh of faces touching the masked vertices, re-indexed compactly.

    Returns (vertices, faces, kept_vertex_indices); falls back to the original
    mesh with ``None`` indices when too few faces survive.
    """
    tri_inside = vertex_mask[faces]
    keep_faces = np.sum(tri_inside, axis=1) >= 2
    if int(np.count_nonzero(keep_faces)) < min_faces:
        keep_faces = np.sum(tri_inside, axis=1) >= 1
    if int(np.count_nonzero(keep_faces)) < min_faces:
        return vertices_cam, faces, None

    selected_faces = faces[keep_faces]
    used_vertices = np.unique(selected_faces.reshape(-1))
    if used_vertices.size < 16:
        return vertices_cam, faces, None

    remap = np.full(vertices_cam.shape[0], -1, dtype=np.int32)
    remap[used_vertices] = np.arange(used_vertices.shape[0], dtype=np.int32)
    compact_faces = remap[selected_faces]
    compact_vertices = vertices_cam[used_vertices]
    if compact_faces.size == 0 or compact_vertices.size == 0:
        return vertices_cam, faces, None
    return compact_vertices, compact_faces, used_vertices


def _select_dominant_hand_keypoints(
    keypoints_2d: np.ndarray | None,
    keypoints_3d: np.ndarray | None,
    image_shape: tuple[int, int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Pick the more visible hand (left vs right) and return its in-view 2D/3D keypoints."""
    if keypoints_2d is None or keypoints_2d.ndim != 2 or keypoints_2d.shape[1] < 2:
        return None, None

    h, w = image_shape[:2]
    n = keypoints_2d.shape[0]
    candidates: list[tuple[float, np.ndarray, np.ndarray | None]] = []
    for idxs in (LEFT_HAND_KEYPOINT_IDXS_70, RIGHT_HAND_KEYPOINT_IDXS_70):
        if int(idxs.max(initial=0)) >= n:
            continue
        hand_2d_raw = keypoints_2d[idxs, :2].astype(np.float32, copy=False)
        finite_2d = np.isfinite(hand_2d_raw).all(axis=1)
        if not np.any(finite_2d):
            continue

        hand_2d = hand_2d_raw[finite_2d]
        x = hand_2d[:, 0]
        y = hand_2d[:, 1]
        in_view = (x >= -40.0) & (x <= (w + 40.0)) & (y >= -40.0) & (y <= (h + 40.0))
        if np.count_nonzero(in_view) < 4:
            continue

        hand_2d = hand_2d[in_view]
        x = hand_2d[:, 0]
        y = hand_2d[:, 1]
        span_x = float(np.max(x) - np.min(x))
        span_y = float(np.max(y) - np.min(y))
        area = max(1.0, span_x * span_y)

        hand_3d: np.ndarray | None = None
        if keypoints_3d is not None and keypoints_3d.ndim == 2 and keypoints_3d.shape[1] >= 3:
            if int(idxs.max(initial=0)) < keypoints_3d.shape[0]:
                hand_3d_raw = keypoints_3d[idxs, :3].astype(np.float32, copy=False)
                finite_3d = np.isfinite(hand_3d_raw).all(axis=1)
                hand_3d = hand_3d_raw[finite_3d] if np.any(finite_3d) else None

        score = float(hand_2d.shape[0]) * 1000.0 + area
        candidates.append((score, hand_2d, hand_3d))

    if len(candidates) == 0:
        return None, None
    _, best_2d, best_3d = max(candidates, key=lambda item: item[0])
    return best_2d, best_3d


def extract_hand_mesh_from_keypoints(
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    focal_length: float,
    image_shape: tuple[int, int, int],
    keypoints_2d: np.ndarray | None,
    keypoints_3d: np.ndarray | None,
    *,
    min_faces: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Carve the hand sub-mesh out of the full-body mesh using the dominant hand's keypoints.

    Crops by the hand's projected 2D box and (when available) a 3D depth band so
    only the hand's vertices survive. Returns (vertices, faces, kept_indices).
    """
    if vertices_cam.size == 0 or faces.size == 0:
        return vertices_cam, faces, None

    hand_2d, hand_3d = _select_dominant_hand_keypoints(keypoints_2d, keypoints_3d, image_shape)
    if hand_2d is None or hand_2d.shape[0] < 4:
        return vertices_cam, faces, None

    h, w = image_shape[:2]
    x = hand_2d[:, 0]
    y = hand_2d[:, 1]
    span = max(float(np.max(x) - np.min(x)), float(np.max(y) - np.min(y)), 1.0)
    pad = max(28.0, span * 0.9)
    x1 = max(0.0, float(np.min(x) - pad))
    y1 = max(0.0, float(np.min(y) - pad))
    x2 = min(float(w - 1), float(np.max(x) + pad))
    y2 = min(float(h - 1), float(np.max(y) + pad))

    u, v, valid = project_mesh_to_image(vertices_cam, focal_length, image_shape)
    in_box = valid & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    if not np.any(in_box):
        return vertices_cam, faces, None

    vertex_mask = in_box.copy()
    if hand_3d is not None and hand_3d.shape[0] >= 4:
        z_vals = hand_3d[:, 2]
        z_med = float(np.median(z_vals))
        z_mad = float(np.median(np.abs(z_vals - z_med)))
        z_tol = max(0.12, 3.0 * z_mad + 0.08)
        depth_ok = np.abs(vertices_cam[:, 2] - z_med) <= z_tol
        depth_filtered = vertex_mask & depth_ok
        if np.count_nonzero(depth_filtered) >= 36:
            vertex_mask = depth_filtered

    if np.count_nonzero(vertex_mask) < 24:
        return vertices_cam, faces, None
    return _compact_mesh_from_vertex_mask(vertices_cam, faces, vertex_mask, min_faces=min_faces)


def extract_hand_mesh_from_bbox(
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    focal_length: float,
    image_shape: tuple[int, int, int],
    bbox_xyxy: np.ndarray,
    *,
    expand_ratio: float = 0.35,
    min_faces: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Carve the hand sub-mesh by cropping projected vertices to an expanded bbox (keypoint fallback)."""
    if vertices_cam.size == 0 or faces.size == 0:
        return vertices_cam, faces, None

    h, w = image_shape[:2]
    bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
    if bbox.shape[0] != 4:
        return vertices_cam, faces, None

    x1, y1, x2, y2 = bbox.tolist()
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    x_pad = bw * float(max(expand_ratio, 0.0))
    y_pad = bh * float(max(expand_ratio, 0.0))

    x1e = max(0.0, x1 - x_pad)
    y1e = max(0.0, y1 - y_pad)
    x2e = min(float(w - 1), x2 + x_pad)
    y2e = min(float(h - 1), y2 + y_pad)
    if x2e <= x1e or y2e <= y1e:
        return vertices_cam, faces, None

    u, v, valid = project_mesh_to_image(vertices_cam, focal_length, image_shape)
    in_box = (
        valid
        & (u >= x1e)
        & (u <= x2e)
        & (v >= y1e)
        & (v <= y2e)
    )
    if not np.any(in_box):
        return vertices_cam, faces, None

    return _compact_mesh_from_vertex_mask(vertices_cam, faces, in_box, min_faces=min_faces)


def _project_visible_triangles(
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    focal_length: float,
    image_shape: tuple[int, int, int],
    face_stride: int,
    margin: float = 5.0,
) -> np.ndarray:
    """Project (every ``face_stride``-th) triangle to integer image points, keeping in-view ones."""
    u, v, valid = project_mesh_to_image(vertices_cam, focal_length, image_shape)
    sampled_faces = faces[:: max(face_stride, 1)]
    if sampled_faces.size == 0:
        return np.empty((0, 3, 2), dtype=np.int32)

    face_valid = valid[sampled_faces].all(axis=1)
    sampled_faces = sampled_faces[face_valid]
    if sampled_faces.size == 0:
        return np.empty((0, 3, 2), dtype=np.int32)

    tri_u = u[sampled_faces]
    tri_v = v[sampled_faces]
    h, w = image_shape[:2]
    in_view = (
        (tri_u.max(axis=1) >= -margin)
        & (tri_u.min(axis=1) <= (w + margin))
        & (tri_v.max(axis=1) >= -margin)
        & (tri_v.min(axis=1) <= (h + margin))
    )
    if not np.any(in_view):
        return np.empty((0, 3, 2), dtype=np.int32)

    tri_pts = np.stack([tri_u[in_view], tri_v[in_view]], axis=2)
    return np.round(tri_pts).astype(np.int32)


def draw_mesh_overlay(
    frame_bgr: np.ndarray,
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    focal_length: float,
    face_stride: int,
) -> np.ndarray:
    """Draw the projected mesh wireframe over a copy of the frame."""
    overlay = frame_bgr.copy()
    tri_pts = _project_visible_triangles(
        vertices_cam=vertices_cam,
        faces=faces,
        focal_length=focal_length,
        image_shape=frame_bgr.shape,
        face_stride=face_stride,
    )
    for pts_i in tri_pts:
        cv2.polylines(
            overlay,
            [pts_i],
            isClosed=True,
            color=(80, 220, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return overlay


def draw_mesh_overlay_and_mask(
    frame_bgr: np.ndarray,
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    focal_length: float,
    face_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the filled+outlined mesh overlay on the frame and return (overlay, binary mask)."""
    overlay = frame_bgr.copy()
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    tri_pts = _project_visible_triangles(
        vertices_cam=vertices_cam,
        faces=faces,
        focal_length=focal_length,
        image_shape=frame_bgr.shape,
        face_stride=face_stride,
    )
    for pts_i in tri_pts:
        cv2.fillConvexPoly(mask, pts_i, 255, lineType=cv2.LINE_AA)

    if np.any(mask):
        in_mask = mask > 0
        overlay_color = np.array([95, 228, 255], dtype=np.float32)
        alpha = 0.44
        overlay[in_mask] = (
            overlay[in_mask].astype(np.float32) * (1.0 - alpha) + overlay_color * alpha
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Dense vertex dots keep the "full mesh" visual with much lower CPU cost.
        u, v, valid = project_mesh_to_image(vertices_cam, focal_length, frame_bgr.shape)
        ui = np.round(u[valid]).astype(np.int32)
        vi = np.round(v[valid]).astype(np.int32)
        h, w = frame_bgr.shape[:2]
        in_bounds = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        overlay[vi[in_bounds], ui[in_bounds]] = (60, 210, 255)

    return overlay, mask


def rasterize_mesh_mask(
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    focal_length: float,
    image_shape: tuple[int, int, int],
    face_stride: int,
) -> np.ndarray:
    """Rasterise the projected mesh into a binary silhouette mask."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    u, v, valid = project_mesh_to_image(vertices_cam, focal_length, image_shape)
    sampled_faces = faces[:: max(face_stride, 1)]

    for tri in sampled_faces:
        if not (valid[tri[0]] and valid[tri[1]] and valid[tri[2]]):
            continue
        pts = np.round(
            np.array(
                [
                    [u[tri[0]], v[tri[0]]],
                    [u[tri[1]], v[tri[1]]],
                    [u[tri[2]], v[tri[2]]],
                ],
                dtype=np.float32,
            )
        ).astype(np.int32)
        if np.all(pts[:, 0] < 0) or np.all(pts[:, 0] >= w):
            continue
        if np.all(pts[:, 1] < 0) or np.all(pts[:, 1] >= h):
            continue
        cv2.fillConvexPoly(mask, pts, 255)

    return mask


def draw_segmentation_panel(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tinted+outlined segmentation panel for the 2x2 preview's top-right tile."""
    panel = frame_bgr.copy()
    color = np.zeros_like(panel)
    color[:] = (40, 220, 80)
    alpha = 0.35

    in_mask = mask > 0
    panel[in_mask] = (
        panel[in_mask].astype(np.float32) * (1.0 - alpha)
        + color[in_mask].astype(np.float32) * alpha
    ).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, (20, 255, 20), 2, lineType=cv2.LINE_AA)
    cv2.putText(
        panel,
        "Segmentation mask",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 255, 20),
        2,
        cv2.LINE_AA,
    )
    return panel


def _camera_to_debug_world(points_cam: np.ndarray, mode: str = "body") -> np.ndarray:
    """Rotate SAM-3D camera-space points into the debug-world frame (per body/hand convention)."""
    # Camera coordinates (SAM-3D): +X right, +Y down, +Z away from camera.
    # body mode: +Y right, +Z up, +X towards camera.
    # hand mode: +X right, +Z up, +Y depth (closer visual orientation to input video).
    p = np.asarray(points_cam, dtype=np.float32)
    world = np.empty_like(p, dtype=np.float32)
    if str(mode).strip().lower() == "hand":
        world[:, 0] = p[:, 0]   # X: right
        world[:, 1] = p[:, 2]   # Y: depth
        world[:, 2] = -p[:, 1]  # Z: up
    else:
        world[:, 0] = -p[:, 2]  # X: towards camera
        world[:, 1] = p[:, 0]   # Y: right
        world[:, 2] = -p[:, 1]  # Z: up
    return world


def _project_debug_world(
    points_xyz: np.ndarray,
    depth_skew_x: float = 0.20,
    depth_skew_y: float = 0.08,
) -> np.ndarray:
    """Camera-aligned oblique 3D->2D projection that preserves orientation while cueing depth."""
    # Camera-aligned oblique projection:
    # - preserves orientation with the original camera in screen Y/Z,
    # - keeps depth cue from X (towards camera) without rotating the person.
    p = np.asarray(points_xyz, dtype=np.float32)
    out = np.empty((p.shape[0], 2), dtype=np.float32)
    out[:, 0] = p[:, 1] - depth_skew_x * p[:, 0]   # screen x
    out[:, 1] = -p[:, 2] + depth_skew_y * p[:, 0]  # screen y (down positive)
    return out


def _alpha_fill_triangle(
    canvas: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Alpha-blend a filled triangle, blending only its bounding-box ROI for speed."""
    if alpha >= 0.99:
        cv2.fillConvexPoly(canvas, pts, color, lineType=cv2.LINE_AA)
        return
    # Critical perf path: blend only triangle ROI, not the full frame.
    h, w = canvas.shape[:2]
    x, y, bw, bh = cv2.boundingRect(pts)
    if bw <= 0 or bh <= 0:
        return

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + bw)
    y1 = min(h, y + bh)
    if x1 <= x0 or y1 <= y0:
        return

    roi = canvas[y0:y1, x0:x1]
    tri = pts.astype(np.int32).copy()
    tri[:, 0] -= x0
    tri[:, 1] -= y0

    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, tri, 255, lineType=cv2.LINE_AA)
    if not np.any(mask):
        return

    # Keep AA edges by using mask intensity (0..255) as per-pixel weight.
    weight = (mask.astype(np.float32) / 255.0) * float(alpha)
    roi_f = roi.astype(np.float32)
    color_f = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    roi_f = roi_f * (1.0 - weight[..., None]) + color_f * weight[..., None]
    roi[:] = np.clip(roi_f, 0.0, 255.0).astype(np.uint8)


def _draw_ground_grid_and_axes(
    canvas: np.ndarray,
    project_points_fn,
    scale: float,
    center_x: float,
    center_y: float,
    *,
    draw_grid: bool = True,
    draw_axes: bool = True,
) -> None:
    """Draw the floor grid (z=0) and/or the coloured XYZ axes onto the 3D space-view canvas."""
    depth_extent = 1.2
    lateral_extent = 1.2
    grid_step = 0.2
    x_vals = np.arange(-depth_extent, depth_extent + 1e-6, grid_step, dtype=np.float32)
    y_vals = np.arange(-lateral_extent, lateral_extent + 1e-6, grid_step, dtype=np.float32)

    # Ground grid on plane z=0 (world up axis).
    if draw_grid:
        for yv in y_vals:
            p0 = np.array([[-depth_extent, yv, 0.0]], dtype=np.float32)
            p1 = np.array([[depth_extent, yv, 0.0]], dtype=np.float32)
            t0 = project_points_fn(p0)[0]
            t1 = project_points_fn(p1)[0]
            pt0 = (int(center_x + t0[0] * scale), int(center_y + t0[1] * scale))
            pt1 = (int(center_x + t1[0] * scale), int(center_y + t1[1] * scale))
            cv2.line(canvas, pt0, pt1, (220, 220, 220), 1, lineType=cv2.LINE_AA)
        for xv in x_vals:
            p0 = np.array([[xv, -lateral_extent, 0.0]], dtype=np.float32)
            p1 = np.array([[xv, lateral_extent, 0.0]], dtype=np.float32)
            t0 = project_points_fn(p0)[0]
            t1 = project_points_fn(p1)[0]
            pt0 = (int(center_x + t0[0] * scale), int(center_y + t0[1] * scale))
            pt1 = (int(center_x + t1[0] * scale), int(center_y + t1[1] * scale))
            cv2.line(canvas, pt0, pt1, (225, 225, 225), 1, lineType=cv2.LINE_AA)

    if draw_axes:
        # Axes at origin (X-red, Y-green, Z-blue).
        origin = project_points_fn(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))[0]
        x_end = project_points_fn(np.array([[0.45, 0.0, 0.0]], dtype=np.float32))[0]
        y_end = project_points_fn(np.array([[0.0, 0.45, 0.0]], dtype=np.float32))[0]
        z_end = project_points_fn(np.array([[0.0, 0.0, 0.55]], dtype=np.float32))[0]
        o = (int(center_x + origin[0] * scale), int(center_y + origin[1] * scale))
        px = (int(center_x + x_end[0] * scale), int(center_y + x_end[1] * scale))
        py = (int(center_x + y_end[0] * scale), int(center_y + y_end[1] * scale))
        pz = (int(center_x + z_end[0] * scale), int(center_y + z_end[1] * scale))
        cv2.line(canvas, o, px, (30, 30, 220), 3, lineType=cv2.LINE_AA)
        cv2.line(canvas, o, py, (30, 180, 30), 3, lineType=cv2.LINE_AA)
        cv2.line(canvas, o, pz, (220, 80, 30), 3, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "X", (px[0] + 4, px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Y", (py[0] + 4, py[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 180, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Z", (pz[0] + 4, pz[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 80, 30), 1, cv2.LINE_AA)


def render_3d_space_view(
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    image_h: int,
    image_w: int,
    face_stride: int,
    joints_3d_cam: np.ndarray | None = None,
    title: str = "3D mesh space view",
    mesh_alpha: float = 0.88,
    level_joints: bool = False,
    world_anchor: dict[str, float] | None = None,
    lock_world_anchor: bool = False,
    view_state: dict[str, float] | None = None,
    lock_view_state: bool = False,
    view_zoom: float = 1.18,
    enforce_ground_contact: bool = True,
    ground_contact_auto: bool = True,
    ground_contact_auto_calib_frames: int = 45,
    ground_contact_quantile: float = 1.5,
    ground_contact_smoothing: float = 0.35,
    camera_align_mode: str = "body",
) -> np.ndarray:
    """Render the offline 3D space-view panel: floored, centred mesh with ground grid and axes.

    Normalises the subject onto the floor plane, optionally locks the world anchor
    / view transform (so the camera stays fixed across frames) and auto-calibrates
    the ground-contact quantile. Returns a rendered BGR canvas.
    """
    canvas = np.full((image_h, image_w, 3), 255, dtype=np.uint8)

    v_cam = vertices_cam.copy()
    j_cam = joints_3d_cam.copy() if joints_3d_cam is not None else None

    # Normalize world position: feet on the floor plane and body centered in camera X/Z.
    # In SAM-3D camera space, +Y points downward (same sign as image Y),
    # so the floor is near the high Y percentile, not the low one.
    if j_cam is not None and j_cam.size > 0:
        floor_y_est = float(np.percentile(j_cam[:, 1], 97.0))
        center_x_est = float(np.mean(j_cam[:, 0]))
        center_z_est = float(np.mean(j_cam[:, 2]))
    elif v_cam.size > 0:
        floor_y_est = float(np.percentile(v_cam[:, 1], 97.0))
        center_x_est = float(np.mean(v_cam[:, 0]))
        center_z_est = float(np.mean(v_cam[:, 2]))
    else:
        floor_y_est = 0.0
        center_x_est = 0.0
        center_z_est = 0.0

    if lock_world_anchor:
        if (
            world_anchor is not None
            and "floor_y" in world_anchor
            and "center_x" in world_anchor
            and "center_z" in world_anchor
        ):
            floor_y = float(world_anchor["floor_y"])
            center_x = float(world_anchor["center_x"])
            center_z = float(world_anchor["center_z"])
        else:
            floor_y = floor_y_est
            center_x = center_x_est
            center_z = center_z_est
            if world_anchor is not None:
                world_anchor["floor_y"] = floor_y
                world_anchor["center_x"] = center_x
                world_anchor["center_z"] = center_z
    else:
        floor_y = floor_y_est
        center_x = center_x_est
        center_z = center_z_est
        if world_anchor is not None:
            world_anchor["floor_y"] = floor_y
            world_anchor["center_x"] = center_x
            world_anchor["center_z"] = center_z

    v_cam[:, 1] -= floor_y
    v_cam[:, 0] -= center_x
    v_cam[:, 2] -= center_z
    if j_cam is not None:
        j_cam[:, 1] -= floor_y
        j_cam[:, 0] -= center_x
        j_cam[:, 2] -= center_z

    # Camera-aligned debug world.
    align_mode = "hand" if str(camera_align_mode).strip().lower() == "hand" else "body"
    v = _camera_to_debug_world(v_cam, mode=align_mode)
    j = _camera_to_debug_world(j_cam, mode=align_mode) if j_cam is not None else None

    if v.shape[0] == 0 and (j is None or j.shape[0] == 0):
        cv2.putText(
            canvas,
            title,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (50, 50, 50),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "No mesh to display on this frame",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (90, 90, 90),
            2,
            cv2.LINE_AA,
        )
        return canvas

    if enforce_ground_contact and v.shape[0] > 0:
        q_default = float(np.clip(ground_contact_quantile, 0.0, 12.0))
        q_effective = q_default
        auto_enabled = bool(ground_contact_auto and lock_world_anchor and world_anchor is not None)
        if auto_enabled and world_anchor is not None:
            auto_state = world_anchor.get("_ground_contact_auto_state")
            if not isinstance(auto_state, dict):
                candidates = [0.5, 1.0, 1.5, 2.5, 3.5]
                auto_state = {
                    "frame_count": 0,
                    "candidates": candidates,
                    "prev_shift": [None] * len(candidates),
                    "jitter_sum": [0.0] * len(candidates),
                    "chosen_q": None,
                }

            chosen_q = auto_state.get("chosen_q")
            candidates = auto_state.get("candidates")
            if isinstance(chosen_q, (int, float)) and np.isfinite(chosen_q):
                q_effective = float(np.clip(float(chosen_q), 0.0, 12.0))
            elif (
                isinstance(candidates, list)
                and len(candidates) > 0
                and all(isinstance(c, (int, float)) for c in candidates)
            ):
                prev_shift = auto_state.get("prev_shift")
                jitter_sum = auto_state.get("jitter_sum")
                if (
                    not isinstance(prev_shift, list)
                    or not isinstance(jitter_sum, list)
                    or len(prev_shift) != len(candidates)
                    or len(jitter_sum) != len(candidates)
                ):
                    prev_shift = [None] * len(candidates)
                    jitter_sum = [0.0] * len(candidates)

                for idx, q_cand in enumerate(candidates):
                    qv = float(np.clip(float(q_cand), 0.0, 12.0))
                    target_shift_q = -float(np.percentile(v[:, 2], qv))
                    prev_q = prev_shift[idx]
                    if isinstance(prev_q, (int, float)) and np.isfinite(prev_q):
                        jitter_sum[idx] = float(jitter_sum[idx]) + abs(target_shift_q - float(prev_q))
                    prev_shift[idx] = float(target_shift_q)

                frame_count = int(auto_state.get("frame_count", 0)) + 1
                auto_state["frame_count"] = frame_count
                auto_state["prev_shift"] = prev_shift
                auto_state["jitter_sum"] = jitter_sum

                calib_frames = max(8, int(ground_contact_auto_calib_frames))
                if frame_count >= calib_frames:
                    denom = max(1, frame_count - 1)
                    means = [float(js) / float(denom) for js in jitter_sum]
                    min_mean = float(np.min(means))
                    stable_idx = [
                        i for i, m in enumerate(means)
                        if m <= (min_mean * 1.35 + 1e-9)
                    ]
                    chosen_idx = min(stable_idx) if stable_idx else int(np.argmin(means))
                    chosen_q = float(np.clip(float(candidates[chosen_idx]), 0.0, 12.0))
                    auto_state["chosen_q"] = chosen_q
                    q_effective = chosen_q
                else:
                    q_effective = q_default

            world_anchor["_ground_contact_auto_state"] = auto_state
            world_anchor["_ground_contact_quantile_effective"] = float(q_effective)
            world_anchor["_ground_contact_auto_enabled"] = True
        elif world_anchor is not None:
            world_anchor["_ground_contact_quantile_effective"] = float(q_effective)
            world_anchor["_ground_contact_auto_enabled"] = False

        foot_z_now = float(np.percentile(v[:, 2], q_effective))
        target_shift_z = -foot_z_now
        shift_z = target_shift_z

        if lock_world_anchor and world_anchor is not None:
            prev_shift = world_anchor.get("_ground_contact_shift_z")
            if isinstance(prev_shift, (int, float)) and np.isfinite(prev_shift):
                alpha = float(np.clip(ground_contact_smoothing, 0.0, 1.0))
                shift_z = (1.0 - alpha) * float(prev_shift) + alpha * target_shift_z
            world_anchor["_ground_contact_shift_z"] = float(shift_z)

        v[:, 2] += float(shift_z)
        if j is not None:
            j[:, 2] += float(shift_z)

    def project_points(points_xyz: np.ndarray) -> np.ndarray:
        """Project debug-world points with this panel's fixed oblique skew."""
        return _project_debug_world(points_xyz, depth_skew_x=0.20, depth_skew_y=0.08)

    view_points = v if v.shape[0] > 0 else j
    assert view_points is not None and view_points.shape[0] > 0
    v2d_bounds = project_points(view_points)
    xmin, xmax = float(v2d_bounds[:, 0].min()), float(v2d_bounds[:, 0].max())
    ymin, ymax = float(v2d_bounds[:, 1].min()), float(v2d_bounds[:, 1].max())
    span_x = max(xmax - xmin, 1e-4)
    span_y = max(ymax - ymin, 1e-4)

    if (
        lock_view_state
        and view_state is not None
        and "scale" in view_state
        and "center_x" in view_state
        and "center_y" in view_state
    ):
        scale = float(view_state["scale"])
        center_x = float(view_state["center_x"])
        center_y = float(view_state["center_y"])
    else:
        # Fit body while keeping room for grid/axes, then slightly zoom-in.
        scale_x = (image_w * 0.62) / span_x
        scale_y = (image_h * 0.70) / span_y
        scale = max(1e-4, min(scale_x, scale_y)) * float(max(0.2, view_zoom))

        center_x = image_w * 0.5 - ((xmin + xmax) * 0.5) * scale
        target_feet_y = image_h * 0.86 if level_joints else image_h * 0.82
        center_y = target_feet_y - ymax * scale

        if view_state is not None:
            view_state["scale"] = float(scale)
            view_state["center_x"] = float(center_x)
            view_state["center_y"] = float(center_y)

    v2d = project_points(v) if v.shape[0] > 0 else np.empty((0, 2), dtype=np.float32)
    x2d = center_x + v2d[:, 0] * scale
    y2d = center_y + v2d[:, 1] * scale

    _draw_ground_grid_and_axes(
        canvas=canvas,
        project_points_fn=project_points,
        scale=scale,
        center_x=center_x,
        center_y=center_y,
        draw_grid=True,
        draw_axes=False,
    )

    sampled_faces = faces[:: max(face_stride, 1)]
    if sampled_faces.size > 0:
        tri_x = x2d[sampled_faces]
        tri_y = y2d[sampled_faces]
        in_view = (
            (tri_x.max(axis=1) >= -10)
            & (tri_x.min(axis=1) <= image_w + 10)
            & (tri_y.max(axis=1) >= -10)
            & (tri_y.min(axis=1) <= image_h + 10)
        )
        sampled_faces = sampled_faces[in_view]
        tri_x = tri_x[in_view]
        tri_y = tri_y[in_view]

        if sampled_faces.size > 0:
            depth_mean = v[sampled_faces][:, :, 0].mean(axis=1)
            order = np.argsort(depth_mean)
            mesh_layer = np.zeros_like(canvas)
            mesh_mask = np.zeros((image_h, image_w), dtype=np.uint8)

            for idx in order:
                pts_i = np.round(
                    np.stack([tri_x[idx], tri_y[idx]], axis=1)
                ).astype(np.int32)
                depth = float(depth_mean[idx])
                tone = int(np.clip(182 + depth * 16, 90, 228))
                fill_color = (tone - 26, tone - 6, tone)
                cv2.fillConvexPoly(mesh_layer, pts_i, fill_color, lineType=cv2.LINE_AA)
                cv2.fillConvexPoly(mesh_mask, pts_i, 255, lineType=cv2.LINE_AA)

            if np.any(mesh_mask):
                mesh_pixels = mesh_mask > 0
                if mesh_alpha >= 0.99:
                    canvas[mesh_pixels] = mesh_layer[mesh_pixels]
                else:
                    alpha = float(np.clip(mesh_alpha, 0.0, 1.0))
                    canvas[mesh_pixels] = (
                        canvas[mesh_pixels].astype(np.float32) * (1.0 - alpha)
                        + mesh_layer[mesh_pixels].astype(np.float32) * alpha
                    ).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mesh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(canvas, contours, -1, (95, 95, 95), 1, lineType=cv2.LINE_AA)

    if j is not None:
        j2d = project_points(j)
        jx = center_x + j2d[:, 0] * scale
        jy = center_y + j2d[:, 1] * scale
        for xj, yj in zip(jx, jy):
            if xj < 0 or xj >= image_w or yj < 0 or yj >= image_h:
                continue
            cv2.circle(canvas, (int(xj), int(yj)), 3, (0, 70, 255), -1, lineType=cv2.LINE_AA)

    # Draw axes last so they stay visible for debugging orientation.
    _draw_ground_grid_and_axes(
        canvas=canvas,
        project_points_fn=project_points,
        scale=scale,
        center_x=center_x,
        center_y=center_y,
        draw_grid=False,
        draw_axes=True,
    )

    cv2.putText(
        canvas,
        title,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )
    return canvas


def compose_rich_2x2(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray | None = None,
) -> np.ndarray:
    """Stitch panels into the 2x2 preview grid (centre-pad the bottom row when only 3 panels)."""
    top = np.concatenate([top_left, top_right], axis=1)
    if bottom_right is None:
        # Leveled view disabled: keep geometry (no horizontal stretch).
        if bottom_left.shape[1] == top.shape[1]:
            bottom = bottom_left
        elif bottom_left.shape[1] < top.shape[1]:
            pad_total = top.shape[1] - bottom_left.shape[1]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            bottom = cv2.copyMakeBorder(
                bottom_left,
                0,
                0,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
        else:
            # Safety fallback only if caller gave a wider panel.
            bottom = cv2.resize(
                bottom_left,
                (top.shape[1], bottom_left.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
    else:
        bottom = np.concatenate([bottom_left, bottom_right], axis=1)
    return np.concatenate([top, bottom], axis=0)


def open_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
    preferred_codec: str,
    *,
    allow_mp4v_fallback: bool = True,
) -> tuple[cv2.VideoWriter, str]:
    """Open a cv2.VideoWriter, trying the preferred codec then fallbacks; raise if none work."""
    codec = preferred_codec.strip().lower()
    if codec in {"h264", "avc1"}:
        candidates = ["avc1", "h264", "H264"]
        if allow_mp4v_fallback:
            candidates.append("mp4v")
    elif codec == "mp4v":
        candidates = ["mp4v", "avc1", "h264"]
    else:
        candidates = [codec, "avc1", "h264", "mp4v"]
    candidates = list(dict.fromkeys(candidates))

    last_error = ""
    for code in candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*code),
            fps,
            frame_size,
        )
        if writer.isOpened():
            return writer, code
        writer.release()
        last_error = f"Codec '{code}' unsupported by backend."

    raise RuntimeError(
        f"Cannot create output video writer at {output_path}. "
        f"Tried codecs={candidates}. {last_error}"
    )


class RollingPreviewWriter:
    """Periodically re-encodes the last N frames into a short looping MP4 for live preview."""

    def __init__(
        self,
        output_path: Path,
        fps: float,
        frame_size: tuple[int, int],
        preferred_codec: str,
        buffer_frames: int,
        refresh_every: int = 1,
    ) -> None:
        """Set up the rolling buffer and write an initial placeholder loop."""
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()
        self.fps = fps
        self.frame_size = frame_size
        self.preferred_codec = preferred_codec
        self.buffer_frames = max(int(buffer_frames), 1)
        self.refresh_every = max(int(refresh_every), 1)
        self.codec_effective: str | None = None
        self.frames: list[np.ndarray] = []
        self.frame_count = 0
        self.update_count = 0
        self.files: list[Path] = [self.output_path]
        self._write_placeholder_loop()

    def _write_placeholder_loop(self) -> None:
        """Seed the preview file with a "waiting for first frame" placeholder."""
        w, h = self.frame_size
        placeholder = np.full((h, w, 3), 22, dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Live preview: waiting for first frame...",
            (20, max(40, h // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        self.frames = [placeholder]
        self._flush()
        self.frames = []
        self.frame_count = 0

    def _flush(self) -> None:
        """Re-encode the buffered frames to a temp file and atomically swap it in."""
        if not self.frames:
            return
        tmp_path = self.output_path.with_suffix(".tmp.mp4")
        writer, codec = open_video_writer(
            output_path=tmp_path,
            fps=self.fps,
            frame_size=self.frame_size,
            preferred_codec=self.preferred_codec,
            allow_mp4v_fallback=False,
        )
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        tmp_path.replace(self.output_path)
        self.codec_effective = codec if self.codec_effective is None else self.codec_effective
        self.update_count += 1
        if self.update_count == 1:
            print(f"Live preview ready: {self.output_path.name} (codec={codec})")

    def write(self, frame_bgr: np.ndarray) -> None:
        """Append a frame to the rolling buffer and re-flush on the refresh cadence."""
        self.frames.append(frame_bgr.copy())
        if len(self.frames) > self.buffer_frames:
            self.frames = self.frames[-self.buffer_frames :]
        self.frame_count += 1
        if self.frame_count % self.refresh_every == 0:
            self._flush()

    def close(self) -> None:
        """Flush any remaining buffered frames to the preview file."""
        self._flush()


def run_pipeline(cfg: PipelineConfig, runtime: PipelineRuntime | None = None) -> dict[str, Any]:
    """Run the full per-video pipeline: detect/track the subject, infer 3D meshes, render and export.

    Loads (or reuses) the SAM3D runtime, walks the video frame by frame applying
    identity-locked tracking (or hand temporal postprocessing), exports per-frame
    meshes + joint timeseries, writes preview/live videos, optionally runs the
    offline identity resolver, and persists run_metadata.json. Returns the
    metadata dict.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = cfg.output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    if runtime is None:
        runtime = build_pipeline_runtime(cfg)
        runtime_reused = False
    else:
        _configure_torch_runtime(cfg)
        runtime_reused = True
        print(
            "Reusing SAM3D runtime: "
            f"device={runtime.device}, "
            f"precision={'float16' if runtime.inference_dtype == torch.float16 else 'float32'}, "
            f"mhr_backend={runtime.mhr_backend}"
        )

    device = runtime.device
    inference_dtype = runtime.inference_dtype
    mps_mhr_mode = runtime.mps_mhr_mode
    estimator = runtime.estimator
    mhr_backend = runtime.mhr_backend
    faces = runtime.faces

    inference_target = normalize_inference_target(cfg.inference_target)
    auto_mode = cfg.auto_init_mode.strip().lower()
    auto_select_strategy = cfg.auto_select_strategy
    if inference_target != "body" and auto_select_strategy.strip().lower() == "patient":
        # In non-full mode, prefer tight local regions over full-person boxes.
        auto_select_strategy = "tightest"
    sam3_text_prompts = normalize_sam3_text_prompts(cfg.sam3_text_prompts)
    sam3_auto_detector = None
    needs_sam3_detector = auto_mode in {"smart", "sam3"} and (
        cfg.prompt_bbox is None
        or (inference_target == "body" and bool(cfg.identity_lock_enabled))
    )
    if needs_sam3_detector:
        sam3_auto_detector = try_build_human_detector(
            detector_name="sam3",
            device=device,
            sam3_code_root=cfg.sam3_code_root,
        )
        if sam3_auto_detector is not None:
            print(
                f"SAM3 target={inference_target} text prompts: "
                f"{', '.join(sam3_text_prompts)}"
            )

    cap = cv2.VideoCapture(str(cfg.video_input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.video_input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = cfg.output_dir / f"{cfg.video_input.stem}_processed.mp4"
    # High-quality mode: preserve source FPS and process frame-by-frame.
    out_fps = fps
    effective_frame_step = 1
    if cfg.frame_step != 1:
        print(
            f"Requested frame_step={cfg.frame_step} ignored to preserve source FPS and quality."
        )
    writer = None
    output_codec_effective: str | None = None
    if cfg.render_preview:
        writer, output_codec_effective = open_video_writer(
            output_path=out_video_path,
            fps=out_fps,
            frame_size=(width * 2, height * 2),
            preferred_codec=cfg.output_codec,
        )
        print(f"Video writer codec: requested={cfg.output_codec}, effective={output_codec_effective}")

    live_preview_panel = cfg.live_preview_panel.strip().lower()
    if live_preview_panel not in {"top-left", "full"}:
        live_preview_panel = "top-left"
    live_preview_writer: RollingPreviewWriter | None = None
    live_preview_dir: Path | None = None
    live_preview_path: Path | None = None
    if cfg.live_preview:
        live_preview_dir = cfg.output_dir / "live_preview"
        live_preview_path = live_preview_dir / "live_preview_loop.mp4"
        live_frame_size = (
            (width, height) if live_preview_panel == "top-left" else (width * 2, height * 2)
        )
        live_preview_writer = RollingPreviewWriter(
            output_path=live_preview_path,
            fps=out_fps,
            frame_size=live_frame_size,
            preferred_codec=cfg.live_preview_codec,
            buffer_frames=cfg.live_preview_chunk_frames,
            refresh_every=cfg.live_preview_refresh_every,
        )

    manual_subject_bboxes: dict[int, np.ndarray] = {}
    manual_subject_tracking_info: dict[str, Any] = {
        "enabled": False,
        "reason": "no_prompt_bbox_frame",
    }
    if cfg.subject_track_file:
        # The chosen-subject track is already dense and is fed as hard
        # tracking_anchors (anchor_track), so the expensive LK optical-flow guide
        # is redundant — skip it.
        manual_subject_tracking_info = {
            "enabled": False,
            "reason": "subject_track_anchors",
            "source": cfg.subject_track_file,
        }
    elif cfg.prompt_bbox is not None and cfg.prompt_bbox_frame is not None:
        tracking_build_start = time.perf_counter()
        prompt_bbox_clipped = clip_bbox(cfg.prompt_bbox, (height, width, 3))
        manual_subject_bboxes, manual_subject_tracking_info = build_manual_subject_bbox_track(
            cfg.video_input,
            anchor_frame=int(cfg.prompt_bbox_frame),
            anchor_bbox=prompt_bbox_clipped,
            total_frames=total_frames,
            width=width,
            height=height,
        )
        manual_subject_tracking_info["build_seconds"] = (
            time.perf_counter() - tracking_build_start
        )
        if manual_subject_bboxes:
            print(
                "Manual subject guide: "
                f"anchor={manual_subject_tracking_info.get('anchor_frame_effective')} "
                f"tracked_frames={len(manual_subject_bboxes)}"
            )

    start_frame = max(cfg.start_frame, 0)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Target frame count the loop will actually process: from start_frame to EOF,
    # honoring the effective step (forced to 1) and the optional max_frames cap.
    # Persisted into run_metadata.json so the viewer can show a real percentage.
    frames_after_start = max(0, total_frames - start_frame)
    stepped_total = (frames_after_start + effective_frame_step - 1) // effective_frame_step
    progress_total_target = (
        min(stepped_total, cfg.max_frames) if cfg.max_frames else stepped_total
    ) or None

    frame_idx = start_frame - 1
    processed_idx = 0
    patient_bbox = (
        manual_subject_bboxes.get(start_frame)
        if manual_subject_bboxes
        else (cfg.prompt_bbox.copy() if cfg.prompt_bbox is not None else None)
    )
    auto_init_attempted = False
    # While no subject has appeared yet, re-run the heavy SAM3 auto-init detector
    # only every few frames instead of every frame — otherwise a long empty intro
    # (e.g. a corridor before the patient walks in) burns minutes hunting nobody.
    auto_init_retry_every = max(1, int(cfg.identity_reacquire_when_lost_every_n))
    frames_since_auto_init = 0
    auto_init_result: dict[str, Any] = {
        "mode": (
            "manual_prompt_bbox_track"
            if manual_subject_bboxes
            else ("manual_prompt_bbox" if cfg.prompt_bbox is not None else auto_mode)
        ),
        "used": cfg.prompt_bbox is not None,
        "sam3_text_prompts": list(sam3_text_prompts),
        "selected_source": "manual" if cfg.prompt_bbox is not None else None,
        "selected_score": None,
        "num_candidates": 0,
        "fallback_used": False,
        "prompt_bbox_frame": (
            int(cfg.prompt_bbox_frame) if cfg.prompt_bbox_frame is not None else None
        ),
        "manual_subject_tracking": manual_subject_tracking_info,
    }
    records: list[dict[str, Any]] = []
    # Per-detector-frame log for the offline (non-causal) identity resolver: the
    # candidate boxes + their frozen-gallery affinity, tied to each frame's
    # record index. Replayed after the forward pass to suppress greedy teleports.
    identity_trace: list[dict[str, Any]] = []
    timing_infer_s = 0.0
    timing_mesh_export_s = 0.0
    timing_render_s = 0.0
    timing_tracking_s = 0.0
    timing_reacquire_detect_s = 0.0
    bbox_smoothing_alpha_values: list[float] = []
    bbox_alpha_slow = float(np.clip(cfg.bbox_smoothing_alpha_slow, 0.0, 1.0))
    bbox_alpha_fast = float(np.clip(cfg.bbox_smoothing_alpha_fast, 0.0, 1.0))
    if bbox_alpha_fast < bbox_alpha_slow:
        bbox_alpha_slow, bbox_alpha_fast = bbox_alpha_fast, bbox_alpha_slow
    bbox_fast_motion_ratio = max(1e-4, float(cfg.bbox_smoothing_fast_motion_ratio))
    fixed_world_anchor: dict[str, float] = {}
    fixed_view_state: dict[str, float] = {}
    identity_tracker: IdentityLockedBboxTracker | None = None
    hand_temporal: HandTemporalPostprocessor | None = None
    identity_detect_every_n = max(1, int(cfg.identity_detect_every_n))

    # All user-signalled sightings become hard re-assertion points (consumed in
    # order as we reach their frame), not just the single median anchor. Anchor
    # frame indices share the processed video_frame coordinate system; sightings
    # outside the processed range are simply never reached and ignored.
    anchor_track: list[tuple[int, np.ndarray]] = []
    if cfg.tracking_anchors and inference_target != "hand":
        for _a in cfg.tracking_anchors:
            try:
                _fi = int(_a.get("frameIndex"))
                _bb = _a.get("bbox")
            except (TypeError, ValueError):
                continue
            if _bb is None:
                continue
            anchor_track.append(
                (_fi, clip_bbox(np.asarray(_bb, dtype=np.float32), (height, width, 3)))
            )
        anchor_track.sort(key=lambda item: item[0])
        if anchor_track:
            print(f"Subject anchors armed: {len(anchor_track)} re-assertion point(s)")
    next_anchor_i = 0

    interrupted = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if effective_frame_step > 1 and frame_idx % effective_frame_step != 0:
                continue
            if cfg.max_frames is not None and processed_idx >= cfg.max_frames:
                break

            # Masked span: keep the frame in the output at its original timing
            # but skip all inference. Record it as subject-absent ("masked") so
            # the viewer can label it, and write a placeholder preview panel so
            # the preview video stays aligned with the frame timeline.
            if frame_is_masked(frame_idx, fps, cfg.mask_time_ranges):
                if cfg.render_preview or live_preview_writer is not None:
                    masked_panel = draw_masked_overlay_panel(frame)
                    black = np.zeros((height, width, 3), dtype=np.uint8)
                    masked_rich = compose_rich_2x2(
                        masked_panel, black, black.copy(), bottom_right=None
                    )
                    if cfg.render_preview and writer is not None:
                        writer.write(masked_rich)
                    if live_preview_writer is not None:
                        live_preview_writer.write(
                            masked_rich if live_preview_panel == "full" else masked_panel
                        )
                append_subject_absent_record(
                    records,
                    frame_idx=frame_idx,
                    patient_bbox=None,
                    reason="masked",
                )
                processed_idx += 1
                continue

            manual_guided_bbox = manual_subject_bboxes.get(frame_idx)
            if manual_guided_bbox is not None:
                patient_bbox = manual_guided_bbox.copy()

            if patient_bbox is None:
                if auto_mode != "off" and not auto_init_attempted:
                    detected_bbox, info = auto_initialize_patient_bbox(
                        frame_bgr=frame,
                        auto_init_mode=auto_mode,
                        auto_select_strategy=auto_select_strategy,
                        auto_detector_threshold=cfg.auto_detector_threshold,
                        sam3_text_prompts=sam3_text_prompts,
                        sam3_detector=sam3_auto_detector,
                    )
                    auto_init_attempted = True
                    auto_init_result.update(info)
                    if detected_bbox is not None:
                        patient_bbox = detected_bbox
                        auto_init_result["used"] = True
                        print(
                            "Auto-initialized patient bbox "
                            f"from {info.get('selected_source')} candidates="
                            f"{info.get('num_candidates')}"
                        )
                if patient_bbox is None:
                    if auto_mode == "off":
                        patient_bbox = default_center_bbox(frame.shape)
                        auto_init_result["fallback_used"] = True
                    else:
                        # Throttle the auto-init hunt (see auto_init_retry_every):
                        # only re-arm the detector every few empty frames.
                        frames_since_auto_init += 1
                        if frames_since_auto_init >= auto_init_retry_every:
                            auto_init_attempted = False
                            frames_since_auto_init = 0
                        append_subject_absent_record(
                            records,
                            frame_idx=frame_idx,
                            patient_bbox=None,
                            reason="subject_not_initialized",
                        )
                        processed_idx += 1
                        if processed_idx == 1 or processed_idx % 10 == 0:
                            write_progress_metadata(
                                cfg.output_dir,
                                video_input=cfg.video_input,
                                mesh_dir=mesh_dir,
                                records=records,
                                fps_output=out_fps,
                                video_width=width,
                                video_height=height,
                                inference_target=inference_target,
                                space_view={
                                    "mode": "fixed_world_anchor",
                                    "world_anchor": (
                                        fixed_world_anchor if fixed_world_anchor else None
                                    ),
                                    "view_state": fixed_view_state if fixed_view_state else None,
                                    "view_zoom": 1.18,
                                },
                                total_frames_target=progress_total_target,
                                processing_status="running",
                            )
                        continue
            patient_bbox = clip_bbox(patient_bbox, frame.shape)
            if identity_tracker is None:
                identity_jump_ratio = float(cfg.identity_max_center_jump_ratio)
                identity_min_app = float(cfg.identity_min_appearance_similarity)
                identity_reacq_min = float(cfg.identity_reacquire_min_similarity)
                identity_reacq_every = int(cfg.identity_reacquire_every_n)
                identity_gallery_floor = float(cfg.identity_gallery_floor)
                identity_coast_floor = float(cfg.identity_coast_gallery_floor)
                identity_absence_patience = int(cfg.identity_absence_patience)
                gallery_seeds: list[np.ndarray] = []
                if inference_target == "hand":
                    # Hand appearance histograms are unstable frame-to-frame;
                    # rely on motion gating + hand temporal postprocessing, and
                    # keep the appearance gallery/absence machinery disabled.
                    identity_jump_ratio = max(identity_jump_ratio, 0.85)
                    identity_min_app = 0.0
                    identity_reacq_min = 0.0
                    identity_reacq_every = max(identity_reacq_every, 6)
                    identity_gallery_floor = 0.0
                    identity_coast_floor = 0.0
                    identity_absence_patience = 10 ** 6
                else:
                    seed_hist = extract_bbox_appearance_hist(frame, patient_bbox)
                    if seed_hist is not None:
                        gallery_seeds.append(seed_hist)
                identity_tracker = IdentityLockedBboxTracker(
                    initial_bbox=patient_bbox,
                    frame_shape=frame.shape,
                    enabled=cfg.identity_lock_enabled,
                    warmup_frames=cfg.identity_warmup_frames,
                    max_center_jump_ratio=identity_jump_ratio,
                    min_appearance_similarity=identity_min_app,
                    reacquire_min_similarity=identity_reacq_min,
                    reacquire_every_n=identity_reacq_every,
                    reacquire_when_lost_every_n=int(cfg.identity_reacquire_when_lost_every_n),
                    max_hold_frames=cfg.identity_max_hold_frames,
                    gallery_floor=identity_gallery_floor,
                    coast_gallery_floor=identity_coast_floor,
                    absence_patience=identity_absence_patience,
                    gallery_max_size=int(cfg.identity_gallery_max_size),
                    gallery_seeds=gallery_seeds,
                )
                if cfg.identity_lock_enabled:
                    print(
                        "Identity lock enabled: "
                        f"warmup={cfg.identity_warmup_frames}, "
                        f"jump_ratio={identity_jump_ratio:.3f}, "
                        f"detect_every={int(cfg.identity_detect_every_n)}, "
                        f"gallery_floor={identity_gallery_floor:.2f}"
                    )
            if inference_target == "hand" and hand_temporal is None:
                hand_temporal = HandTemporalPostprocessor(
                    frame_shape=frame.shape,
                    enabled=cfg.hand_temporal_enabled,
                    occlusion_hold_frames=cfg.hand_occlusion_hold_frames,
                    interpolation_max_gap=cfg.hand_interpolation_max_gap,
                    reentry_blend_frames=cfg.hand_reentry_blend_frames,
                    max_center_jump_ratio=cfg.hand_drift_max_center_jump_ratio,
                    min_iou=cfg.hand_drift_min_iou,
                    max_area_ratio=cfg.hand_drift_max_area_ratio,
                    bbox_smoothing_alpha=cfg.hand_bbox_smoothing_alpha,
                    hold_follow_alpha=cfg.hand_hold_follow_alpha,
                    mesh_smoothing_alpha=cfg.hand_mesh_smoothing_alpha,
                )
                print(
                    "Hand temporal postprocess: "
                    f"enabled={cfg.hand_temporal_enabled}, "
                    f"hold={cfg.hand_occlusion_hold_frames}, "
                    f"reentry={cfg.hand_reentry_blend_frames}, "
                    f"jump_ratio={cfg.hand_drift_max_center_jump_ratio:.3f}"
                )

            infer_start = time.perf_counter()
            try:
                output = infer_single_person_from_bbox(
                    estimator,
                    frame,
                    patient_bbox,
                    inference_dtype=inference_dtype,
                    inference_target=inference_target,
                )
            except RuntimeError as exc:
                err = str(exc)
                fp16_unsupported = (
                    inference_dtype == torch.float16
                    and (
                        "not implemented for 'Half'" in err
                        or "doesn't support dtype Half" in err
                        or "float16" in err.lower()
                        or "fp16" in err.lower()
                    )
                )
                recovered_from_fp16 = False
                if fp16_unsupported:
                    print(
                        "float16 op unsupported at runtime. "
                        "Falling back to float32 for stability..."
                    )
                    inference_dtype = torch.float32
                    output = infer_single_person_from_bbox(
                        estimator,
                        frame,
                        patient_bbox,
                        inference_dtype=inference_dtype,
                        inference_target=inference_target,
                    )
                    runtime.inference_dtype = inference_dtype
                    recovered_from_fp16 = True

                if not recovered_from_fp16:
                    needs_cpu_fallback = (
                        device.type == "mps"
                        and "MPS framework doesn't support float64" in err
                    )
                    if not needs_cpu_fallback:
                        raise
                    if mhr_backend == "native_mps_patched":
                        if mps_mhr_mode == "native":
                            raise RuntimeError(
                                "Strict native MPS mode failed on an unsupported float64 op. "
                                "CPU fallback is disabled; use SAM3D_MHR_MODE=auto|wrapper to allow fallback."
                            ) from exc
                        print(
                            "Native MPS MHR path hit float64 op at runtime. "
                            "Falling back to CPU wrapper on MPS..."
                        )
                        estimator = load_estimator(
                            checkpoint_path=cfg.checkpoint_path,
                            mhr_path=cfg.mhr_path,
                            device=device,
                            mps_mhr_mode="wrapper",
                        )
                        mhr_backend = getattr(estimator, "mhr_backend", "unknown")
                        faces = estimator.faces
                        runtime.device = device
                        runtime.inference_dtype = inference_dtype
                        runtime.mps_mhr_mode = "wrapper"
                        runtime.estimator = estimator
                        runtime.mhr_backend = mhr_backend
                        runtime.faces = faces
                        output = infer_single_person_from_bbox(
                            estimator,
                            frame,
                            patient_bbox,
                            inference_dtype=inference_dtype,
                            inference_target=inference_target,
                        )
                    else:
                        print(
                            "MPS inference unsupported by MHR TorchScript (float64 op). "
                            "Falling back to CPU..."
                        )
                        device = torch.device("cpu")
                        inference_dtype = torch.float32
                        estimator = load_estimator(
                            checkpoint_path=cfg.checkpoint_path,
                            mhr_path=cfg.mhr_path,
                            device=device,
                        )
                        mhr_backend = getattr(estimator, "mhr_backend", "unknown")
                        faces = estimator.faces
                        runtime.device = device
                        runtime.inference_dtype = inference_dtype
                        runtime.mps_mhr_mode = mps_mhr_mode
                        runtime.estimator = estimator
                        runtime.mhr_backend = mhr_backend
                        runtime.faces = faces
                        output = infer_single_person_from_bbox(
                            estimator,
                            frame,
                            patient_bbox,
                            inference_dtype=inference_dtype,
                            inference_target=inference_target,
                        )
            timing_infer_s += time.perf_counter() - infer_start

            if output is None:
                print(f"No SAM 3D Body output for frame {frame_idx}; skipping frame.")
                append_subject_absent_record(
                    records,
                    frame_idx=frame_idx,
                    patient_bbox=patient_bbox,
                    reason="no_output",
                )
                processed_idx += 1
                if processed_idx == 1 or processed_idx % 10 == 0:
                    write_progress_metadata(
                        cfg.output_dir,
                        video_input=cfg.video_input,
                        mesh_dir=mesh_dir,
                        records=records,
                        fps_output=out_fps,
                        video_width=width,
                        video_height=height,
                        inference_target=inference_target,
                        space_view={
                            "mode": "fixed_world_anchor",
                            "world_anchor": (
                                fixed_world_anchor if fixed_world_anchor else None
                            ),
                            "view_state": fixed_view_state if fixed_view_state else None,
                            "view_zoom": 1.18,
                        },
                        total_frames_target=progress_total_target,
                        processing_status="running",
                    )
                continue

            vertices_cam = output["pred_vertices"] + output["pred_cam_t"][None, :]
            joints_cam = output["pred_keypoints_3d"] + output["pred_cam_t"][None, :]
            focal_length = float(output["focal_length"])

            # Camera is fixed in this project, so the world-space and
            # camera-space positions coincide. Keep the *_space_cam names so
            # downstream consumers (mesh tracker, manifest, viewer) don't
            # need to be renamed.
            vertices_space_cam = vertices_cam
            joints_space_cam = joints_cam

            display_vertices_cam = vertices_cam
            display_vertices_space_cam = vertices_space_cam
            display_faces = faces
            keypoints_2d = output["pred_keypoints_2d"]
            suggested_bbox: np.ndarray | None = None
            hand_bbox_source = "none"
            hand_temporal_info: dict[str, Any] = {
                "enabled": False,
                "status": "na",
                "bbox_source": None,
                "mesh_source": None,
                "missing_frames": 0,
                "drift_rejected": False,
                "center_jump_ratio": None,
                "iou_to_ref": None,
                "area_ratio_to_ref": None,
                "hold_active": False,
                "bbox_available": False,
                "mesh_available": bool(display_vertices_cam.shape[0] > 0),
                "lost_events_total": 0,
                "drift_rejects_total": 0,
                "reentry_events_total": 0,
            }

            if inference_target == "hand":
                compact_vertex_indices: np.ndarray | None = None
                hand_keypoints_2d, _ = _select_dominant_hand_keypoints(
                    keypoints_2d,
                    joints_cam,
                    frame.shape,
                )
                (
                    display_vertices_cam,
                    display_faces,
                    compact_vertex_indices,
                ) = extract_hand_mesh_from_keypoints(
                    vertices_cam,
                    faces,
                    focal_length=focal_length,
                    image_shape=frame.shape,
                    keypoints_2d=keypoints_2d,
                    keypoints_3d=joints_cam,
                    min_faces=18,
                )
                if compact_vertex_indices is None:
                    fallback_bbox = output["bbox"]
                    if hand_keypoints_2d is not None and hand_keypoints_2d.shape[0] >= 5:
                        hand_bbox = bbox_from_keypoints(
                            hand_keypoints_2d,
                            frame.shape,
                            expand_ratio=0.18,
                            min_size=24,
                            in_view_only=True,
                        )
                        if hand_bbox is not None:
                            fallback_bbox = hand_bbox
                    (
                        display_vertices_cam,
                        display_faces,
                        compact_vertex_indices,
                    ) = extract_hand_mesh_from_bbox(
                        vertices_cam,
                        faces,
                        focal_length=focal_length,
                        image_shape=frame.shape,
                        bbox_xyxy=fallback_bbox,
                        expand_ratio=0.05,
                        min_faces=18,
                    )
                if compact_vertex_indices is not None:
                    compact_bbox = bbox_from_projected_mesh(
                        display_vertices_cam,
                        focal_length=focal_length,
                        frame_shape=frame.shape,
                        expand_ratio=0.0,
                        min_size=1,
                        quantile_low=0.01,
                        quantile_high=0.99,
                    )
                    if compact_bbox is not None:
                        box_w = float(max(1.0, compact_bbox[2] - compact_bbox[0]))
                        box_h = float(max(1.0, compact_bbox[3] - compact_bbox[1]))
                        frame_area = float(max(1.0, frame.shape[0] * frame.shape[1]))
                        target_bbox = output["bbox"].astype(np.float32)
                        target_w = float(max(1.0, target_bbox[2] - target_bbox[0]))
                        target_h = float(max(1.0, target_bbox[3] - target_bbox[1]))
                        target_area = target_w * target_h
                        coverage_ratio = (box_w * box_h) / frame_area
                        compact_to_target_ratio = (box_w * box_h) / max(1.0, target_area)
                        if coverage_ratio > 0.45 or compact_to_target_ratio > 3.2:
                            # Safety net: in non-full mode, refuse broad/full-body meshes.
                            compact_vertex_indices = None
                if compact_vertex_indices is not None:
                    display_vertices_space_cam = vertices_space_cam[compact_vertex_indices]
                else:
                    # Strict non-full behavior: never display a full-body fallback mesh.
                    display_vertices_cam = np.empty((0, 3), dtype=np.float32)
                    display_faces = np.empty((0, 3), dtype=np.int32)
                    display_vertices_space_cam = np.empty((0, 3), dtype=np.float32)

                if hand_keypoints_2d is not None and hand_keypoints_2d.shape[0] >= 5:
                    suggested_bbox = bbox_from_keypoints(
                        hand_keypoints_2d,
                        frame.shape,
                        expand_ratio=0.16,
                        min_size=24,
                        in_view_only=True,
                    )
                    if suggested_bbox is not None:
                        hand_bbox_source = "hand_keypoints"
                if suggested_bbox is None:
                    suggested_bbox = bbox_from_projected_mesh(
                        display_vertices_cam,
                        focal_length=focal_length,
                        frame_shape=frame.shape,
                        expand_ratio=0.10,
                        min_size=24,
                    )
                    if suggested_bbox is not None:
                        hand_bbox_source = "mesh_projection"
                if suggested_bbox is None:
                    suggested_bbox = clip_bbox(output["bbox"], frame.shape)
                    hand_bbox_source = "raw_output"

                if hand_temporal is not None:
                    (
                        suggested_bbox,
                        display_vertices_cam,
                        display_vertices_space_cam,
                        display_faces,
                        hand_temporal_info,
                    ) = hand_temporal.update(
                        prior_bbox=patient_bbox,
                        measured_bbox=suggested_bbox,
                        measured_vertices_cam=display_vertices_cam,
                        measured_vertices_space_cam=display_vertices_space_cam,
                        measured_faces=display_faces,
                        bbox_source=hand_bbox_source,
                    )
                else:
                    hand_temporal_info["status"] = "disabled"
                    hand_temporal_info["bbox_source"] = hand_bbox_source
                    hand_temporal_info["mesh_source"] = (
                        "measured" if display_vertices_cam.shape[0] > 0 else "empty"
                    )
            else:
                suggested_bbox = bbox_from_keypoints(keypoints_2d, frame.shape)

            track_start = time.perf_counter()
            candidate_bbox: np.ndarray | None = None
            alpha_adaptive: float | None = None
            if suggested_bbox is not None:
                if inference_target == "hand":
                    candidate_bbox = clip_bbox(suggested_bbox, frame.shape)
                else:
                    alpha_adaptive = adaptive_bbox_smoothing_alpha(
                        patient_bbox,
                        suggested_bbox,
                        alpha_slow=bbox_alpha_slow,
                        alpha_fast=bbox_alpha_fast,
                        fast_motion_ratio=bbox_fast_motion_ratio,
                    )
                    bbox_smoothing_alpha_values.append(alpha_adaptive)
                    candidate_bbox = smooth_bbox(patient_bbox, suggested_bbox, alpha=alpha_adaptive)
            if (
                manual_guided_bbox is not None
                and inference_target != "hand"
            ):
                guide_bbox = clip_bbox(manual_guided_bbox, frame.shape)
                if candidate_bbox is None:
                    candidate_bbox = guide_bbox
                else:
                    # The optical-flow manual guide is a legacy pre-detector
                    # follower that drifts during turns. With the identity lock's
                    # detector + fixed gallery now correcting every cadence frame,
                    # keep the guide as a light nudge so it stops injecting drift
                    # the detector then has to snap back (status churn).
                    guide_alpha = (
                        0.82
                        if (identity_tracker is not None and identity_tracker.enabled)
                        else 0.60
                    )
                    candidate_bbox = smooth_bbox(guide_bbox, candidate_bbox, alpha=guide_alpha)

            identity_info: dict[str, Any] = {
                "status": "na",
                "present": True,
                "supported": False,
                "is_lost": False,
                "lost_frames": 0,
                "appearance_similarity": None,
                "gallery_similarity": None,
                "stability_score": None,
                "reacquire_scanned": False,
                "reacquire_candidates": 0,
                "blocked_switches_total": 0,
                "reacquired_total": 0,
            }
            if identity_tracker is not None:
                # Re-assert user ground truth when we reach an anchor frame.
                anchor_bbox_now: np.ndarray | None = None
                while (
                    next_anchor_i < len(anchor_track)
                    and anchor_track[next_anchor_i][0] <= frame_idx
                ):
                    anchor_bbox_now = anchor_track[next_anchor_i][1]
                    next_anchor_i += 1

                # Independent detector evidence on a cadence (and whenever the
                # tracker is unsure): this is the anti-drift / absence ground truth.
                detections: list[dict[str, Any]] | None = None
                if (
                    inference_target != "hand"
                    and anchor_bbox_now is None
                    and (
                        processed_idx % identity_detect_every_n == 0
                        or identity_tracker.needs_reacquire_candidates()
                    )
                ):
                    reacq_start = time.perf_counter()
                    detections = collect_person_candidates_for_reacquire(
                        frame,
                        sam3_detector=sam3_auto_detector,
                        auto_detector_threshold=cfg.auto_detector_threshold,
                        sam3_text_prompts=sam3_text_prompts,
                    )
                    timing_reacquire_detect_s += time.perf_counter() - reacq_start

                patient_bbox, identity_info = identity_tracker.update(
                    frame,
                    candidate_bbox,
                    detections=detections,
                    anchor_bbox=anchor_bbox_now,
                )
            elif candidate_bbox is not None:
                patient_bbox = candidate_bbox
            timing_tracking_s += time.perf_counter() - track_start

            subject_present = bool(
                identity_info.get("present", not identity_info.get("is_lost", False))
            )
            if not subject_present:
                display_vertices_cam = np.empty((0, 3), dtype=np.float32)
                display_vertices_space_cam = np.empty((0, 3), dtype=np.float32)
                display_faces = np.empty((0, 3), dtype=np.int32)

            mesh_path = (
                mesh_dir / f"frame_{frame_idx:06d}.ply"
                if cfg.export_meshes and subject_present
                else None
            )
            if mesh_path is not None:
                mesh_export_start = time.perf_counter()
                export_mesh(display_vertices_cam, display_faces, mesh_path)
                timing_mesh_export_s += time.perf_counter() - mesh_export_start

            preview_bbox = clip_bbox(patient_bbox, frame.shape)
            should_render_any = (cfg.render_preview and writer is not None) or (
                live_preview_writer is not None
            )
            if should_render_any:
                render_start = time.perf_counter()
                if subject_present:
                    panel_top_left, mask = draw_mesh_overlay_and_mask(
                        frame,
                        display_vertices_cam,
                        display_faces,
                        focal_length=focal_length,
                        face_stride=cfg.face_stride_overlay,
                    )
                else:
                    panel_top_left = frame.copy()
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

                bbox_i = preview_bbox.astype(np.int32)
                if subject_present:
                    cv2.rectangle(
                        panel_top_left,
                        (int(bbox_i[0]), int(bbox_i[1])),
                        (int(bbox_i[2]), int(bbox_i[3])),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        panel_top_left,
                        "Partial target" if inference_target == "hand" else "Patient",
                        (int(bbox_i[0]), max(20, int(bbox_i[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        panel_top_left,
                        "Subject not visible",
                        (24, 44),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )

                need_rich_frame = (cfg.render_preview and writer is not None) or (
                    live_preview_writer is not None and live_preview_panel == "full"
                )
                rich_frame: np.ndarray | None = None
                if need_rich_frame:
                    panel_top_right = draw_segmentation_panel(frame, mask)

                    panel_bottom_left = render_3d_space_view(
                        display_vertices_space_cam,
                        display_faces,
                        image_h=height,
                        image_w=width * 2,
                        face_stride=cfg.face_stride_3d,
                        joints_3d_cam=(
                            None
                            if inference_target == "hand" or not subject_present
                            else joints_space_cam
                        ),
                        title=(
                            "3D local part space + axes"
                            if inference_target == "hand"
                            else "3D space + axes + ground"
                        ),
                        mesh_alpha=1.0,
                        level_joints=False,
                        world_anchor=fixed_world_anchor,
                        lock_world_anchor=True,
                        view_state=fixed_view_state,
                        lock_view_state=True,
                        view_zoom=1.18,
                        enforce_ground_contact=(
                            bool(cfg.enforce_ground_contact) if inference_target != "hand" else False
                        ),
                        ground_contact_auto=(
                            bool(cfg.ground_contact_auto) if inference_target != "hand" else False
                        ),
                        ground_contact_auto_calib_frames=int(cfg.ground_contact_auto_calib_frames),
                        ground_contact_quantile=float(cfg.ground_contact_quantile),
                        ground_contact_smoothing=float(cfg.ground_contact_smoothing),
                        camera_align_mode="hand" if inference_target == "hand" else "body",
                    )
                    rich_frame = compose_rich_2x2(
                        panel_top_left,
                        panel_top_right,
                        panel_bottom_left,
                        bottom_right=None,
                    )

                if cfg.render_preview and writer is not None:
                    assert rich_frame is not None
                    writer.write(rich_frame)

                if live_preview_writer is not None:
                    if live_preview_panel == "full":
                        assert rich_frame is not None
                        live_preview_writer.write(rich_frame)
                    else:
                        live_preview_writer.write(panel_top_left)
                timing_render_s += time.perf_counter() - render_start

            identity_status = identity_info.get("status")
            record_item: dict[str, Any] = {
                "video_frame": int(frame_idx),
                "mesh_path": str(mesh_path) if mesh_path is not None else None,
                "bbox_xyxy": [float(v) for v in patient_bbox.tolist()],
                "subject_present": bool(subject_present),
                "inference_status": (
                    "ok"
                    if subject_present
                    else ("subject_absent" if identity_status == "absent" else "subject_lost")
                ),
                "subject_tracking_status": (
                    str(identity_status)
                    if identity_status is not None
                    else ("tracked" if subject_present else "subject_lost")
                ),
                "subject_detector_supported": bool(identity_info.get("supported", False)),
                "focal_length": focal_length,
                "manual_subject_guide": bool(manual_guided_bbox is not None),
                "identity_lock_status": str(identity_status),
                "identity_gallery_similarity": (
                    float(identity_info["gallery_similarity"])
                    if identity_info.get("gallery_similarity") is not None
                    else None
                ),
                "identity_is_lost": bool(identity_info.get("is_lost", False)),
                "identity_lost_frames": int(identity_info.get("lost_frames", 0)),
                "identity_appearance_similarity": (
                    float(identity_info["appearance_similarity"])
                    if identity_info.get("appearance_similarity") is not None
                    else None
                ),
                "identity_stability_score": (
                    float(identity_info["stability_score"])
                    if identity_info.get("stability_score") is not None
                    else None
                ),
                "identity_reacquire_scanned": bool(
                    identity_info.get("reacquire_scanned", False)
                ),
                "identity_reacquire_candidates": int(
                    identity_info.get("reacquire_candidates", 0)
                ),
                "hand_temporal_status": (
                    str(hand_temporal_info.get("status"))
                    if inference_target == "hand"
                    else None
                ),
                "hand_temporal_missing_frames": (
                    int(hand_temporal_info.get("missing_frames", 0))
                    if inference_target == "hand"
                    else None
                ),
                "hand_temporal_drift_rejected": (
                    bool(hand_temporal_info.get("drift_rejected", False))
                    if inference_target == "hand"
                    else None
                ),
                "hand_temporal_bbox_source": (
                    str(hand_temporal_info.get("bbox_source"))
                    if inference_target == "hand" and hand_temporal_info.get("bbox_source") is not None
                    else None
                ),
                "hand_temporal_mesh_source": (
                    str(hand_temporal_info.get("mesh_source"))
                    if inference_target == "hand" and hand_temporal_info.get("mesh_source") is not None
                    else None
                ),
            }
            if cfg.export_joint_timeseries and subject_present:
                record_item["joints_space_cam_xyz"] = (
                    np.round(joints_space_cam.astype(np.float32), decimals=5).tolist()
                )
                record_item["joints_cam_xyz"] = (
                    np.round(joints_cam.astype(np.float32), decimals=5).tolist()
                )
                root_cam = (
                    np.mean(joints_cam[[9, 10]], axis=0)
                    if joints_cam.shape[0] > 10
                    else np.mean(joints_cam, axis=0)
                )
                root_world = np.asarray(
                    [-root_cam[2], root_cam[0], -root_cam[1]],
                    dtype=np.float32,
                )
                record_item["root_world_raw"] = (
                    np.round(root_world, decimals=5).tolist()
                )
                record_item["root_world_stabilized"] = (
                    np.round(root_world, decimals=5).tolist()
                )
            records.append(record_item)
            # Log this detector-cadence frame for the offline identity resolver:
            # every candidate box with its affinity to the frozen patient gallery.
            if (
                cfg.identity_offline_resolve
                and inference_target != "hand"
                and identity_tracker is not None
                and detections is not None
            ):
                candidate_log: list[dict[str, Any]] = []
                for det in detections:
                    det_box = clip_bbox(det["bbox"], frame.shape)
                    det_hist = extract_bbox_appearance_hist(frame, det_box)
                    candidate_log.append(
                        {
                            "bbox": det_box.astype(np.float32),
                            "gallery": gallery_match_score(
                                identity_tracker.fixed_gallery, det_hist
                            ),
                        }
                    )
                identity_trace.append(
                    {
                        "record_index": len(records) - 1,
                        "frame_idx": int(frame_idx),
                        "candidates": candidate_log,
                    }
                )
            processed_idx += 1
            if processed_idx == 1 or processed_idx % 10 == 0:
                write_progress_metadata(
                    cfg.output_dir,
                    video_input=cfg.video_input,
                    mesh_dir=mesh_dir,
                    records=records,
                    fps_output=out_fps,
                    video_width=width,
                    video_height=height,
                    inference_target=inference_target,
                    space_view={
                        "mode": "fixed_world_anchor",
                        "world_anchor": fixed_world_anchor if fixed_world_anchor else None,
                        "view_state": fixed_view_state if fixed_view_state else None,
                        "view_zoom": 1.18,
                    },
                    total_frames_target=progress_total_target,
                    processing_status="running",
                )
            if processed_idx % 10 == 0:
                print(f"Processed {processed_idx} frame(s) ...")
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted by user. Finalizing output files...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if live_preview_writer is not None:
            live_preview_writer.close()

    # Offline (non-causal) identity resolution. With the whole forward pass in
    # hand, replay the logged detector candidates through a Viterbi that uses
    # future frames to undo the greedy tracker's brief jumps onto a bystander.
    identity_offline_summary: dict[str, Any] = {"applied": False}
    if cfg.identity_offline_resolve and inference_target != "hand" and identity_trace:
        try:
            identity_offline_summary = apply_offline_identity_resolution(
                records,
                identity_trace,
                frame_shape=(height, width, 3),
                mesh_dir=mesh_dir,
                gallery_floor=float(cfg.identity_coast_gallery_floor),
            )
            # Persist the resolved patient boxes so an optional high-quality
            # re-inference pass can be fed them as manual subject guides (the
            # pipeline already accepts manual_subject_bboxes), recovering the
            # frames this pass could only suppress.
            resolved_boxes = identity_offline_summary.pop("resolved_subject_bboxes", None)
            if resolved_boxes:
                write_json(cfg.output_dir / "resolved_subject_bboxes.json", resolved_boxes)
            print(
                "Offline identity resolve: suppressed "
                f"{identity_offline_summary.get('suppressed_frames', 0)} teleport frame(s) "
                f"over {identity_offline_summary.get('detection_frames', 0)} detector frames."
            )
        except Exception as exc:  # never let post-processing sink a finished run
            print(f"Offline identity resolution skipped ({exc.__class__.__name__}: {exc}).")
            identity_offline_summary = {"applied": False, "error": str(exc)}

    metadata = {
        "video_input": str(cfg.video_input),
        "checkpoint_path": str(cfg.checkpoint_path),
        "mhr_path": str(cfg.mhr_path),
        "sam3d_code_root": str(cfg.sam3d_code_root),
        "sam3_code_root": str(cfg.sam3_code_root) if cfg.sam3_code_root is not None else None,
        "output_video": str(out_video_path) if cfg.render_preview else None,
        "mesh_dir": str(mesh_dir),
        "mesh_export_enabled": bool(cfg.export_meshes),
        "joint_timeseries_export_enabled": bool(cfg.export_joint_timeseries),
        "device": str(device),
        "mhr_backend": mhr_backend,
        "runtime_reused": bool(runtime_reused),
        "inference_precision_requested": cfg.inference_precision,
        "inference_precision_effective": (
            "float16" if inference_dtype == torch.float16 else "float32"
        ),
        "mps_mhr_mode_requested": mps_mhr_mode if device.type == "mps" else None,
        "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "1") == "1",
        "render_preview": cfg.render_preview,
        "frame_step_requested": cfg.frame_step,
        "frame_step_effective": effective_frame_step,
        "fps_input": fps,
        "fps_output": out_fps,
        "output_codec_requested": cfg.output_codec,
        "output_codec_effective": output_codec_effective,
        "live_preview_enabled": cfg.live_preview,
        "live_preview_panel": live_preview_panel,
        "live_preview_chunk_frames": int(cfg.live_preview_chunk_frames),
        "live_preview_refresh_every": int(cfg.live_preview_refresh_every),
        "live_preview_codec_requested": cfg.live_preview_codec if cfg.live_preview else None,
        "live_preview_codec_effective": (
            live_preview_writer.codec_effective if live_preview_writer is not None else None
        ),
        "live_preview_dir": str(live_preview_dir) if live_preview_dir is not None else None,
        "live_preview_path": str(live_preview_path) if live_preview_path is not None else None,
        "live_preview_files": (
            [str(p) for p in live_preview_writer.files]
            if live_preview_writer is not None
            else []
        ),
        "auto_init_mode": cfg.auto_init_mode,
        "auto_select_strategy_effective": auto_select_strategy,
        "inference_target": inference_target,
        "sam3_text_prompts": list(sam3_text_prompts),
        "auto_init_result": auto_init_result,
        "manual_subject_tracking": manual_subject_tracking_info,
        "video_width": width,
        "video_height": height,
        "total_frames_input": total_frames,
        "total_frames_processed": processed_idx,
        "timing_seconds": {
            "inference": timing_infer_s,
            "mesh_export": timing_mesh_export_s,
            "render_write": timing_render_s,
            "tracking_bbox": timing_tracking_s,
            "reacquire_detection": timing_reacquire_detect_s,
            "total_accounted": (
                timing_infer_s
                + timing_mesh_export_s
                + timing_render_s
                + timing_tracking_s
                + timing_reacquire_detect_s
            ),
        },
        "timing_ms_per_frame": {
            "inference": (1000.0 * timing_infer_s / processed_idx) if processed_idx > 0 else None,
            "mesh_export": (
                1000.0 * timing_mesh_export_s / processed_idx
            ) if processed_idx > 0 else None,
            "render_write": (1000.0 * timing_render_s / processed_idx) if processed_idx > 0 else None,
            "tracking_bbox": (
                1000.0 * timing_tracking_s / processed_idx
            ) if processed_idx > 0 else None,
            "reacquire_detection": (
                1000.0 * timing_reacquire_detect_s / processed_idx
            ) if processed_idx > 0 else None,
        },
        "bbox_smoothing": {
            "mode": "adaptive_center_motion",
            "alpha_slow": bbox_alpha_slow,
            "alpha_fast": bbox_alpha_fast,
            "fast_motion_ratio": bbox_fast_motion_ratio,
            "alpha_mean": (
                float(np.mean(bbox_smoothing_alpha_values))
                if bbox_smoothing_alpha_values
                else None
            ),
        },
        "identity_lock": (
            identity_tracker.summary() if identity_tracker is not None else None
        ),
        "identity_offline_resolution": identity_offline_summary,
        "hand_temporal_postprocess": (
            hand_temporal.summary() if hand_temporal is not None else None
        ),
        "space_view": {
            "mode": "fixed_world_anchor",
            "world_anchor": fixed_world_anchor if fixed_world_anchor else None,
            "view_state": fixed_view_state if fixed_view_state else None,
            "view_zoom": 1.18,
            "ground_contact_lock": {
                "enabled": bool(cfg.enforce_ground_contact),
                "auto": bool(cfg.ground_contact_auto),
                "auto_calib_frames": int(cfg.ground_contact_auto_calib_frames),
                "quantile": float(cfg.ground_contact_quantile),
                "smoothing": float(cfg.ground_contact_smoothing),
                "quantile_effective": (
                    float(fixed_world_anchor["_ground_contact_quantile_effective"])
                    if (
                        "_ground_contact_quantile_effective" in fixed_world_anchor
                        and isinstance(
                            fixed_world_anchor["_ground_contact_quantile_effective"],
                            (int, float),
                        )
                    )
                    else None
                ),
            },
        },
        "interrupted": interrupted,
        "processing_status": "interrupted" if interrupted else "completed",
        "records": records,
    }
    metadata_path = cfg.output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata
