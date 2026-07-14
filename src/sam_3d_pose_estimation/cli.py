from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# The SAM3 detector uses an op (aten::_assert_async) with no MPS kernel; without
# the per-op CPU fallback it raises and detection silently finds NOBODY on Apple
# Silicon. Default it ON — harmless on CUDA/CPU. (An explicit env value still wins.)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .analytics import AnalysisParams, analyze_run
from .artifacts import build_run_manifest, ensure_run_layout, write_json
from .doctor import run_doctor
from .pipeline import PipelineConfig, parse_bbox, run_pipeline
from .workspace import DEFAULT_ANALYSIS_PRESET, DEFAULT_CONFIG_PROFILE, project_root_from


DEFAULT_PROJECT_ROOT = project_root_from(Path(__file__))
DEFAULT_MODELS_ROOT = Path(os.environ.get("SAM3D_MODELS_ROOT", DEFAULT_PROJECT_ROOT / "models"))
DEFAULT_SAM3D_CODE_ROOT = Path(os.environ.get("SAM3D_CODE_ROOT", DEFAULT_PROJECT_ROOT / "vendor" / "sam-3d-body-main"))
DEFAULT_SAM3_CODE_ROOT = Path(os.environ.get("SAM3_CODE_ROOT", DEFAULT_PROJECT_ROOT / "vendor" / "sam3-main"))
DEFAULT_CHECKPOINT_PATH = Path(
    os.environ.get(
        "SAM3D_CHECKPOINT_PATH",
        DEFAULT_MODELS_ROOT / "sam-3d-body-dinov3" / "model.ckpt",
    )
)
DEFAULT_MHR_PATH = Path(
    os.environ.get(
        "SAM3D_MHR_PATH",
        DEFAULT_MODELS_ROOT / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt",
    )
)
SUBCOMMANDS = {"run", "analyze", "doctor"}


def sanitize_token(input_text: str) -> str:
    """Slugify text into a filesystem-safe lowercase token (falls back to 'run')."""
    token = re.sub(r"[^a-zA-Z0-9_-]+", "_", input_text.strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "run"


def parse_mask_time_ranges(raw: str) -> tuple[tuple[float, float], ...]:
    """Parse 'start-end,start-end' (seconds) into a tuple of (start, end) pairs."""
    ranges: list[tuple[float, float]] = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk or "-" not in chunk:
            continue
        start_str, _, end_str = chunk.partition("-")
        try:
            start = float(start_str)
            end = float(end_str)
        except ValueError:
            continue
        if end > start >= 0:
            ranges.append((start, end))
    return tuple(ranges)


def add_run_arguments(parser: argparse.ArgumentParser, *, require_output_dir: bool) -> None:
    """Register the shared `run` flags so the modern subcommand and the legacy
    flat CLI expose an identical set of options."""
    parser.add_argument("--video-input", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=require_output_dir,
        help="Output directory for meshes, preview video, metadata, and manifests.",
    )
    parser.add_argument("--run-id", type=str, default="", help="Run identifier used under output/.")
    parser.add_argument("--sam3d-code-root", type=Path, default=DEFAULT_SAM3D_CODE_ROOT)
    parser.add_argument("--sam3-code-root", type=Path, default=DEFAULT_SAM3_CODE_ROOT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--mhr-path", type=Path, default=DEFAULT_MHR_PATH)
    parser.add_argument("--prompt-bbox", type=str, default="")
    parser.add_argument(
        "--prompt-bbox-frame",
        "--subject-frame",
        dest="prompt_bbox_frame",
        type=int,
        default=None,
        help="Frame where --prompt-bbox was selected; used as a subject-tracking anchor.",
    )
    parser.add_argument(
        "--prompt-anchors-json",
        type=str,
        default="",
        help=(
            "Optional JSON list of subject-tracking anchors: "
            "'[{\"frameIndex\":int,\"bbox\":[x1,y1,x2,y2]}, ...]'. "
            "When provided, the anchor closest to the median requested frame "
            "is used as the primary --prompt-bbox/--prompt-bbox-frame, and "
            "the full list is persisted in run_metadata.json under "
            "tracking_anchors for future multi-anchor tracking."
        ),
    )
    parser.add_argument(
        "--subject-track-file",
        type=str,
        default="",
        help=(
            "Path to a JSON file with a dense per-frame box track of the chosen "
            "subject (from the detect-step streaming preview): "
            '{"frames": {"0": [x1,y1,x2,y2], ...}}. When provided, the run pins '
            "the subject to that track every detected frame so it reconstructs "
            "exactly the person the user picked."
        ),
    )
    parser.add_argument(
        "--subject-index",
        type=int,
        default=0,
        help=(
            "Which subject of the --subject-track-file to reconstruct (0-based). "
            "Multi-subject selections run once per subject with this index."
        ),
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--mask-time-ranges",
        type=str,
        default="",
        help="Comma-separated 'start-end' second spans to keep but skip inference on (masking).",
    )
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-mesh-export", action="store_true")
    parser.add_argument("--no-joint-timeseries", action="store_true")
    parser.add_argument("--face-stride-overlay", type=int, default=1)
    parser.add_argument("--face-stride-3d", type=int, default=1)
    parser.add_argument("--output-codec", type=str, default="h264", choices=["h264", "avc1", "mp4v"])
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--live-preview", action="store_true")
    parser.add_argument("--auto-init-mode", type=str, default="sam3", choices=["off", "smart", "sam3"])
    parser.add_argument("--inference-target", type=str, default="body", choices=["body", "hand", "partial"])
    parser.add_argument("--auto-detector-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-text-prompts", type=str, default="")
    parser.add_argument(
        "--auto-select-strategy",
        type=str,
        default="patient",
        choices=["patient", "largest", "leftmost", "rightmost", "center", "tightest"],
    )
    parser.add_argument("--disable-ground-contact-lock", action="store_true")
    parser.add_argument("--disable-ground-contact-auto", action="store_true")
    parser.add_argument("--ground-contact-auto-calib-frames", type=int, default=45)
    parser.add_argument("--ground-contact-quantile", type=float, default=1.5)
    parser.add_argument("--ground-contact-smoothing", type=float, default=0.35)
    parser.add_argument("--bbox-smoothing-alpha-slow", type=float, default=0.55)
    parser.add_argument("--bbox-smoothing-alpha-fast", type=float, default=0.95)
    parser.add_argument("--bbox-smoothing-fast-motion-ratio", type=float, default=0.10)
    parser.add_argument("--disable-identity-lock", action="store_true")
    parser.add_argument("--identity-warmup-frames", type=int, default=10)
    parser.add_argument("--identity-max-center-jump-ratio", type=float, default=0.35)
    parser.add_argument("--identity-min-appearance-sim", type=float, default=0.32)
    parser.add_argument("--identity-reacquire-min-sim", type=float, default=0.42)
    parser.add_argument("--identity-reacquire-every", type=int, default=4)
    parser.add_argument("--identity-max-hold-frames", type=int, default=240)
    parser.add_argument("--disable-hand-temporal-postprocess", action="store_true")
    parser.add_argument("--hand-occlusion-hold-frames", type=int, default=16)
    parser.add_argument("--hand-interpolation-max-gap", type=int, default=8)
    parser.add_argument("--hand-reentry-blend-frames", type=int, default=6)
    parser.add_argument("--hand-drift-max-center-jump-ratio", type=float, default=0.82)
    parser.add_argument("--hand-drift-min-iou", type=float, default=0.03)
    parser.add_argument("--hand-drift-max-area-ratio", type=float, default=2.6)
    parser.add_argument("--hand-bbox-smoothing-alpha", type=float, default=0.72)
    parser.add_argument("--hand-hold-follow-alpha", type=float, default=0.22)
    parser.add_argument("--hand-mesh-smoothing-alpha", type=float, default=0.58)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with the run/analyze/doctor subcommands."""
    parser = argparse.ArgumentParser(prog="sam3d", description="SAM3D workstation CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run inference and produce a versioned run manifest.")
    add_run_arguments(run_parser, require_output_dir=False)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a run and export kinematics artifacts.")
    analyze_parser.add_argument("--run-id", required=True)
    analyze_parser.add_argument("--preset", default=DEFAULT_ANALYSIS_PRESET)
    analyze_parser.add_argument("--json", action="store_true")

    doctor_parser = subparsers.add_parser("doctor", help="Validate environment and model dependencies.")
    doctor_parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    doctor_parser.add_argument("--mhr-path", type=Path, default=DEFAULT_MHR_PATH)
    doctor_parser.add_argument("--sam3d-code-root", type=Path, default=DEFAULT_SAM3D_CODE_ROOT)
    doctor_parser.add_argument("--sam3-code-root", type=Path, default=DEFAULT_SAM3_CODE_ROOT)
    doctor_parser.add_argument("--json", action="store_true")
    return parser


def parse_sam3_text_prompts(text: str | None) -> tuple[str, ...]:
    """Split a comma/semicolon/pipe-separated prompt string into a deduplicated,
    case-insensitive tuple (max 8), defaulting to ('person',) when empty."""
    raw = text or ""
    prompts: list[str] = []
    seen: set[str] = set()
    for token in raw.replace(";", ",").replace("|", ",").split(","):
        prompt = token.strip()
        if not prompt:
            continue
        key = prompt.casefold()
        if key in seen:
            continue
        seen.add(key)
        prompts.append(prompt)
        if len(prompts) >= 8:
            break
    if len(prompts) == 0:
        return ("person",)
    return tuple(prompts)


def _frames_to_anchors(frames: dict) -> list[dict]:
    anchors: list[dict] = []
    for key, box in frames.items():
        try:
            frame_index = int(key)
        except (TypeError, ValueError):
            continue
        if frame_index < 0 or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            bbox = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        anchors.append({"frameIndex": frame_index, "bbox": bbox})
    anchors.sort(key=lambda a: a["frameIndex"])
    return anchors


def load_subject_tracks(path: str) -> list[dict]:
    """Load the detect-step chosen-subject track file into a list of subjects,
    each {"subject_id", "anchors":[{frameIndex,bbox}]}. Accepts the multi-subject
    format ({"subjects":[{subjectId,frames}]}) and the legacy single-subject one
    ({"frames":{...}}). Each frame becomes a hard tracking anchor."""
    text = (path or "").strip()
    if not text:
        return []
    try:
        data = json.loads(Path(text).read_text())
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("subjects"), list):
        out: list[dict] = []
        for i, subj in enumerate(data["subjects"]):
            if not isinstance(subj, dict) or not isinstance(subj.get("frames"), dict):
                continue
            anchors = _frames_to_anchors(subj["frames"])
            if anchors:
                out.append({
                    "subject_id": str(subj.get("subjectId", i)),
                    "label": str(subj.get("label", len(out) + 1)),
                    "color": str(subj.get("color", "")),
                    "anchors": anchors,
                })
        return out
    frames = data.get("frames", {})
    if isinstance(frames, dict):
        anchors = _frames_to_anchors(frames)
        if anchors:
            return [{"subject_id": "0", "label": "1", "color": "", "anchors": anchors}]
    return []


def select_subject_track(path: str, index: int) -> dict | None:
    """Pick ONE subject of the chosen-subject track file for this run.
    Multi-subject reconstruction = one pipeline run per subject: the web layer
    spawns N jobs over the same file with --subject-index 0..N-1."""
    subjects = load_subject_tracks(path)
    if not subjects:
        return None
    return subjects[max(0, min(int(index), len(subjects) - 1))]


def load_subject_track_anchors(path: str, index: int = 0) -> list[dict]:
    """Anchors of the chosen subject (see select_subject_track)."""
    subject = select_subject_track(path, index)
    return subject["anchors"] if subject else []


def default_prompt_for_target(inference_target: str) -> str:
    """Pick the default open-vocab detector prompt for the inference target
    (body parts for hand/partial runs, otherwise the whole 'person')."""
    target = inference_target.strip().lower()
    if target in {"hand", "partial", "part", "non_full", "non-full"}:
        return "hand,arm,leg,foot,head,face"
    return "person"


def parse_prompt_anchors_json(raw: str) -> list[dict]:
    """Parse the multi-anchor JSON. Returns sorted list of dicts with
    integer frameIndex and 4-float bbox. Invalid items are skipped silently.
    """
    text = (raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    anchors: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        frame_raw = item.get("frameIndex", item.get("frame", item.get("frame_index")))
        bbox_raw = item.get("bbox", item.get("box"))
        try:
            frame_index = int(frame_raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
            continue
        try:
            bbox = [float(v) for v in bbox_raw]
        except (TypeError, ValueError):
            continue
        if frame_index < 0:
            continue
        anchors.append({"frameIndex": frame_index, "bbox": bbox})
    anchors.sort(key=lambda a: a["frameIndex"])
    return anchors


def select_primary_anchor(anchors: list[dict]) -> dict | None:
    """Pick the anchor closest to the median frameIndex — most representative
    for the current single-prompt tracker."""
    if not anchors:
        return None
    sorted_anchors = sorted(anchors, key=lambda a: a["frameIndex"])
    median = sorted_anchors[len(sorted_anchors) // 2]
    return median


def build_pipeline_config(args: argparse.Namespace, output_dir: Path) -> PipelineConfig:
    """Translate parsed CLI args into a fully-resolved PipelineConfig, including
    prompt/anchor promotion and clamping of numeric options to safe ranges."""
    prompt_bbox = parse_bbox(args.prompt_bbox) if args.prompt_bbox else None
    sam3_text_prompts = parse_sam3_text_prompts(args.sam3_text_prompts)
    if len(args.sam3_text_prompts.strip()) == 0:
        sam3_text_prompts = parse_sam3_text_prompts(default_prompt_for_target(args.inference_target))

    anchors = parse_prompt_anchors_json(getattr(args, "prompt_anchors_json", ""))
    # A chosen-subject track file (from the detect step) supersedes any other
    # anchors: every detected frame becomes a hard anchor so the run reconstructs
    # exactly the picked person (one run per subject via --subject-index).
    subject = select_subject_track(
        getattr(args, "subject_track_file", ""), getattr(args, "subject_index", 0)
    )
    if subject and subject["anchors"]:
        anchors = subject["anchors"]
    # If multi-anchor list provided and no explicit single-box was passed,
    # promote the median anchor to the primary prompt fields so the existing
    # single-anchor tracker can consume it.
    primary_anchor = None
    if anchors and prompt_bbox is None:
        primary_anchor = select_primary_anchor(anchors)
        if primary_anchor is not None:
            prompt_bbox = parse_bbox(
                ",".join(str(v) for v in primary_anchor["bbox"])
            )

    prompt_bbox_frame_arg = args.prompt_bbox_frame
    if primary_anchor is not None and prompt_bbox_frame_arg is None:
        prompt_bbox_frame_arg = primary_anchor["frameIndex"]

    return PipelineConfig(
        video_input=args.video_input.expanduser().resolve(),
        output_dir=output_dir.expanduser().resolve(),
        sam3d_code_root=args.sam3d_code_root.expanduser().resolve(),
        checkpoint_path=args.checkpoint_path.expanduser().resolve(),
        mhr_path=args.mhr_path.expanduser().resolve(),
        prompt_bbox=prompt_bbox,
        prompt_bbox_frame=(
            max(prompt_bbox_frame_arg, 0)
            if prompt_bbox_frame_arg is not None
            else None
        ),
        tracking_anchors=anchors or None,
        subject_track_file=(getattr(args, "subject_track_file", "") or "").strip() or None,
        subject_index=max(0, int(getattr(args, "subject_index", 0) or 0)),
        subject_id=(subject or {}).get("subject_id"),
        subject_label=(subject or {}).get("label"),
        subject_color=((subject or {}).get("color") or "").strip() or None,
        start_frame=max(args.start_frame, 0),
        frame_step=max(args.frame_step, 1),
        max_frames=args.max_frames,
        mask_time_ranges=parse_mask_time_ranges(getattr(args, "mask_time_ranges", "")),
        force_cpu=args.force_cpu,
        cpu_threads=max(args.cpu_threads, 0),
        render_preview=not args.no_preview,
        export_meshes=not args.no_mesh_export,
        export_joint_timeseries=not args.no_joint_timeseries,
        face_stride_overlay=max(args.face_stride_overlay, 1),
        face_stride_3d=max(args.face_stride_3d, 1),
        output_codec=args.output_codec,
        inference_precision=args.precision,
        live_preview=bool(args.live_preview),
        live_preview_panel="top-left",
        live_preview_chunk_frames=60,
        live_preview_refresh_every=1,
        live_preview_codec="h264",
        sam3_code_root=args.sam3_code_root.expanduser().resolve(),
        inference_target=args.inference_target,
        auto_init_mode=args.auto_init_mode,
        auto_detector_threshold=float(args.auto_detector_threshold),
        sam3_text_prompts=sam3_text_prompts,
        auto_select_strategy=args.auto_select_strategy,
        enforce_ground_contact=not bool(args.disable_ground_contact_lock),
        ground_contact_auto=not bool(args.disable_ground_contact_auto),
        ground_contact_auto_calib_frames=max(8, int(args.ground_contact_auto_calib_frames)),
        ground_contact_quantile=float(args.ground_contact_quantile),
        ground_contact_smoothing=float(args.ground_contact_smoothing),
        bbox_smoothing_alpha_slow=float(args.bbox_smoothing_alpha_slow),
        bbox_smoothing_alpha_fast=float(args.bbox_smoothing_alpha_fast),
        bbox_smoothing_fast_motion_ratio=float(args.bbox_smoothing_fast_motion_ratio),
        identity_lock_enabled=not bool(args.disable_identity_lock),
        identity_warmup_frames=max(1, int(args.identity_warmup_frames)),
        identity_max_center_jump_ratio=float(args.identity_max_center_jump_ratio),
        identity_min_appearance_similarity=float(args.identity_min_appearance_sim),
        identity_reacquire_min_similarity=float(args.identity_reacquire_min_sim),
        identity_reacquire_every_n=max(1, int(args.identity_reacquire_every)),
        identity_max_hold_frames=max(1, int(args.identity_max_hold_frames)),
        hand_temporal_enabled=not bool(args.disable_hand_temporal_postprocess),
        hand_occlusion_hold_frames=max(1, int(args.hand_occlusion_hold_frames)),
        hand_interpolation_max_gap=max(0, int(args.hand_interpolation_max_gap)),
        hand_reentry_blend_frames=max(0, int(args.hand_reentry_blend_frames)),
        hand_drift_max_center_jump_ratio=float(args.hand_drift_max_center_jump_ratio),
        hand_drift_min_iou=float(args.hand_drift_min_iou),
        hand_drift_max_area_ratio=float(args.hand_drift_max_area_ratio),
        hand_bbox_smoothing_alpha=float(args.hand_bbox_smoothing_alpha),
        hand_hold_follow_alpha=float(args.hand_hold_follow_alpha),
        hand_mesh_smoothing_alpha=float(args.hand_mesh_smoothing_alpha),
    )


def infer_run_id(video_input: Path, requested_run_id: str) -> str:
    """Return the sanitized run id, deriving one from the video name when none is given."""
    if requested_run_id.strip():
        return sanitize_token(requested_run_id)
    return sanitize_token(f"{video_input.stem}_processed")


def cmd_run(args: argparse.Namespace) -> int:
    """Run the inference pipeline, persist the run manifest, and print a summary."""
    project_root = project_root_from(Path.cwd())
    run_id = infer_run_id(args.video_input, args.run_id)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ensure_run_layout(run_id, project_root)
    )
    cfg = build_pipeline_config(args, output_dir)
    if cfg.tracking_anchors:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "tracking_anchors.json",
            {
                "schema": "kinesia.tracking_anchors.v1",
                "anchors": cfg.tracking_anchors,
                "primary_anchor_frame": cfg.prompt_bbox_frame,
            },
        )
    metadata = run_pipeline(cfg)
    manifest = build_run_manifest(
        run_id=run_id,
        run_directory=output_dir,
        metadata=metadata,
        config_profile=DEFAULT_CONFIG_PROFILE,
    )
    write_json(output_dir / "run_manifest.json", manifest)
    if args.output_dir is None:
        print(json.dumps({
            "run_id": run_id,
            "run_dir": str(output_dir),
            "run_manifest": str(output_dir / "run_manifest.json"),
            "processed_frames": metadata.get("total_frames_processed"),
        }, indent=2))
    else:
        print("\nRun completed.")
        print(f"Device: {metadata['device']}")
        print(f"Processed frames: {metadata['total_frames_processed']}")
        if metadata["output_video"]:
            print(f"Output preview video: {metadata['output_video']}")
        else:
            print("Output preview video: disabled (--no-preview)")
        print(f"Output mesh directory: {metadata['mesh_dir']}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze an existing run and print either the full manifest or a short summary."""
    params = AnalysisParams(preset=args.preset)
    result = analyze_run(run_id=args.run_id, params=params)
    if args.json:
        print(json.dumps(result["manifest"], indent=2))
    else:
        print(json.dumps({
            "analysis_id": result["analysis_id"],
            "qa_status": result["qa"]["status"],
            "needs_review": result["qa"]["needs_review"],
        }, indent=2))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Validate the environment and model dependencies, printing the doctor report."""
    summary = run_doctor(
        checkpoint_path=args.checkpoint_path.expanduser().resolve(),
        mhr_path=args.mhr_path.expanduser().resolve(),
        sam3d_code_root=args.sam3d_code_root.expanduser().resolve(),
        sam3_code_root=args.sam3_code_root.expanduser().resolve(),
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, indent=2))
    return 0


def main() -> None:
    """CLI entry point; dispatches to a subcommand or the legacy flat run interface."""
    argv = sys.argv[1:]
    if len(argv) == 0 or (argv[0] not in SUBCOMMANDS and argv[0].startswith("-")):
        legacy_parser = argparse.ArgumentParser(
            prog="sam3d-video",
            description="Run SAM 3D Body on a video and export patient-centered 3D meshes.",
        )
        add_run_arguments(legacy_parser, require_output_dir=True)
        args = legacy_parser.parse_args(argv)
        raise SystemExit(cmd_run(args))

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        raise SystemExit(cmd_run(args))
    if args.command == "analyze":
        raise SystemExit(cmd_analyze(args))
    if args.command == "doctor":
        raise SystemExit(cmd_doctor(args))
    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
