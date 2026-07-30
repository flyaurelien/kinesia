"""Reconstruct an object's shape from one image and one mask.

The object model runs in its own environment — a different interpreter, its own
pinned dependencies, its own Metal kernels — so it is driven as a subprocess
rather than imported. That environment is not part of a clone: when it is
absent the caller is told why and the run continues without objects, because a
missing extra must never cost a finished human reconstruction.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .workspace import project_root_from

# Roughly four minutes are normal for one object; the ceiling is there to stop a
# wedged subprocess holding a job open forever, not to bound normal work.
RECONSTRUCTION_TIMEOUT_S = 1800

# The raw output can contain hundreds of thousands of faces per object, which
# the viewer has to fetch over HTTP for every object in the scene.
SIMPLIFY_RATIO = 0.9


def pointmap_path_for_pose(output_pose: Path) -> Path:
    """Return the scene-pointmap path paired with a model-pose artifact."""
    stem = output_pose.stem.removesuffix("_model_pose").removesuffix("_pose")
    return output_pose.with_name(f"{stem}_pointmap.npz")


def objects_root(project_root: Path | None = None) -> Path:
    """Where the object model lives (`SAM3D_OBJECTS_ROOT`, or vendored)."""
    configured = os.environ.get("SAM3D_OBJECTS_ROOT")
    if configured:
        return Path(configured).expanduser()
    return project_root_from(project_root) / "vendor" / "sam3d-objects-mlx"


def _dynamic_library_path(root: Path) -> str | None:
    """Where the object model's compiled extensions must look for their libraries.

    One of those extensions is built without an embedded search path, so nothing
    tells it where the tensor libraries live. Left alone it fails to load, and
    the failure is swallowed somewhere downstream: the process carries on and
    dies later inside an array copy, with a segmentation fault and no message.
    """
    for lib in sorted(root.glob(".venv/lib/python3.*/site-packages/torch/lib")):
        if lib.is_dir():
            return str(lib)
    return None


def unavailable_reason(root: Path | None = None) -> str | None:
    """Why object reconstruction cannot run here, or None when it can."""
    base = root or objects_root()
    if not base.is_dir():
        return f"object model not installed at {base}"
    interpreter = base / ".venv" / "bin" / "python"
    if not interpreter.exists():
        return f"object model has no environment at {interpreter}"
    if not (base / "main.py").exists():
        return f"object model entry point missing at {base / 'main.py'}"
    # exists() follows symlinks, and these checkpoints ARE symlinks into a
    # separate download tree — a broken link has to read as missing.
    config = base / "checkpoints" / "hf" / "pipeline.yaml"
    if not config.exists():
        return f"object model weights missing at {config}"
    return None


def reconstruct_mesh(
    image_path: Path,
    mask_path: Path,
    output_glb: Path,
    output_pose: Path,
    cache_dir: Path,
    root: Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Build a mesh from an image and its mask, and return the written .glb.

    Raises RuntimeError if the mesh does not come out.
    """
    base = root or objects_root()
    reason = unavailable_reason(base)
    if reason:
        raise RuntimeError(reason)

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pointmap_output = pointmap_path_for_pose(output_pose)
    for stale_output in (output_glb, output_pose, pointmap_output):
        if not stale_output.exists():
            continue
        # A stale artifact from an earlier attempt must not be mistaken for
        # this run's output when the external process exits without writing it.
        stale_output.unlink()

    command = [
        str(base / ".venv" / "bin" / "python"),
        str(project_root_from() / "scripts" / "run_sam3d_objects_pose.py"),
        "--image", str(image_path.resolve()),
        "--mask", str(mask_path.resolve()),
        "--output", str(output_glb.resolve()),
        "--pose-output", str(output_pose.resolve()),
        "--pointmap-output", str(pointmap_output.resolve()),
        # Per run: the cache keys on a weak hash of the input and is written
        # non-atomically, so a shared directory lets one object read another's
        # geometry.
        "--cache-dir", str(cache_dir.resolve()),
        "--simplify", str(SIMPLIFY_RATIO),
    ]
    environment = _runtime_environment(base)
    # The object model prefers its own hand-written Metal shaders for sparse
    # convolution. They segfault partway through generation when another
    # process is already working the GPU hard — a local model server is enough
    # — and a crash inside vendored kernels is not something this pipeline can
    # recover from. The portable path costs about the same (measured: four
    # minutes against the four the shaders take) and does not crash, so it is
    # what an unattended step uses. Override to compare.
    log(f"reconstructing shape from {image_path.name} + {mask_path.name}")
    try:
        completed = subprocess.run(
            command,
            env=environment,
            # The entry point resolves its configuration relative to the working
            # directory, and its package is only importable from there.
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=RECONSTRUCTION_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"object reconstruction timed out after {RECONSTRUCTION_TIMEOUT_S}s"
        ) from error

    for line in (completed.stdout or "").splitlines()[-12:]:
        log(f"  {line}")

    # The exit code cannot be trusted: a failed decode prints an error and still
    # returns zero. The mesh on disk is the only proof.
    if not output_glb.exists():
        raise RuntimeError(
            f"no mesh produced ({_failure_detail(completed)})"
        )
    if not output_pose.exists():
        raise RuntimeError("the object runtime produced no pose output")
    log(f"shape written: {output_glb.name} ({output_glb.stat().st_size // 1024} kB)")
    return output_glb


def reconstruct_pose(
    image_path: Path,
    mask_path: Path,
    output_pose: Path,
    cache_dir: Path,
    root: Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Run SAM 3D Objects for one image/mask pose without exporting a mesh."""
    base = root or objects_root()
    reason = unavailable_reason(base)
    if reason:
        raise RuntimeError(reason)

    output_pose.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pointmap_output = pointmap_path_for_pose(output_pose)
    for stale_output in (output_pose, pointmap_output):
        if stale_output.exists():
            stale_output.unlink()
    output_placeholder = cache_dir / f"{output_pose.stem}.stl"
    command = [
        str(base / ".venv" / "bin" / "python"),
        str(project_root_from() / "scripts" / "run_sam3d_objects_pose.py"),
        "--image", str(image_path.resolve()),
        "--mask", str(mask_path.resolve()),
        "--output", str(output_placeholder.resolve()),
        "--pose-output", str(output_pose.resolve()),
        "--pointmap-output", str(pointmap_output.resolve()),
        "--cache-dir", str(cache_dir.resolve()),
        "--pose-only",
    ]
    environment = _runtime_environment(base)
    log(f"reconstructing pose from {image_path.name} + {mask_path.name}")
    try:
        completed = subprocess.run(
            command, env=environment, cwd=str(base), capture_output=True,
            text=True, timeout=RECONSTRUCTION_TIMEOUT_S, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"object pose reconstruction timed out after {RECONSTRUCTION_TIMEOUT_S}s"
        ) from error
    for line in (completed.stdout or "").splitlines()[-12:]:
        log(f"  {line}")
    if not output_pose.exists():
        raise RuntimeError(f"no pose produced ({_failure_detail(completed)})")
    return output_pose


def _runtime_environment(base: Path) -> dict[str, str]:
    """Build the external runtime environment shared by mesh and pose calls."""
    environment = dict(os.environ)
    environment.setdefault("SPARSE_BACKEND", "mps")
    environment.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    libraries = _dynamic_library_path(base)
    if libraries:
        existing = environment.get("DYLD_LIBRARY_PATH")
        environment["DYLD_LIBRARY_PATH"] = (
            f"{libraries}:{existing}" if existing else libraries
        )
    return environment


def _failure_detail(completed: subprocess.CompletedProcess) -> str:
    """The most useful line about why a reconstruction produced nothing.

    Taking the last line of stderr is not enough: a shutdown warning about
    leaked semaphores routinely comes after the real cause and would be
    reported instead of it.
    """
    noise = ("warnings.warn", "UserWarning", "FutureWarning", "DeprecationWarning")
    lines = [
        line.strip()
        for line in (completed.stderr or "").splitlines()
        if line.strip() and not any(marker in line for marker in noise)
    ]
    if completed.returncode < 0:
        return f"killed by signal {-completed.returncode}" + (
            f"; last output: {lines[-1]}" if lines else ""
        )
    if lines:
        return lines[-1]
    return f"exit code {completed.returncode}"


def clear_cache(cache_dir: Path) -> None:
    """Drop a run's reconstruction cache; it is scratch, and it is large."""
    shutil.rmtree(cache_dir, ignore_errors=True)
