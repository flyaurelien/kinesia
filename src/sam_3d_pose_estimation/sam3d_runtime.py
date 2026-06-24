from __future__ import annotations

import os
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


class _MHRMPSWrapper(torch.nn.Module):
    """Run TorchScript MHR on CPU when model tensors are on MPS."""

    def __init__(self, mhr_module: torch.nn.Module):
        """Pin the wrapped MHR module to CPU in frozen eval mode."""
        super().__init__()
        self.mhr_module = mhr_module.to("cpu").eval()
        for param in self.mhr_module.parameters():
            param.requires_grad = False

    @staticmethod
    def _move_output(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
        """Recursively move tensors back to the caller's device/dtype, preserving container shape."""
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                return value.to(device=device, dtype=dtype)
            return value.to(device=device)
        if isinstance(value, tuple):
            return tuple(_MHRMPSWrapper._move_output(v, device, dtype) for v in value)
        if isinstance(value, list):
            return [_MHRMPSWrapper._move_output(v, device, dtype) for v in value]
        return value

    def forward(
        self,
        shape_params: torch.Tensor,
        model_params: torch.Tensor,
        expr_params: torch.Tensor | None = None,
    ) -> Any:
        """Run MHR on CPU when inputs are on MPS, then move results back to the input device."""
        if shape_params.device.type != "mps":
            return self.mhr_module(shape_params, model_params, expr_params)

        out_device = shape_params.device
        out_dtype = shape_params.dtype
        shape_cpu = shape_params.to("cpu", dtype=torch.float32)
        model_cpu = model_params.to("cpu", dtype=torch.float32)
        expr_cpu = (
            expr_params.to("cpu", dtype=torch.float32)
            if expr_params is not None
            else None
        )
        outputs = self.mhr_module(shape_cpu, model_cpu, expr_cpu)
        return self._move_output(outputs, out_device, out_dtype)


def _patch_mhr_torchscript_for_mps_float32(ts_module: torch.jit.ScriptModule) -> bool:
    """
    Patch TorchScript MHR graph to avoid hard-coded float64 cast on MPS.

    In the serialized graph, `local_skeleton_state_to_skeleton_state` inlines
    a cast `aten::to(..., dtype=7)` (float64). MPS does not support float64.
    We replace only this cast target dtype with the input tensor dtype.
    """
    skel = ts_module.character_torch.skeleton
    method = skel._c._get_method("local_skeleton_state_to_skeleton_state")
    graph = method.graph

    torch._C._jit_pass_inline(graph)

    input_dtype = None
    for node in graph.nodes():
        if node.kind() == "prim::dtype":
            input_dtype = node.output()
            break
    if input_dtype is None:
        return False

    changed = 0
    for node in graph.nodes():
        if node.kind() != "aten::to":
            continue
        try:
            dtype_input = node.inputsAt(1)
            dtype_const = dtype_input.node()
        except Exception:
            continue
        if dtype_const.kind() != "prim::Constant" or not dtype_const.hasAttribute("value"):
            continue
        try:
            if dtype_const.i("value") != 7:
                continue
        except Exception:
            continue
        node.replaceInput(1, input_dtype)
        changed += 1
        break

    if changed == 0:
        return False

    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    return True


def add_sam3d_repo_to_path(sam3d_code_root: Path) -> Path:
    """Prepend the SAM 3D Body repo to sys.path so its packages import; return the resolved path."""
    resolved = sam3d_code_root.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"SAM 3D code root not found: {resolved}")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    return resolved


def add_optional_repo_to_path(repo_root: Path | None) -> Path | None:
    """Like add_sam3d_repo_to_path but tolerant: skip silently when the path is None or missing."""
    if repo_root is None:
        return None
    resolved = repo_root.expanduser().resolve()
    if not resolved.exists():
        return None
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    return resolved


def _ensure_pkg_resources_shim() -> None:
    """Install a minimal pkg_resources shim when the real one is absent.

    Some recent environments ship setuptools without pkg_resources, but SAM3
    still imports pkg_resources.resource_filename for an asset path.
    """
    try:
        import pkg_resources  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import importlib

    shim = types.ModuleType("pkg_resources")

    def resource_filename(package_or_requirement: str, resource_name: str) -> str:
        """Resolve an asset path relative to an installed package's directory."""
        package = importlib.import_module(package_or_requirement)
        base = Path(package.__file__).resolve().parent
        return str((base / resource_name).resolve())

    shim.resource_filename = resource_filename  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = shim


_OFFCUDA_COMPAT_DONE = False


def _enable_offcuda_compat() -> None:
    """Let CUDA-hardcoded SAM3 code run on a host without CUDA (e.g. macOS).

    SAM3 hardcodes ``torch.autocast(device_type="cuda", ...)``, ``.cuda()`` and
    ``device="cuda"`` in several places. On a CUDA-less machine those raise.
    This redirects them onto the best local device — MPS on Apple Silicon, else
    CPU — process-wide, once. No-op when CUDA is available, so the GPU path is
    unchanged.
    """
    global _OFFCUDA_COMPAT_DONE
    if _OFFCUDA_COMPAT_DONE or torch.cuda.is_available():
        return
    _OFFCUDA_COMPAT_DONE = True

    # Run CUDA-hardcoded code on the best available local device: the Apple
    # Silicon GPU (MPS) when present, otherwise CPU.
    target = "mps" if torch.backends.mps.is_available() else "cpu"

    def _coerce(dev):  # type: ignore[no-untyped-def]
        """Map any cuda device (torch.device or str) to the local target; leave others as-is."""
        if isinstance(dev, torch.device):
            return torch.device(target) if dev.type == "cuda" else dev
        if isinstance(dev, str) and (dev == "cuda" or dev.startswith("cuda:")):
            return target
        return dev

    # Do NOT coerce dtypes. Models want different low precisions off CUDA — SAM3
    # gets bf16 from a cuda autocast (disabled above), while SAM 3D Body's DINOv3
    # backbone is *explicitly* bf16 and stays internally consistent (bf16 ops
    # fall back to CPU on MPS). Coercing only some bf16 tensors to fp32 mixes
    # dtypes inside one matmul (bf16 weights + fp32 bias) and aborts Metal. Leave
    # dtypes untouched; the autocast-disable is enough to keep SAM3 in fp32.
    def _coerce_dtype(dt):  # type: ignore[no-untyped-def]
        return dt

    # autocast(device_type="cuda", ...) disabled (used as decorators + context managers)
    _orig_autocast_init = torch.autocast.__init__

    def _autocast_init(self, device_type, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Fully disable a cuda autocast off-CUDA; pass other device types through unchanged."""
        if device_type == "cuda":
            # On a CUDA-less host the cuda autocast was a harmless no-op. Merely
            # coercing its device_type (while a positional bf16 dtype survives in
            # *args) turns it into an ACTIVE bf16 autocast, and a bf16 matmul with
            # an fp32 bias aborts Metal on MPS ("Destination and Accumulator ...
            # different datatype"). Fully disable it so the model runs in plain
            # fp32 — exactly as it did on a machine without CUDA.
            return _orig_autocast_init(self, "cpu", enabled=False)
        return _orig_autocast_init(self, device_type, *args, **kwargs)

    torch.autocast.__init__ = _autocast_init  # type: ignore[assignment]

    # .cuda() -> .to(target). bf16 is unsupported on MPS -> coerce .bfloat16() to
    # fp32. fp16 IS supported, so leave .half() as a real fp16 cast (coercing it
    # corrupts other models' fp16 paths and crashes Metal matmuls).
    torch.Tensor.cuda = lambda self, *a, **k: self.to(target)  # type: ignore[assignment]
    torch.nn.Module.cuda = lambda self, *a, **k: self.to(target)  # type: ignore[assignment]
    # pin_memory is a CUDA host-transfer optimization; on macOS it pins to MPS
    # and corrupts device placement. Make it a no-op.
    torch.Tensor.pin_memory = lambda self, *a, **k: self  # type: ignore[assignment]

    # SAM3 sprinkles debug `torch._assert_async(...)` sanity checks. That op has
    # no MPS kernel, so each one silently falls back to the CPU — a host<->device
    # sync on every detector forward. They are assertions, not compute; disable
    # them off-CUDA to keep the detector entirely on the GPU.
    if hasattr(torch, "_assert_async"):
        torch._assert_async = lambda *a, **k: None  # type: ignore[assignment]

    # Tensor factories with a hardcoded device="cuda" kwarg -> target.
    for _name in (
        "zeros", "ones", "empty", "full", "tensor", "as_tensor", "arange",
        "randn", "rand", "randint", "eye", "linspace", "zeros_like", "ones_like",
    ):
        _orig = getattr(torch, _name, None)
        if _orig is None:
            continue

        def _wrap(orig):  # type: ignore[no-untyped-def]
            """Wrap a torch tensor factory so its device/dtype kwargs get coerced off-CUDA."""
            def _factory(*a, **k):  # type: ignore[no-untyped-def]
                if "device" in k:
                    k["device"] = _coerce(k["device"])
                if "dtype" in k:
                    k["dtype"] = _coerce_dtype(k["dtype"])
                return orig(*a, **k)
            return _factory

        setattr(torch, _name, _wrap(_orig))

    # .to(...) with a cuda device (positional or keyword) -> target; other
    # devices (e.g. mps, used by SAM 3D Body) pass through untouched.
    _orig_tensor_to = torch.Tensor.to

    def _coerce_arg(x):  # type: ignore[no-untyped-def]
        """Coerce a positional .to(...) argument, dispatching on dtype vs device."""
        return _coerce_dtype(x) if isinstance(x, torch.dtype) else _coerce(x)

    def _tensor_to(self, *a, **k):  # type: ignore[no-untyped-def]
        """Tensor.to wrapper that redirects cuda targets to the local device off-CUDA."""
        if "device" in k:
            k["device"] = _coerce(k["device"])
        if "dtype" in k:
            k["dtype"] = _coerce_dtype(k["dtype"])
        if a:
            a = tuple(_coerce_arg(x) for x in a)
        return _orig_tensor_to(self, *a, **k)

    torch.Tensor.to = _tensor_to  # type: ignore[assignment]

    _orig_module_to = torch.nn.Module.to

    def _module_to(self, *a, **k):  # type: ignore[no-untyped-def]
        """Module.to wrapper that redirects cuda targets to the local device off-CUDA."""
        if "device" in k:
            k["device"] = _coerce(k["device"])
        if "dtype" in k:
            k["dtype"] = _coerce_dtype(k["dtype"])
        if a:
            a = tuple(_coerce_arg(x) for x in a)
        return _orig_module_to(self, *a, **k)

    torch.nn.Module.to = _module_to  # type: ignore[assignment]

    print(f"SAM3: CUDA unavailable — running on {target} (cuda tensors/autocast redirected).")


def try_build_human_detector(
    detector_name: str,
    device: torch.device,
    sam3_code_root: Path | None = None,
) -> Any | None:
    """Build the SAM3 human detector, or None when detection is disabled.

    Requires the checkpoint to already be in the local HF cache (unless
    SAM3_AUTO_DETECTOR_ALLOW_DOWNLOAD=1) so runs stay offline. Detection is
    strictly SAM3 with no fallback; any load failure is re-raised.
    """
    detector_name = detector_name.strip().lower()
    if detector_name in {"", "none", "off"}:
        return None
    try:
        if detector_name == "sam3":
            add_optional_repo_to_path(sam3_code_root)
            _ensure_pkg_resources_shim()
            allow_download = os.environ.get("SAM3_AUTO_DETECTOR_ALLOW_DOWNLOAD", "0") == "1"
            if not allow_download:
                try:
                    from huggingface_hub import try_to_load_from_cache

                    ckpt_path = try_to_load_from_cache("facebook/sam3", "sam3.pt")
                except Exception:
                    ckpt_path = None
                if ckpt_path is None:
                    raise RuntimeError(
                        "SAM3 detector checkpoint not found in the local HF cache "
                        "(facebook/sam3 / sam3.pt). Pre-download it once; runs are then "
                        "fully offline. Subject detection is strictly SAM3 — no fallback."
                    )
            # Make CUDA-hardcoded SAM3 run on the local device (MPS/CPU) when
            # this host has no CUDA.
            _enable_offcuda_compat()
        # `tools` lives in the SAM 3D Body repo. The main pipeline puts it on
        # sys.path when it loads the estimator, but lighter callers (e.g. the
        # subject-detection preview) don't — ensure it's importable so the
        # detector builds regardless of entry point.
        try:
            import tools.build_detector  # noqa: F401
        except ModuleNotFoundError:
            add_sam3d_repo_to_path(
                Path(
                    os.environ.get(
                        "SAM3D_CODE_ROOT",
                        Path(__file__).resolve().parents[2] / "vendor" / "sam-3d-body-main",
                    )
                )
            )
        from tools.build_detector import HumanDetector

        # Auto device: CUDA on a GPU box, MPS on Apple Silicon, else CPU. The
        # compat shim above redirects SAM3's cuda-hardcoded paths onto it when
        # CUDA is absent.
        return HumanDetector(name=detector_name, device=str(device))
    except Exception as exc:
        raise RuntimeError(
            f"SAM3 auto detector failed to load ({exc.__class__.__name__}: {exc}). "
            f"Subject detection is strictly SAM3 — no torchvision fallback."
        ) from exc


def select_device(force_cpu: bool = False) -> torch.device:
    """Pick the fastest available device: CUDA > MPS (Apple Silicon) > CPU."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def patch_sam3d_cuda_assumptions() -> None:
    """Monkey-patch SAM3DBody.get_ray_condition to build its meshgrid on the batch's device.

    Idempotent; must run after importing the local SAM 3D Body repository. The
    upstream method assumes a CUDA tensor, which breaks on MPS/CPU.
    """
    from sam_3d_body.models.meta_arch.sam3d_body import SAM3DBody

    if getattr(SAM3DBody, "_sam3d_pose_patch_applied", False):
        return

    def get_ray_condition_device_aware(self: Any, batch: dict[str, Any]) -> torch.Tensor:
        """Device-aware replacement: allocate the ray meshgrid on the input batch's device."""
        bsize, num_person, _, height, width = batch["img"].shape
        device = batch["img"].device
        meshgrid_xy = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(height, device=device),
                    torch.arange(width, device=device),
                    indexing="xy",
                ),
                dim=2,
            )[None, None, :, :, :]
            .repeat(bsize, num_person, 1, 1, 1)
        )
        meshgrid_xy = (
            meshgrid_xy / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )
        meshgrid_xy = (
            meshgrid_xy
            - batch["affine_trans"][:, :, None, None, [0, 1], [2, 2]]
            / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )
        meshgrid_xy = (
            meshgrid_xy - batch["cam_int"][:, None, None, None, [0, 1], [2, 2]]
        )
        meshgrid_xy = (
            meshgrid_xy / batch["cam_int"][:, None, None, None, [0, 1], [0, 1]]
        )
        return meshgrid_xy.permute(0, 1, 4, 2, 3).to(batch["img"].dtype)

    SAM3DBody.get_ray_condition = get_ray_condition_device_aware
    SAM3DBody._sam3d_pose_patch_applied = True


def load_estimator(
    checkpoint_path: Path,
    mhr_path: Path,
    device: torch.device,
    mps_mhr_mode: str = "auto",
) -> Any:
    """Load the SAM 3D Body model and wrap it in an estimator.

    On MPS the float64-incompatible MHR is handled per mps_mhr_mode: "auto"
    tries the in-graph float32 patch and falls back to a CPU wrapper, "native"
    requires the patch, "wrapper" forces the CPU wrapper. The chosen backend is
    recorded on estimator.mhr_backend.
    """
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    model, model_cfg = load_sam_3d_body(
        checkpoint_path=str(checkpoint_path),
        device=str(device),
        mhr_path=str(mhr_path),
    )

    # Apple MPS path:
    # 1) Try a TorchScript graph patch that removes hard-coded float64 cast in MHR.
    # 2) Fallback to CPU wrapper for MHR if patch is unavailable/fails.
    mhr_backend = "native"
    if device.type == "mps":
        mode = mps_mhr_mode.strip().lower()
        use_wrapper = mode == "wrapper"
        if mode in {"auto", "native"}:
            try:
                patched_body = _patch_mhr_torchscript_for_mps_float32(model.head_pose.mhr)
                patched_hand = _patch_mhr_torchscript_for_mps_float32(model.head_pose_hand.mhr)
                if patched_body and patched_hand:
                    mhr_backend = "native_mps_patched"
                else:
                    use_wrapper = True
                    if mode == "native":
                        raise RuntimeError("Unable to patch MHR TorchScript graph for MPS.")
            except Exception:
                if mode == "native":
                    raise
                use_wrapper = True
        elif mode != "wrapper":
            use_wrapper = True

        if use_wrapper:
            model.head_pose.mhr = _MHRMPSWrapper(model.head_pose.mhr)
            model.head_pose_hand.mhr = _MHRMPSWrapper(model.head_pose_hand.mhr)
            mhr_backend = "cpu_wrapper_on_mps"

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )
    setattr(estimator, "mhr_backend", mhr_backend)
    return estimator


def infer_single_person_from_bbox(
    estimator: Any,
    frame_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    inference_dtype: torch.dtype = torch.float32,
    inference_target: str = "body",
) -> dict[str, np.ndarray | float | str]:
    """Run SAM 3D Body on one cropped person and return its predictions as numpy arrays.

    inference_target selects the body vs hand ("non-full") head; the hand head
    falls back to the body head when unavailable. Only the tensors this pipeline
    consumes are transferred off the device, for performance.
    """
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.utils import recursive_to

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = bbox_xyxy.reshape(1, 4).astype(np.float32)

    batch = prepare_batch(image_rgb, estimator.transform, boxes, masks=None, masks_score=None)
    batch = recursive_to(batch, estimator.device)

    estimator.model._initialize_batch(batch)
    use_autocast = (
        inference_dtype == torch.float16
        and estimator.device.type in {"mps", "cuda"}
    )
    target_value = str(inference_target).strip().lower()
    is_non_full_target = target_value in {"hand", "partial", "part", "non_full", "non-full"}

    def _run_model(target_mode: str) -> dict[str, Any]:
        """Run a forward pass under inference mode, with fp16 autocast only when applicable."""
        with torch.inference_mode():
            with (
                torch.autocast(device_type=estimator.device.type, dtype=inference_dtype)
                if use_autocast
                else nullcontext()
            ):
                return estimator.model.run_inference(
                    image_rgb,
                    batch,
                    inference_type=target_mode,
                    transform_hand=estimator.transform_hand,
                    thresh_wrist_angle=estimator.thresh_wrist_angle,
                )

    inference_type = "hand" if is_non_full_target else "body"
    pose_output = _run_model(inference_type)

    # Critical perf path: transfer only tensors we actually consume in this pipeline.
    mhr_key = "mhr_hand" if inference_type == "hand" else "mhr"
    mhr = pose_output.get(mhr_key) if isinstance(pose_output, dict) else None
    if not isinstance(mhr, dict) and inference_type == "hand" and is_non_full_target:
        # Non-full mode: if the dedicated hand head is unavailable, retry with body head,
        # then let the pipeline crop strictly to the visible local region.
        inference_type = "body"
        pose_output = _run_model(inference_type)
        mhr = pose_output.get("mhr") if isinstance(pose_output, dict) else None
    if not isinstance(mhr, dict) and inference_type == "body":
        mhr = pose_output.get("mhr") if isinstance(pose_output, dict) else None
    if not isinstance(mhr, dict):
        raise RuntimeError(
            f"Unexpected SAM3D inference output format for target '{inference_type}'."
        )

    def _to_numpy(value: Any) -> np.ndarray:
        """Convert a tensor/array to a CPU numpy array, normalizing floats to float32."""
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            arr = value.detach().to("cpu").numpy()
            if np.issubdtype(arr.dtype, np.floating):
                return arr.astype(np.float32, copy=False)
            return arr
        raise TypeError(f"Unsupported output type for tensor conversion: {type(value)}")

    def _first_required(key: str) -> np.ndarray:
        """Fetch a required mhr field as numpy, dropping the leading batch dim; raise if absent."""
        if key not in mhr:
            raise KeyError(f"SAM3D output missing key: {key}")
        arr = _to_numpy(mhr[key])
        if arr.ndim == 0:
            return arr
        return arr[0]

    def _first_optional(key: str) -> np.ndarray | None:
        """Like _first_required but returns None when the key is missing instead of raising."""
        if key not in mhr:
            return None
        arr = _to_numpy(mhr[key])
        if arr.ndim == 0:
            return arr
        return arr[0]

    batch_bbox = _to_numpy(batch["bbox"])
    bbox = batch_bbox.reshape(-1, 4)[0] if batch_bbox.size else boxes[0]
    focal_length_arr = _first_required("focal_length")

    result: dict[str, Any] = {
        "bbox": bbox.astype(np.float32, copy=False),
        "focal_length": float(np.asarray(focal_length_arr).reshape(-1)[0]),
        "pred_keypoints_3d": _first_required("pred_keypoints_3d"),
        "pred_keypoints_2d": _first_required("pred_keypoints_2d"),
        "pred_vertices": _first_required("pred_vertices"),
        "pred_cam_t": _first_required("pred_cam_t"),
        "pred_pose_raw": _first_optional("pred_pose_raw"),
        "global_rot": _first_optional("global_rot"),
        "body_pose_params": _first_optional("body_pose"),
        "hand_pose_params": _first_optional("hand"),
        "scale_params": _first_optional("scale"),
        "shape_params": _first_optional("shape"),
        "expr_params": _first_optional("face"),
        "pred_joint_coords": _first_optional("pred_joint_coords"),
        "pred_global_rots": _first_optional("joint_global_rots"),
        "mhr_model_params": _first_optional("mhr_model_params"),
        "inference_type": inference_type,
    }
    return result
