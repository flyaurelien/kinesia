"""Temporal filters for per-frame estimates.

Monocular reconstructions jitter from frame to frame, but people also sprint,
jump and spin, so a fixed low-pass either leaves the jitter or flattens the
motion. The One-Euro filter adapts its cutoff to the speed of the signal:
heavy smoothing when still, almost none during fast moves. Run offline, it is
applied forwards and backwards and the two passes are averaged, which cancels
its lag.
"""

from __future__ import annotations

import numpy as np

from . import quat


def _alpha(cutoff: np.ndarray | float, rate: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff))
    return 1.0 / (1.0 + tau * rate)


def one_euro(values: np.ndarray, rate: float, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> np.ndarray:
    """Causal One-Euro filter over the first axis of ``values`` (T, ...).

    The adaptive cutoff uses the norm of the derivative over the trailing axes,
    so every component of a vector (or quaternion) is smoothed consistently.
    """
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2:
        return x.copy()
    out = np.empty_like(x)
    out[0] = x[0]
    derivative = np.zeros_like(x[0])
    a_d = _alpha(d_cutoff, rate)
    for t in range(1, len(x)):
        raw_derivative = (x[t] - out[t - 1]) * rate
        derivative = a_d * raw_derivative + (1 - a_d) * derivative
        speed = np.linalg.norm(derivative) if derivative.ndim <= 1 else np.linalg.norm(derivative, axis=-1, keepdims=True)
        a = _alpha(min_cutoff + beta * speed, rate)
        out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out


def smooth(values: np.ndarray, rate: float, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> np.ndarray:
    """Zero-lag One-Euro: average of a forward and a backward pass."""
    x = np.asarray(values, dtype=np.float64)
    forward = one_euro(x, rate, min_cutoff, beta, d_cutoff)
    backward = one_euro(x[::-1], rate, min_cutoff, beta, d_cutoff)[::-1]
    return 0.5 * (forward + backward)


def smooth_quaternions(q: np.ndarray, rate: float, min_cutoff: float, beta: float) -> np.ndarray:
    """Smooth unit quaternions (T, ..., 4) along time, keeping them unit length."""
    aligned = quat.align_hemisphere(q, axis=0)
    return quat.normalize(smooth(aligned, rate, min_cutoff, beta))


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centered running median (T, ...) with a shrinking window at the ends."""
    x = np.asarray(values, dtype=np.float64)
    half = max(0, window // 2)
    out = np.empty_like(x)
    for t in range(len(x)):
        out[t] = np.median(x[max(0, t - half) : t + half + 1], axis=0)
    return out


def despike(values: np.ndarray, window: int = 5, factor: float = 4.0, floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Replace isolated outliers by the running median.

    A frame is an outlier when its distance to the running median exceeds both
    ``floor`` and ``factor`` times the typical (median) distance. Returns the
    cleaned series and the boolean mask of replaced frames.
    """
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 3:
        return x.copy(), np.zeros(len(x), dtype=bool)
    median = rolling_median(x, window)
    residual = np.linalg.norm((x - median).reshape(len(x), -1), axis=1)
    # A smooth signal has almost no residual; judge against its own frame-to-frame step too.
    step = np.linalg.norm(np.diff(x, axis=0).reshape(len(x) - 1, -1), axis=1)
    typical = max(float(np.median(residual)), float(np.median(step)), 1e-9)
    spikes = (residual > factor * typical) & (residual > floor)
    out = x.copy()
    out[spikes] = median[spikes]
    return out, spikes


def despike_quaternions(q: np.ndarray, max_angle: float, agree_angle: float) -> tuple[np.ndarray, int]:
    """Replace single-frame rotation flips with the slerp of their neighbours.

    Frame ``t`` is replaced when it is farther than ``max_angle`` (radians) from
    the midpoint of ``t-1`` and ``t+1`` while those two agree within
    ``agree_angle``: a pose that jumps for one frame and comes straight back.
    Works per joint on (T, J, 4) arrays.
    """
    out = quat.align_hemisphere(q, axis=0)
    if len(out) < 3:
        return out, 0
    replaced = 0
    for t in range(1, len(out) - 1):
        prev, nxt = out[t - 1], out[t + 1]
        mid = quat.slerp(prev, nxt, 0.5)
        bad = (quat.angle(out[t], mid) > max_angle) & (quat.angle(prev, nxt) < agree_angle)
        if np.any(bad):
            out[t][bad] = mid[bad]
            replaced += int(np.count_nonzero(bad))
    return out, replaced


def segments(frames: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Split sorted frame indices into runs whose internal gaps are at most ``max_gap``.

    Returns ``(start, end)`` index pairs into ``frames`` (end exclusive).
    """
    if len(frames) == 0:
        return []
    cuts = np.flatnonzero(np.diff(frames) > max_gap + 1) + 1
    bounds = [0, *cuts.tolist(), len(frames)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def kalman_smooth(
    measurements: np.ndarray,
    rate: float,
    accel_std: float,
    noise: np.ndarray,
) -> np.ndarray:
    """Rauch-Tung-Striebel smoother with a constant-velocity model.

    ``measurements`` is (T, D); ``noise`` is the per-frame measurement
    covariance, (T, D, D) or (D, D). ``accel_std`` (units/s^2) is how hard the
    motion may accelerate: the smoother follows sprints and turns within it
    and treats faster wiggles as measurement noise. Offline and zero-lag.
    """
    z = np.asarray(measurements, dtype=np.float64)
    count, dim = z.shape
    if count < 3:
        return z.copy()
    dt = 1.0 / rate
    F = np.eye(2 * dim)
    F[:dim, dim:] = dt * np.eye(dim)
    q = accel_std**2
    Q = np.zeros((2 * dim, 2 * dim))
    Q[:dim, :dim] = q * dt**4 / 4 * np.eye(dim)
    Q[:dim, dim:] = Q[dim:, :dim] = q * dt**3 / 2 * np.eye(dim)
    Q[dim:, dim:] = q * dt**2 * np.eye(dim)
    H = np.zeros((dim, 2 * dim))
    H[:, :dim] = np.eye(dim)
    R = np.broadcast_to(noise, (count, dim, dim))

    x = np.zeros(2 * dim)
    x[:dim] = z[0]
    P = np.eye(2 * dim)
    P[:dim, :dim] = R[0]
    P[dim:, dim:] = np.eye(dim) * 25.0  # unknown initial velocity (5 units/s)
    means, covs, pred_means, pred_covs = [], [], [], []
    for t in range(count):
        if t:
            x = F @ x
            P = F @ P @ F.T + Q
        pred_means.append(x)
        pred_covs.append(P)
        S = H @ P @ H.T + R[t]
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z[t] - H @ x)
        P = (np.eye(2 * dim) - K @ H) @ P
        means.append(x)
        covs.append(P)
    smoothed = means[-1]
    out = np.empty_like(z)
    out[-1] = smoothed[:dim]
    P_s = covs[-1]
    for t in range(count - 2, -1, -1):
        C = covs[t] @ F.T @ np.linalg.inv(pred_covs[t + 1])
        smoothed = means[t] + C @ (smoothed - pred_means[t + 1])
        P_s = covs[t] + C @ (P_s - pred_covs[t + 1]) @ C.T
        out[t] = smoothed[:dim]
    return out
