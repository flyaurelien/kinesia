// One-Euro filter (Casiez, Roussel & Vogel, CHI 2012): an adaptive low-pass
// filter for noisy interactive signals. At low speeds the cutoff drops toward
// `minCutoff`, strongly smoothing jitter (a standing body stops trembling); at
// high speeds the cutoff rises with `beta * |velocity|`, so fast real motion —
// a jump, a quick step — passes through with almost no lag. This is exactly
// the trade a fixed-alpha EMA cannot make: it is either laggy or jittery.

export type OneEuroParams = {
  minCutoff: number; // Hz — smoothing floor at rest (lower = smoother, laggier)
  beta: number; // cutoff growth per unit of speed (higher = snappier at speed)
  dCutoff: number; // Hz — cutoff for the internal derivative estimate
};

function smoothingAlpha(cutoffHz: number, dtSec: number): number {
  const tau = 1 / (2 * Math.PI * Math.max(1e-6, cutoffHz));
  return 1 / (1 + tau / Math.max(1e-6, dtSec));
}

export class OneEuroFilter {
  private prev: number | null = null;
  private prevDeriv = 0;
  private readonly params: OneEuroParams;

  constructor(params: OneEuroParams) {
    this.params = params;
  }

  next(value: number, dtSec: number): number {
    if (this.prev === null) {
      this.prev = value;
      this.prevDeriv = 0;
      return value;
    }
    const rawDeriv = (value - this.prev) / Math.max(1e-6, dtSec);
    const aD = smoothingAlpha(this.params.dCutoff, dtSec);
    this.prevDeriv = this.prevDeriv + aD * (rawDeriv - this.prevDeriv);
    const cutoff = this.params.minCutoff + this.params.beta * Math.abs(this.prevDeriv);
    const a = smoothingAlpha(cutoff, dtSec);
    this.prev = this.prev + a * (value - this.prev);
    return this.prev;
  }
}

// Filter a whole per-frame series at once (deterministic, memo-friendly).
// `null` samples (subject absent) pass through untouched and reset nothing:
// the filter simply holds its state across the gap.
export function filterSeries(
  values: Array<number | null>,
  fps: number,
  params: OneEuroParams,
): Array<number | null> {
  const dt = 1 / Math.max(1, fps);
  const f = new OneEuroFilter(params);
  return values.map((v) => (v === null || !Number.isFinite(v) ? v : f.next(v, dt)));
}
