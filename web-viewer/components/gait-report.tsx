"use client";

// Clinical gait-cycle report: the deliverables a gait lab reads, rendered from
// the run's gait layer (see src/sam_3d_pose_estimation/gait.py):
//
//   - spatiotemporal parameter cards (cadence, speed, stride, stance/swing, ...),
//     each mean +/- SD;
//   - per-joint cycle curves normalized to 0-100% of the gait cycle, left and
//     right overlaid, each a mean line inside a +/- 1 SD band, with the stance
//     phase shaded and toe-off marked;
//   - the calibration provenance (quiet-stance neutral reference + filter).
//
// With several subjects selected the same plots compare them directly: colour
// then encodes the subject and the line style the side (solid left, dashed
// right), and the parameters become one row per subject.
//
// Pure SVG so it is crisp at any size and carries no charting dependency. The
// curves are static (aggregated over every stride), so this view does not track
// the playhead.

import { useMemo } from "react";
import type { GaitCycle, GaitStat, RunGait } from "../lib/types";

const LEFT_COLOR = "#22d3ee"; // cyan — matches the viewer's --cyan
const RIGHT_COLOR = "#fb923c"; // orange — matches --orange
const GRID = "rgba(148, 163, 184, 0.14)";
const ZERO = "rgba(148, 163, 184, 0.4)";
const AXIS_TEXT = "#8290aa";
const STANCE_FILL = "rgba(148, 163, 184, 0.08)";
const TOE_OFF = "rgba(148, 163, 184, 0.5)";

type JointSpec = { title: string; unit: string; left: string; right: string };

const JOINTS: JointSpec[] = [
  { title: "Hip Flexion", unit: "°", left: "gait.hip.left.flexion_deg", right: "gait.hip.right.flexion_deg" },
  { title: "Knee Flexion", unit: "°", left: "gait.knee.left.flexion_deg", right: "gait.knee.right.flexion_deg" },
  { title: "Ankle Dorsiflexion", unit: "°", left: "gait.ankle.left.dorsiflexion_deg", right: "gait.ankle.right.dorsiflexion_deg" },
];

// Plot geometry in SVG user units (scaled responsively by width: 100%).
const W = 340;
const H = 216;
const ML = 36;
const MR = 12;
const MT = 12;
const MB = 30;
const PLOT_W = W - ML - MR;
const PLOT_H = H - MT - MB;

function fmtStat(stat: GaitStat, digits: number, unit: string): { value: string; spread: string | null } {
  if (stat.mean == null || !Number.isFinite(stat.mean)) return { value: "—", spread: null };
  const value = `${stat.mean.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
  const spread = stat.sd != null && Number.isFinite(stat.sd) && stat.n > 1 ? `± ${stat.sd.toFixed(digits)}` : null;
  return { value, spread };
}

function niceStep(range: number): number {
  const raw = range / 4;
  const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const norm = raw / mag;
  const step = norm >= 5 ? 5 : norm >= 2 ? 2 : 1;
  return step * mag;
}

// Shared y-range for a joint's two sides, padded and including zero.
function yRange(cycles: Array<GaitCycle | undefined>): [number, number] {
  let lo = 0;
  let hi = 0;
  let seen = false;
  for (const c of cycles) {
    if (!c || !c.mean) continue;
    const sd = c.sd ?? [];
    for (let i = 0; i < c.mean.length; i += 1) {
      const m = c.mean[i];
      if (!Number.isFinite(m)) continue;
      const s = Number.isFinite(sd[i]) ? sd[i] : 0;
      lo = Math.min(lo, m - s);
      hi = Math.max(hi, m + s);
      seen = true;
    }
  }
  if (!seen) return [-1, 1];
  const pad = (hi - lo) * 0.08 || 1;
  return [lo - pad, hi + pad];
}

// One subject's gait layer, with the identity it carries across the whole UI.
export type GaitSubject = { runId: string; label: string; color: string; gait: RunGait };

type CyclePlotProps = { spec: JointSpec; subjects: GaitSubject[]; stancePct: number | null };

// Curves drawn on one joint's plot: one per (subject, side) that has cycles.
type Trace = { key: string; color: string; dashed: boolean; cycle: GaitCycle; label: string };

function tracesFor(spec: JointSpec, subjects: GaitSubject[]): Trace[] {
  const multi = subjects.length > 1;
  const out: Trace[] = [];
  for (const subject of subjects) {
    const left = subject.gait.cycles.left[spec.left];
    const right = subject.gait.cycles.right[spec.right];
    if (left?.mean) {
      out.push({
        key: `${subject.runId}:left`,
        // One subject: the familiar left/right colours. Several: colour is the
        // subject, so the side has to be carried by the line style instead.
        color: multi ? subject.color : LEFT_COLOR,
        dashed: false,
        cycle: left,
        label: multi ? `${subject.label} L` : "Left",
      });
    }
    if (right?.mean) {
      out.push({
        key: `${subject.runId}:right`,
        color: multi ? subject.color : RIGHT_COLOR,
        dashed: multi,
        cycle: right,
        label: multi ? `${subject.label} R` : "Right",
      });
    }
  }
  return out;
}

function CyclePlot({ spec, subjects, stancePct }: CyclePlotProps) {
  const traces = useMemo(() => tracesFor(spec, subjects), [spec, subjects]);
  const [ymin, ymax] = useMemo(() => yRange(traces.map((t) => t.cycle)), [traces]);

  const x = (pct: number) => ML + (pct / 100) * PLOT_W;
  const y = (v: number) => MT + PLOT_H - ((v - ymin) / (ymax - ymin || 1)) * PLOT_H;

  const step = niceStep(ymax - ymin);
  const ticks: number[] = [];
  for (let t = Math.ceil(ymin / step) * step; t <= ymax + 1e-6; t += step) ticks.push(t);

  const meanPath = (c: GaitCycle | undefined): string => {
    if (!c || !c.mean) return "";
    const n = c.mean.length;
    return c.mean
      .map((m, i) => `${i === 0 ? "M" : "L"}${x((i / (n - 1)) * 100).toFixed(1)},${y(m).toFixed(1)}`)
      .join(" ");
  };
  const bandPath = (c: GaitCycle | undefined): string => {
    if (!c || !c.mean || !c.sd) return "";
    const n = c.mean.length;
    const up: string[] = [];
    const down: string[] = [];
    for (let i = 0; i < n; i += 1) {
      const px = x((i / (n - 1)) * 100);
      up.push(`${i === 0 ? "M" : "L"}${px.toFixed(1)},${y(c.mean[i] + (c.sd[i] || 0)).toFixed(1)}`);
      down.push(`L${x(((n - 1 - i) / (n - 1)) * 100).toFixed(1)},${y(c.mean[n - 1 - i] - (c.sd[n - 1 - i] || 0)).toFixed(1)}`);
    }
    return `${up.join(" ")} ${down.join(" ")} Z`;
  };

  const hasData = traces.length > 0;
  const toeOffX = stancePct != null && stancePct > 0 && stancePct < 100 ? x(stancePct) : null;

  return (
    <div className="gait-plot">
      <div className="gait-plot-head">
        <span className="gait-plot-title">{spec.title}</span>
        <span className="gait-plot-unit">({spec.unit})</span>
        {/* Cycle counts: per side for a single subject, but collapsed to one
            total per subject when comparing, or the header would not fit. */}
        <span className="gait-plot-counts">
          {subjects.length > 1
            ? subjects.map((s) => {
                const n =
                  (s.gait.cycles.left[spec.left]?.n_cycles ?? 0) +
                  (s.gait.cycles.right[spec.right]?.n_cycles ?? 0);
                if (n === 0) return null;
                return (
                  <em key={s.runId} style={{ color: s.color }} title={`${s.label}: ${n} cycles`}>
                    {n}
                  </em>
                );
              })
            : traces.map((t) => (
                <em key={t.key} style={{ color: t.color }}>
                  {t.label} {t.cycle.n_cycles}
                </em>
              ))}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="gait-plot-svg" role="img" aria-label={`${spec.title} over the gait cycle`}>
        {/* stance shading + toe-off */}
        {toeOffX != null ? (
          <>
            <rect x={ML} y={MT} width={toeOffX - ML} height={PLOT_H} fill={STANCE_FILL} />
            <line x1={toeOffX} x2={toeOffX} y1={MT} y2={MT + PLOT_H} stroke={TOE_OFF} strokeWidth={1} strokeDasharray="3 3" />
          </>
        ) : null}
        {/* y gridlines + ticks */}
        {ticks.map((t) => (
          <g key={`y${t}`}>
            <line x1={ML} x2={W - MR} y1={y(t)} y2={y(t)} stroke={Math.abs(t) < 1e-6 ? ZERO : GRID} strokeWidth={1} />
            <text x={ML - 4} y={y(t) + 3} textAnchor="end" fontSize={9} fill={AXIS_TEXT}>
              {step >= 1 ? t.toFixed(0) : t.toFixed(1)}
            </text>
          </g>
        ))}
        {/* x ticks */}
        {[0, 25, 50, 75, 100].map((p) => (
          <g key={`x${p}`}>
            <line x1={x(p)} x2={x(p)} y1={MT + PLOT_H} y2={MT + PLOT_H + 3} stroke={AXIS_TEXT} strokeWidth={1} />
            <text x={x(p)} y={H - 8} textAnchor="middle" fontSize={9} fill={AXIS_TEXT}>
              {p}
            </text>
          </g>
        ))}
        <text x={ML + PLOT_W / 2} y={H} textAnchor="middle" fontSize={9} fill={AXIS_TEXT}>
          % gait cycle
        </text>
        {/* SD bands first, then every mean line on top, so no curve is buried */}
        {traces.map((t) => (
          <path key={`band-${t.key}`} d={bandPath(t.cycle)} fill={t.color} opacity={traces.length > 2 ? 0.1 : 0.15} />
        ))}
        {traces.map((t) => (
          <path
            key={`mean-${t.key}`}
            d={meanPath(t.cycle)}
            fill="none"
            stroke={t.color}
            strokeWidth={1.8}
            strokeDasharray={t.dashed ? "6 3" : undefined}
            strokeLinejoin="round"
          />
        ))}
        {!hasData ? (
          <text x={ML + PLOT_W / 2} y={MT + PLOT_H / 2} textAnchor="middle" fontSize={11} fill={AXIS_TEXT}>
            No cycles
          </text>
        ) : null}
      </svg>
    </div>
  );
}

function ParamCard({ label, stat, digits, unit }: { label: string; stat: GaitStat; digits: number; unit: string }) {
  const { value, spread } = fmtStat(stat, digits, unit);
  return (
    <div className="gait-card">
      <span className="gait-card-label">{label}</span>
      <strong className="gait-card-value">{value}</strong>
      {spread ? <span className="gait-card-spread">{spread}</span> : <span className="gait-card-spread muted">n = {stat.n}</span>}
    </div>
  );
}

// The spatiotemporal parameters of several subjects, one row each, so they can
// be read against one another at a glance.
function ParamTable({ subjects }: { subjects: GaitSubject[] }) {
  const columns: Array<{ label: string; get: (g: RunGait) => GaitStat; digits: number; unit: string }> = [
    { label: "Speed", get: (g) => g.spatiotemporal.walkingSpeedMS, digits: 2, unit: "m/s" },
    { label: "Stride len.", get: (g) => g.spatiotemporal.strideLengthM, digits: 2, unit: "m" },
    { label: "Stride time", get: (g) => g.spatiotemporal.strideTimeS, digits: 2, unit: "s" },
    { label: "Step len.", get: (g) => g.spatiotemporal.stepLengthM, digits: 2, unit: "m" },
    { label: "Stance", get: (g) => g.spatiotemporal.stancePct, digits: 1, unit: "%" },
    { label: "Double sup.", get: (g) => g.spatiotemporal.doubleSupportPct, digits: 1, unit: "%" },
  ];
  return (
    <div className="gait-table-wrap">
      <table className="gait-table">
        <thead>
          <tr>
            <th>Subject</th>
            <th>Cadence</th>
            {columns.map((c) => (
              <th key={c.label}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {subjects.map((subject) => {
            const st = subject.gait.spatiotemporal;
            return (
              <tr key={subject.runId}>
                <th scope="row">
                  <span className="gait-table-subject">
                    <i className="chip-swatch" style={{ background: subject.color }} />
                    {subject.label}
                  </span>
                </th>
                <td>{st.cadenceStepsPerMin != null ? st.cadenceStepsPerMin.toFixed(0) : "—"}</td>
                {columns.map((c) => {
                  const { value, spread } = fmtStat(c.get(subject.gait), c.digits, "");
                  return (
                    <td key={c.label}>
                      {value}
                      {spread ? <small>{spread}</small> : null}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function GaitReport({ subjects }: { subjects: GaitSubject[] }) {
  const walking = subjects.filter((s) => s.gait.spatiotemporal.walkingDetected);
  const multi = subjects.length > 1;
  const primary = subjects[0];
  // Toe-off marker: the mean stance fraction across whichever subjects walk.
  const stancePct = useMemo(() => {
    const values = walking.map((s) => s.gait.spatiotemporal.stancePct.mean).filter((v): v is number => v != null);
    return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
  }, [walking]);

  if (!primary) return null;
  const st = primary.gait.spatiotemporal;
  const nr = primary.gait.neutralReference;

  return (
    <div className="gait-report">
      <div className="gait-report-head">
        <div className="gait-legend">
          {multi ? (
            <>
              {subjects.map((s) => (
                <span key={s.runId}>
                  <i style={{ background: s.color }} /> {s.label}
                </span>
              ))}
              <span className="gait-legend-band">solid = left · dashed = right · band = ± 1 SD</span>
            </>
          ) : (
            <>
              <span><i style={{ background: LEFT_COLOR }} /> Left</span>
              <span><i style={{ background: RIGHT_COLOR }} /> Right</span>
              <span className="gait-legend-band">shaded band = ± 1 SD across cycles</span>
            </>
          )}
        </div>
      </div>

      {walking.length > 0 ? (
        <>
          {multi ? (
            <ParamTable subjects={walking} />
          ) : (
            <div className="gait-cards">
              <div className="gait-card">
                <span className="gait-card-label">Cadence</span>
                <strong className="gait-card-value">
                  {st.cadenceStepsPerMin != null ? `${st.cadenceStepsPerMin.toFixed(0)} steps/min` : "—"}
                </strong>
                <span className="gait-card-spread muted">whole trial</span>
              </div>
              <ParamCard label="Walking speed" stat={st.walkingSpeedMS} digits={2} unit="m/s" />
              <ParamCard label="Stride length" stat={st.strideLengthM} digits={2} unit="m" />
              <ParamCard label="Stride time" stat={st.strideTimeS} digits={2} unit="s" />
              <ParamCard label="Step length" stat={st.stepLengthM} digits={2} unit="m" />
              <ParamCard label="Stance" stat={st.stancePct} digits={1} unit="%" />
              <ParamCard label="Swing" stat={st.swingPct} digits={1} unit="%" />
              <ParamCard label="Double support" stat={st.doubleSupportPct} digits={1} unit="%" />
            </div>
          )}

          <div className="gait-plots">
            {JOINTS.map((spec) => (
              <CyclePlot key={spec.title} spec={spec} subjects={walking} stancePct={stancePct} />
            ))}
          </div>
          {multi && walking.length < subjects.length ? (
            <p className="gait-note">
              Not walking in this clip:{" "}
              {subjects.filter((s) => !s.gait.spatiotemporal.walkingDetected).map((s) => s.label).join(", ")}.
            </p>
          ) : null}
        </>
      ) : (
        <div className="gait-empty">
          <strong>No gait cycles detected</strong>
          <p>
            {multi ? "These subjects are standing" : "The subject is standing"} rather than
            walking, so there are no strides to normalize. The cycle curves and
            spatiotemporal parameters appear for walking clips.
          </p>
        </div>
      )}

      <div className="gait-provenance">
        <span>
          Zero-phase {primary.gait.params.filter.charAt(0).toUpperCase() + primary.gait.params.filter.slice(1)} filter
          (order {primary.gait.params.order}, {primary.gait.params.cutoff_hz.toFixed(0)} Hz).
        </span>
        {nr.applied ? (
          <span>
            {multi ? `${primary.label}: angles` : "Angles"} referenced to the subject&apos;s quiet
            stance ({nr.staticDurationS.toFixed(1)} s):{" "}
            {Object.entries(nr.offsetsDeg)
              .map(([k, v]) => `${k.replace("hip.", "hip ").replace("knee.", "knee ").replace("ankle.", "ankle ")} ${v >= 0 ? "+" : ""}${v.toFixed(0)}°`)
              .join(", ")}
            {" "}subtracted.
          </span>
        ) : (
          <span>{nr.note || "Raw reconstruction angles (no quiet stance found to calibrate against)."}</span>
        )}
      </div>
    </div>
  );
}

export default GaitReport;
