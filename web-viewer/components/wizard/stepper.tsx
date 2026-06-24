"use client";

import { STEP_LABELS, STEP_ORDER, useWizard, type WizardStep } from "./state";

/** Pipeline progress nav: a clickable step per stage, gated by canGoTo. */
export function Stepper() {
  const { state, goTo, canGoTo } = useWizard();
  const currentIdx = STEP_ORDER.indexOf(state.step);

  return (
    <nav className="of-stepper" aria-label="Processing pipeline">
      {STEP_ORDER.map((step: WizardStep, idx: number) => {
        const isCurrent = step === state.step;
        const isDone = idx < currentIdx;
        const reachable = canGoTo(step);
        return (
          <button
            key={step}
            type="button"
            className={`of-step ${isCurrent ? "is-current" : ""} ${isDone ? "is-done" : ""}`}
            onClick={() => goTo(step)}
            disabled={!reachable}
            aria-current={isCurrent ? "step" : undefined}
          >
            <span className="of-step-index">{isDone ? "✓" : idx + 1}</span>
            <span className="of-step-label">{STEP_LABELS[step]}</span>
          </button>
        );
      })}
    </nav>
  );
}
