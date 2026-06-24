"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";

import "./wizard.css";
import { Stepper } from "./stepper";
import {
  WizardActionsProvider,
  WizardProvider,
  useWizard,
  type WizardActions,
} from "./state";
import { UploadStep } from "./steps/upload-step";
import { DetectStep } from "./steps/detect-step";
import { RunStep } from "./steps/run-step";

/** Renders the current step component based on wizard state. */
export function WizardCurrentStep() {
  const { state } = useWizard();
  switch (state.step) {
    case "upload": return <UploadStep />;
    case "detect": return <DetectStep />;
    case "run": return <RunStep />;
    default: return null;
  }
}

/** Embeddable wizard panel: stepper + step content, no outer chrome. */
export function WizardPanel({
  onClose,
  actions,
}: {
  onClose?: () => void;
  actions?: WizardActions;
}) {
  return (
    <WizardActionsProvider actions={actions ?? {}}>
      <header className="of-topbar">
        <Stepper />
        {onClose ? (
          <div className="of-topbar-actions">
            <button className="of-btn is-ghost is-sm" type="button" onClick={onClose}>
              Close
            </button>
          </div>
        ) : null}
      </header>
      <main className="of-main">
        <WizardCurrentStep />
      </main>
    </WizardActionsProvider>
  );
}

/** Standalone wizard route (/process). On run completion, navigates back to / with the run pre-selected. */
export function WizardShell() {
  const router = useRouter();
  const actions: WizardActions = {
    onViewResults: (runId: string) => {
      router.push(`/?run=${encodeURIComponent(runId)}`);
    },
  };
  return (
    <WizardProvider>
      <div className="kinesia-process">
        <header className="of-topbar">
          <div className="of-brand">
            <span className="of-brand-mark">F</span>
            <span>Kinesia · guided processing</span>
          </div>
          <Stepper />
          <div className="of-topbar-actions">
            <Link href="/" className="of-btn is-ghost is-sm">
              ← Back to viewer
            </Link>
          </div>
        </header>
        <WizardActionsProvider actions={actions}>
          <main className="of-main">
            <WizardCurrentStep />
          </main>
        </WizardActionsProvider>
      </div>
    </WizardProvider>
  );
}
