import { ViewerShell } from "../components/viewer-shell";
import { WizardProvider } from "../components/wizard/state";

export default function HomePage() {
  return (
    <WizardProvider>
      <ViewerShell />
    </WizardProvider>
  );
}
