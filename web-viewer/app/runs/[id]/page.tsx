import "../../run.css";
import "../../viewer.css";

import { RunView } from "@/components/run/run-view";

export default function RunPage({ params }: { params: { id: string } }) {
  return <RunView id={params.id} />;
}
