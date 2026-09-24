import Link from "next/link";

import type { RunDetail } from "@/lib/types";

export function RunStatusHeader({ run, connected }: { run: RunDetail; connected: boolean }) {
  return (
    <header className="run-header">
      <div className="breadcrumb"><Link href="/">Runs</Link><span>/</span><span>{run.id.slice(0, 8)}</span></div>
      <div className="run-title-row">
        <div>
          <p className="eyebrow">{run.repository_id}</p>
          <h1>{run.task}</h1>
        </div>
        <div className="run-state-stack">
          <span className={`status-chip large status-${run.status}`}><i aria-hidden="true" />{run.status.replaceAll("_", " ")}</span>
          <span className={`connection-state ${connected ? "online" : ""}`}>
            <i aria-hidden="true" />{connected ? "Live timeline" : "Reconnecting"}
          </span>
        </div>
      </div>
      <div className="run-facts">
        <span><small>Phase</small>{run.phase.replaceAll("_", " ")}</span>
        <span><small>Created</small>{new Date(run.created_at).toLocaleString()}</span>
        <span><small>Run ID</small><code>{run.id}</code></span>
      </div>
      {run.failure_reason && <p className="error-banner" role="alert">{run.failure_reason}</p>}
    </header>
  );
}
