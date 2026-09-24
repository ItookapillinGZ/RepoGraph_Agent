import type { RunTraceSummary } from "@/lib/types";

function duration(value: number | null): string {
  if (value === null) return "incomplete";
  return value >= 1_000 ? `${(value / 1_000).toFixed(2)} s` : `${value.toFixed(1)} ms`;
}

export function TracePanel({ trace }: { trace: RunTraceSummary | null }) {
  if (!trace) return null;
  return (
    <section className="panel artifact-panel" id="trace">
      <div className="panel-heading">
        <div><p className="section-index">Observability</p><h2>Run trace</h2></div>
        <span className="panel-kicker">{trace.trace_id}</span>
      </div>
      <div className="artifact-body verification-grid">
        <article className="evidence-card">
          <div className="evidence-title"><h3>Execution metrics</h3><span>{duration(trace.duration_ms)}</span></div>
          <dl className="evidence-facts">
            <div><dt>LLM calls</dt><dd>{trace.llm_calls}</dd></div>
            <div><dt>Tokens</dt><dd>{trace.total_tokens}</dd></div>
            <div><dt>Tool calls</dt><dd>{trace.tool_calls}</dd></div>
            <div><dt>Sandbox runs</dt><dd>{trace.sandbox_executions}</dd></div>
            <div><dt>Artifacts</dt><dd>{trace.artifact_count}</dd></div>
          </dl>
        </article>
        <article className="evidence-card">
          <div className="evidence-title"><h3>Artifact lineage</h3><span>immutable</span></div>
          <pre className="output-block"><code>{trace.lineage_summary || "No artifacts registered yet."}</code></pre>
        </article>
      </div>
    </section>
  );
}
