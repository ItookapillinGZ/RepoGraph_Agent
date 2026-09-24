import type { VerificationArtifact } from "@/lib/types";

export function VerificationPanel({ verification }: { verification?: VerificationArtifact }) {
  if (!verification) return <EmptyArtifact title="Verification" message="Verification evidence is not available yet." />;
  const test = verification.test_result;
  const findings = verification.static_analysis.flatMap((item) => item.result.findings.map((finding) => ({ ...finding, path: item.path })));
  const tools = Array.from(new Set(verification.static_analysis.flatMap((item) => item.result.tools_run)));
  return (
    <section className="panel artifact-panel" id="verification">
      <div className="panel-heading"><div><p className="section-index">Evidence</p><h2>Verification</h2></div><span className={`status-chip status-${verification.status}`}>{verification.status}</span></div>
      <div className="artifact-body verification-grid">
        <article className="evidence-card"><div className="evidence-title"><h3>Static analysis</h3><span>{findings.length} findings</span></div><p className="tool-line">{tools.length ? tools.join(" · ") : "No tools reported"}</p>{findings.length ? <ul className="finding-list">{findings.map((finding, index) => <li key={`${finding.path}-${index}`}><span className={`severity severity-${finding.severity}`}>{finding.severity}</span><div><code>{finding.path}{finding.line_number ? `:${finding.line_number}` : ""}</code><strong>{finding.tool} {finding.rule_id}</strong><p>{finding.message}</p></div></li>)}</ul> : <p className="quiet-success">No static-analysis findings were reported.</p>}</article>
        <article className="evidence-card"><div className="evidence-title"><h3>Targeted tests</h3><span className={`status-chip status-${test.status}`}>{test.status.replaceAll("_", " ")}</span></div><dl className="evidence-facts"><div><dt>Framework</dt><dd>{test.framework ?? "—"}</dd></div><div><dt>Exit code</dt><dd>{test.exit_code ?? "—"}</dd></div><div><dt>Files</dt><dd>{test.test_files.length}</dd></div></dl>{test.test_files.length > 0 && <div className="code-list">{test.test_files.map((file) => <code key={file}>{file}</code>)}</div>}{test.stdout && <OutputBlock label="stdout" value={test.stdout} />}{test.stderr && <OutputBlock label="stderr" value={test.stderr} />}{test.status === "not_run" && <p className="notice-text">Tests were not run; Studio does not present this as a pass.</p>}</article>
      </div>
    </section>
  );
}

function OutputBlock({ label, value }: { label: string; value: string }) { return <details className="output-block"><summary>{label}</summary><pre>{value}</pre></details>; }
function EmptyArtifact({ title, message }: { title: string; message: string }) { return <section className="panel artifact-panel"><div className="panel-heading"><div><p className="section-index">Evidence</p><h2>{title}</h2></div></div><p className="empty-state">{message}</p></section>; }
