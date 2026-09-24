import type { EngineeringPlan } from "@/lib/types";

export function PlanPanel({ plan }: { plan?: EngineeringPlan }) {
  return (
    <section className="panel artifact-panel" id="plan">
      <div className="panel-heading"><div><p className="section-index">Artifact</p><h2>Engineering plan</h2></div>{plan && <span className="count-badge">{plan.files.length} files</span>}</div>
      {!plan ? <p className="empty-state">Plan artifact is not available yet.</p> : (
        <div className="artifact-body">
          <p className="artifact-summary">{plan.summary}</p>
          <div className="plan-files">
            {plan.files.map((file) => (
              <article key={file.path}>
                <span className={`action-label action-${file.action}`}>{file.action}</span>
                <div><code>{file.path}</code><p>{file.rationale}</p></div>
              </article>
            ))}
          </div>
          <div className="artifact-columns">
            <ArtifactList title="Tests" items={plan.tests.map((test) => `${test.path ?? "Target TBD"} — ${test.purpose}`)} empty="No planned tests." />
            <ArtifactList title="Risks" items={plan.risks} empty="No explicit risks." />
            <ArtifactList title="Assumptions" items={plan.assumptions} empty="No assumptions recorded." />
          </div>
        </div>
      )}
    </section>
  );
}

function ArtifactList({ title, items, empty }: { title: string; items: string[]; empty: string }) {
  return <div className="artifact-list"><h3>{title}</h3>{items.length ? <ul>{items.map((item, index) => <li key={`${title}-${index}`}>{item}</li>)}</ul> : <p>{empty}</p>}</div>;
}
