import type { CorrectionHistoryArtifact } from "@/lib/types";

export function CorrectionPanel({ history }: { history?: CorrectionHistoryArtifact }) {
  return (
    <section className="panel artifact-panel" id="corrections">
      <div className="panel-heading"><div><p className="section-index">History</p><h2>Correction history</h2></div>{history && <span className="count-badge">{history.correction_rounds_used}</span>}</div>
      {!history ? <p className="empty-state">Correction history is not available yet.</p> : history.correction_rounds_used === 0 ? <p className="quiet-success correction-empty">No correction was required.</p> : (
        <div className="attempt-list">
          {history.attempt_history.map((attempt, index) => (
            <article key={attempt.attempt}>
              <div className="attempt-rail"><span>{attempt.attempt}</span>{index < history.attempt_history.length - 1 && <i />}</div>
              <div className="attempt-content"><header><div><p className="section-index">{index === 0 ? "Candidate v1" : `Correction round ${index}`}</p><h3>Candidate v{index + 1}</h3></div><div><span className={`status-chip status-${attempt.verification_status}`}>{attempt.verification_status}</span>{attempt.review_rating && <span className={`status-chip rating-${attempt.review_rating}`}>{attempt.review_rating.replaceAll("_", " ")}</span>}</div></header><p>{attempt.diff_summary}</p>{attempt.failure_reasons.length > 0 && <ul>{attempt.failure_reasons.map((reason, reasonIndex) => <li key={reasonIndex}>{reason}</li>)}</ul>}</div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
