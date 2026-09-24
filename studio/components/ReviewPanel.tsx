import type { ChangeSetReviewArtifact } from "@/lib/types";

export function ReviewPanel({ review }: { review?: ChangeSetReviewArtifact }) {
  return (
    <section className="panel artifact-panel" id="review">
      <div className="panel-heading"><div><p className="section-index">Evidence</p><h2>Change-set review</h2></div>{review && <span className={`status-chip rating-${review.overall_rating}`}>{review.overall_rating.replaceAll("_", " ")}</span>}</div>
      {!review ? <p className="empty-state">Review evidence is not available yet.</p> : (
        <div className="artifact-body">
          <p className="artifact-summary">{review.summary}</p>
          {review.high_risk_files.length > 0 && <div className="risk-strip"><strong>High-risk files</strong>{review.high_risk_files.map((file) => <code key={file}>{file}</code>)}</div>}
          <div className="file-reviews">
            {review.file_results.map((file) => (
              <article key={file.target.path}>
                <header><div><code>{file.target.path}</code><small>{file.target.change_kind} · {file.status}</small></div>{file.review && <span className={`status-chip rating-${file.review.overall_rating}`}>{file.review.overall_rating.replaceAll("_", " ")}</span>}</header>
                {file.review && <><p>{file.review.summary}</p>{file.review.findings.length ? <div className="review-findings">{file.review.findings.map((finding, index) => <div className="review-finding" key={`${finding.title}-${index}`}><span className={`severity severity-${finding.severity}`}>{finding.severity}</span><div><strong>{finding.title}</strong><small>{finding.category}{finding.line_number ? ` · line ${finding.line_number}` : ""}</small><p>{finding.description}</p><blockquote>{finding.suggestion}</blockquote></div></div>)}</div> : <p className="quiet-success">No actionable findings.</p>}</>}
              </article>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
