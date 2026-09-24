import type { ArtifactMap } from "@/lib/types";

export function DeliveryPanel({ artifacts }: { artifacts: ArtifactMap }) {
  const application = artifacts.application_result;
  const local = artifacts.local_git_delivery_result;
  const remote = artifacts.remote_delivery_result;
  if (!application && !local && !remote) return null;
  return (
    <section className="panel artifact-panel" id="delivery">
      <div className="panel-heading"><div><p className="section-index">Delivery</p><h2>Delivery results</h2></div>{remote?.status === "published" && <span className="status-chip status-published">published</span>}</div>
      <div className="delivery-grid artifact-body">
        {application && <DeliveryStep index="1" title="Working tree" status={application.status}><FileSummary files={application.applied_files} empty="No files reported." />{application.failure_reason && <p className="notice-text">{application.failure_reason}</p>}</DeliveryStep>}
        {local && <DeliveryStep index="2" title="Local Git" status={local.status}><Fact label="Branch" value={local.branch_name} /><Fact label="Commit" value={local.commit_sha} /><Fact label="Base" value={local.base_sha} /></DeliveryStep>}
        {remote && <DeliveryStep index="3" title="GitHub PR" status={remote.status}><Fact label="Remote" value={remote.remote_name} /><Fact label="Base branch" value={remote.base_branch} />{remote.pr_url && remote.pr_number && <a className="pr-link" href={remote.pr_url} target="_blank" rel="noreferrer">Open PR #{remote.pr_number}<span aria-hidden="true">↗</span></a>}{remote.failure_reason && <p className="notice-text">{remote.failure_reason}</p>}</DeliveryStep>}
      </div>
    </section>
  );
}

function DeliveryStep({ index, title, status, children }: { index: string; title: string; status: string; children: React.ReactNode }) { return <article className="delivery-step"><header><span>{index}</span><div><h3>{title}</h3><small>{status.replaceAll("_", " ")}</small></div></header><div>{children}</div></article>; }
function Fact({ label, value }: { label: string; value: string | null }) { return value ? <div className="delivery-fact"><small>{label}</small><code>{value}</code></div> : null; }
function FileSummary({ files, empty }: { files: string[]; empty: string }) { return files.length ? <div className="code-list">{files.map((file) => <code key={file}>{file}</code>)}</div> : <p className="notice-text">{empty}</p>; }
