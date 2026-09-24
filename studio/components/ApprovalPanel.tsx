"use client";

import { useState } from "react";

import { approveApply, approveGit, approveRemote, rejectRun } from "@/lib/api";
import type { ArtifactMap, RunDetail } from "@/lib/types";

type ApprovalKind = "apply" | "git" | "remote" | "reject";

const confirmations: Record<ApprovalKind, { title: string; message: string; action: string }> = {
  apply: { title: "Approve filesystem changes", message: "You are approving application of the exact verified candidate to the real repository working tree.", action: "Approve Apply" },
  git: { title: "Approve local branch and commit", message: "You are approving creation of a new local Git branch and commit from the applied candidate.", action: "Create Local Commit" },
  remote: { title: "Approve network delivery", message: "You are approving a network push and GitHub pull-request creation for the approved local commit.", action: "Publish GitHub PR" },
  reject: { title: "Reject verified candidate", message: "The verified preview will be retained for history but can no longer mutate the repository.", action: "Reject Candidate" },
};

export function ApprovalPanel({ run, artifacts, onComplete }: { run: RunDetail; artifacts: ArtifactMap; onComplete: () => Promise<void> }) {
  const [modal, setModal] = useState<ApprovalKind | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [expandedDigest, setExpandedDigest] = useState(false);
  const [branchName, setBranchName] = useState("");
  const [commitMessage, setCommitMessage] = useState("");
  const [remoteName, setRemoteName] = useState("origin");
  const [baseBranch, setBaseBranch] = useState("main");
  const [prTitle, setPrTitle] = useState("");
  const digest = run.approval_digest;
  const approvalStage = run.status === "verified" || (run.status === "interrupted" && run.phase === "application")
    ? "apply"
    : run.status === "applied" || (run.status === "interrupted" && run.phase === "git_delivery")
      ? "git"
      : run.status === "git_created" || run.status === "partial" || (run.status === "interrupted" && run.phase === "remote_delivery")
        ? "remote"
        : null;

  async function confirm() {
    if (!modal || !digest) return;
    setBusy(true); setError("");
    try {
      if (modal === "apply") await approveApply(run.id, digest);
      if (modal === "git") await approveGit(run.id, digest, branchName || null, commitMessage || null);
      if (modal === "remote") await approveRemote(run.id, digest, { remote_name: remoteName, base_branch: baseBranch, branch_name: artifacts.local_git_delivery_result?.branch_name ?? null, pr_title: prTitle || null, pr_body: null });
      if (modal === "reject") await rejectRun(run.id);
      setModal(null);
      await onComplete();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Approval action failed safely.");
    } finally { setBusy(false); }
  }

  const canAct = approvalStage !== null;
  if (!canAct && run.status !== "published") return null;
  return (
    <section className="panel approval-panel" id="approval">
      <div className="approval-accent" />
      <div className="panel-heading"><div><p className="section-index">Human control</p><h2>Approval boundary</h2></div><span className="approval-count">{approvalStage === "apply" ? "1 / 3" : approvalStage === "git" ? "2 / 3" : approvalStage === "remote" ? "3 / 3" : "Complete"}</span></div>
      <div className="approval-body">
        {digest && <div className="digest-block"><small>Approval digest</small><button onClick={() => setExpandedDigest((value) => !value)} aria-expanded={expandedDigest}><code>{expandedDigest ? digest : `${digest.slice(0, 12)}…`}</code><span>{expandedDigest ? "Collapse" : "Expand"}</span></button></div>}
        {run.status === "interrupted" && <p className="notice-text">This mutation was interrupted. Retry remains an explicit human action and will revalidate the approved bundle before any write.</p>}
        {approvalStage === "apply" && <><h3>Approved candidate is ready.</h3><p>Review the plan, diff, verification evidence, and findings before applying any real filesystem changes.</p><div className="approval-actions">{run.status === "verified" && <button className="danger-ghost-button" onClick={() => setModal("reject")}>Reject</button>}<button className="primary-button inline" onClick={() => setModal("apply")}>{run.status === "interrupted" ? "Retry Apply" : "Approve Apply"} <span>→</span></button></div></>}
        {approvalStage === "git" && <><h3>Working-tree changes applied.</h3><p>Create a local branch and commit. Branch and message are optional; the safe delivery boundary supplies deterministic defaults.</p><div className="approval-form-grid"><label>Branch name<input value={branchName} onChange={(event) => setBranchName(event.target.value)} placeholder="Automatic safe branch" /></label><label>Commit message<input value={commitMessage} onChange={(event) => setCommitMessage(event.target.value)} placeholder="Automatic approved message" /></label></div><button className="primary-button inline right" onClick={() => setModal("git")}>{run.status === "interrupted" ? "Retry Local Commit" : "Create Local Commit"} <span>→</span></button></>}
        {approvalStage === "remote" && <><h3>Local approved branch created.</h3><p><code>{artifacts.local_git_delivery_result?.branch_name}</code></p>{!run.github_auth_configured && <p className="notice-text">GitHub authentication is not configured on the server.</p>}<div className="approval-form-grid"><label>Remote name<input value={remoteName} onChange={(event) => setRemoteName(event.target.value)} required /></label><label>Base branch<input value={baseBranch} onChange={(event) => setBaseBranch(event.target.value)} required /></label><label className="wide">PR title<input value={prTitle} onChange={(event) => setPrTitle(event.target.value)} placeholder="Automatic title" /></label></div><button className="primary-button inline right" onClick={() => setModal("remote")} disabled={!baseBranch.trim() || !remoteName.trim() || !run.github_auth_configured}>{run.status === "git_created" ? "Publish GitHub PR" : "Retry Remote Delivery"} <span>→</span></button></>}
        {run.status === "published" && <><h3>GitHub delivery completed.</h3><p>The canonical pull-request result is available below.</p></>}
        {error && <p className="error-banner" role="alert">{error}</p>}
      </div>
      {modal && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target && !busy) setModal(null); }}><div className="confirmation-modal" role="dialog" aria-modal="true" aria-labelledby="confirmation-title"><p className="section-index">Consequential action</p><h3 id="confirmation-title">{confirmations[modal].title}</h3><p>{confirmations[modal].message}</p>{digest && modal !== "reject" && <div className="modal-digest"><small>Exact approval digest</small><code>{digest}</code></div>}<div className="modal-actions"><button className="ghost-button" onClick={() => setModal(null)} disabled={busy}>Cancel</button><button className={modal === "reject" ? "danger-button" : "primary-button inline"} onClick={confirm} disabled={busy}>{busy ? "Working…" : confirmations[modal].action}</button></div></div></div>}
    </section>
  );
}
