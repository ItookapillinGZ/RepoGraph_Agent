"use client";

import { useState } from "react";

function lineClass(line: string) {
  if (line.startsWith("diff --git") || line.startsWith("---") || line.startsWith("+++")) return "diff-file";
  if (line.startsWith("@@")) return "diff-hunk";
  if (line.startsWith("+") && !line.startsWith("+++")) return "diff-add";
  if (line.startsWith("-") && !line.startsWith("---")) return "diff-remove";
  return "diff-context";
}

export function CandidateDiff({ diffText }: { diffText?: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    if (!diffText) return;
    await navigator.clipboard.writeText(diffText);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }
  return (
    <section className="panel artifact-panel" id="diff">
      <div className="panel-heading">
        <div><p className="section-index">Artifact</p><h2>Candidate diff</h2></div>
        <button className="ghost-button" onClick={copy} disabled={!diffText} aria-label="Copy unified candidate diff">{copied ? "Copied" : "Copy diff"}</button>
      </div>
      {!diffText ? <p className="empty-state">Candidate diff is not available yet.</p> : (
        <pre className="diff-view" tabIndex={0} aria-label="Unified candidate diff">{diffText.split("\n").map((line, index) => <span className={lineClass(line)} key={index}>{line || " "}{"\n"}</span>)}</pre>
      )}
    </section>
  );
}
