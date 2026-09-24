"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { ApprovalPanel } from "@/components/ApprovalPanel";
import { CandidateDiff } from "@/components/CandidateDiff";
import { CorrectionPanel } from "@/components/CorrectionPanel";
import { DeliveryPanel } from "@/components/DeliveryPanel";
import { GraphTimeline } from "@/components/GraphTimeline";
import { PlanPanel } from "@/components/PlanPanel";
import { ReviewPanel } from "@/components/ReviewPanel";
import { RunStatusHeader } from "@/components/RunStatusHeader";
import { TracePanel } from "@/components/TracePanel";
import { VerificationPanel } from "@/components/VerificationPanel";
import { getArtifact, getRun, getRunEvents } from "@/lib/api";
import { connectRunEvents } from "@/lib/sse";
import type { ArtifactMap, RunDetail, RunEvent } from "@/lib/types";

const artifactKeys: Array<keyof ArtifactMap> = [
  "engineering_plan",
  "candidate_diff",
  "verification",
  "change_set_review",
  "correction_history",
  "application_result",
  "local_git_delivery_result",
  "remote_delivery_result",
];

export function RunDetailClient({ runId }: { runId: string }) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactMap>({});
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState("");
  const refreshTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(async () => {
    const latest = await getRun(runId);
    setRun(latest);
    const available = artifactKeys.filter((kind) => latest.artifact_kinds.includes(kind));
    const loaded = await Promise.all(available.map(async (kind) => [kind, await getArtifact(runId, kind)] as const));
    setArtifacts((current) => ({ ...current, ...Object.fromEntries(loaded) }));
  }, [runId]);

  useEffect(() => {
    Promise.all([getRun(runId), getRunEvents(runId)])
      .then(async ([initialRun, initialEvents]) => {
        setRun(initialRun);
        setEvents(initialEvents);
        const available = artifactKeys.filter((kind) => initialRun.artifact_kinds.includes(kind));
        const loaded = await Promise.all(available.map(async (kind) => [kind, await getArtifact(runId, kind)] as const));
        setArtifacts(Object.fromEntries(loaded));
      })
      .catch((caught: Error) => setError(caught.message));
  }, [runId]);

  useEffect(() => {
    const close = connectRunEvents(runId, 0, (event) => {
      setEvents((current) => current.some((item) => item.sequence === event.sequence) ? current : [...current, event].sort((a, b) => a.sequence - b.sequence));
      if (refreshTimer.current) clearTimeout(refreshTimer.current);
      refreshTimer.current = setTimeout(() => refresh().catch((caught: Error) => setError(caught.message)), 350);
    }, setConnected);
    return () => {
      close();
      if (refreshTimer.current) clearTimeout(refreshTimer.current);
    };
  }, [refresh, runId]);

  if (error && !run) return <AppShell><main className="run-page"><p className="error-banner" role="alert">{error}</p></main></AppShell>;
  if (!run) return <AppShell><main className="run-page"><p className="empty-state">Loading Studio run…</p></main></AppShell>;

  return (
    <AppShell>
      <main className="run-page">
        <RunStatusHeader run={run} connected={connected} />
        <TracePanel trace={run.trace} />
        <GraphTimeline events={events} />
        <nav className="artifact-nav" aria-label="Run artifacts">
          <a href="#plan">Plan</a><a href="#diff">Diff</a><a href="#verification">Verification</a><a href="#review">Review</a><a href="#corrections">Corrections</a><a href="#approval">Approval</a>
        </nav>
        <div className="artifact-stack">
          <PlanPanel plan={artifacts.engineering_plan} />
          <CandidateDiff diffText={artifacts.candidate_diff?.diff_text} />
          <VerificationPanel verification={artifacts.verification} />
          <ReviewPanel review={artifacts.change_set_review} />
          <CorrectionPanel history={artifacts.correction_history} />
          <ApprovalPanel run={run} artifacts={artifacts} onComplete={refresh} />
          <DeliveryPanel artifacts={artifacts} />
        </div>
      </main>
    </AppShell>
  );
}
