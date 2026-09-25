import type {
  ArtifactResponse,
  RepositorySummary,
  RunDetail,
  RunEvent,
  StartRunInput,
  StudioRun,
} from "@/lib/types";

export const API_BASE = (process.env.NEXT_PUBLIC_REPOGRAPH_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!response.ok) {
    let message = "RepoGraph Studio request failed.";
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {}
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function getRepositories(): Promise<RepositorySummary[]> {
  const response = await requestJson<{ repositories: RepositorySummary[] }>("/api/repositories");
  return response.repositories;
}

export async function registerRepository(path: string): Promise<RepositorySummary> {
  return requestJson<RepositorySummary>("/api/repositories", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
}

export async function getRuns(): Promise<StudioRun[]> {
  const response = await requestJson<{ runs: StudioRun[] }>("/api/runs");
  return response.runs;
}

export async function createRun(input: StartRunInput): Promise<string> {
  const response = await requestJson<{ run_id: string }>("/api/runs", {
    method: "POST",
    body: JSON.stringify(input),
  });
  return response.run_id;
}

export async function getRun(runId: string): Promise<RunDetail> {
  return requestJson<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`);
}

export async function getRunEvents(runId: string, after = 0): Promise<RunEvent[]> {
  const response = await requestJson<{ events: RunEvent[] }>(
    `/api/runs/${encodeURIComponent(runId)}/events?after=${after}`,
  );
  return response.events;
}

export async function getArtifact<T>(runId: string, kind: string): Promise<T> {
  const response = await requestJson<ArtifactResponse<T>>(
    `/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(kind)}`,
  );
  return response.payload;
}

async function approve<T extends object>(runId: string, action: string, input: T): Promise<void> {
  await requestJson(`/api/runs/${encodeURIComponent(runId)}/approve/${action}`, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function approveApply(runId: string, approvalDigest: string): Promise<void> {
  return approve(runId, "apply", { approved: true, approval_digest: approvalDigest });
}

export async function approveGit(
  runId: string,
  approvalDigest: string,
  branchName: string | null,
  commitMessage: string | null,
): Promise<void> {
  return approve(runId, "git", {
    approved: true,
    approval_digest: approvalDigest,
    branch_name: branchName || null,
    commit_message: commitMessage || null,
  });
}

export async function approveRemote(
  runId: string,
  approvalDigest: string,
  input: {
    remote_name: string;
    base_branch: string;
    branch_name: string | null;
    pr_title: string | null;
    pr_body: string | null;
  },
): Promise<void> {
  return approve(runId, "remote", {
    approved: true,
    approval_digest: approvalDigest,
    ...input,
  });
}

export async function rejectRun(runId: string): Promise<void> {
  await requestJson(`/api/runs/${encodeURIComponent(runId)}/reject`, {
    method: "POST",
    body: JSON.stringify({ rejected: true }),
  });
}
