export type RepositorySummary = {
  id: string;
  name: string;
  git_repository: boolean;
};

export type StudioRun = {
  id: string;
  repository_id: string;
  task: string;
  status: string;
  phase: string;
  created_at: string;
  updated_at: string;
  approval_digest: string | null;
  failure_reason: string | null;
};

export type RunTraceSummary = {
  trace_id: string;
  duration_ms: number | null;
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  tool_calls: number;
  sandbox_executions: number;
  artifact_count: number;
  lineage_summary: string;
};

export type RunDetail = StudioRun & {
  artifact_kinds: string[];
  latest_event_sequence: number;
  github_auth_configured: boolean;
  trace: RunTraceSummary | null;
};

export type RunEvent = {
  run_id: string;
  sequence: number;
  timestamp: string;
  phase: string;
  event_type: string;
  status: "started" | "completed" | "failed" | "info" | "waiting";
  title: string;
  message: string;
  metadata: Record<string, unknown>;
};

export type ArtifactResponse<T = unknown> = {
  run_id: string;
  kind: string;
  version: number;
  created_at: string;
  payload: T;
};

export type EngineeringPlan = {
  summary: string;
  files: Array<{ path: string; action: "modify" | "add" | "delete"; rationale: string }>;
  tests: Array<{ path: string | null; purpose: string }>;
  risks: string[];
  assumptions: string[];
};

export type CandidateDiffArtifact = { diff_text: string };

export type StaticFinding = {
  tool: string;
  rule_id: string;
  category: string;
  severity: string;
  message: string;
  line_number: number | null;
  column: number | null;
};

export type VerificationArtifact = {
  status: "verified" | "failed" | "error";
  static_analysis: Array<{
    path: string;
    result: { findings: StaticFinding[]; tools_run: string[]; tool_errors: string[] };
  }>;
  test_result: {
    status: "not_run" | "passed" | "failed" | "error" | "timed_out";
    framework: string | null;
    test_files: string[];
    exit_code: number | null;
    stdout: string;
    stderr: string;
    cases: Array<{ node_id: string | null; status: string; message: string | null }>;
    warnings: string[];
  };
  warnings: string[];
};

export type ReviewFinding = {
  category: string;
  severity: string;
  title: string;
  description: string;
  line_number: number | null;
  suggestion: string;
};

export type ChangeSetReviewArtifact = {
  overall_rating: string;
  summary: string;
  high_risk_files: string[];
  warnings: string[];
  file_results: Array<{
    target: { path: string; change_kind: string };
    status: string;
    warnings: string[];
    review: null | {
      overall_rating: string;
      summary: string;
      findings: ReviewFinding[];
    };
  }>;
};

export type CorrectionHistoryArtifact = {
  correction_rounds_used: number;
  attempt_history: Array<{
    attempt: number;
    diff_summary: string;
    verification_status: string;
    review_rating: string | null;
    failure_reasons: string[];
  }>;
};

export type ApplicationResultArtifact = {
  status: string;
  applied_files: string[];
  rolled_back_files: string[];
  warnings: string[];
  failure_reason: string | null;
};

export type LocalDeliveryArtifact = {
  status: string;
  branch_name: string | null;
  commit_sha: string | null;
  base_sha: string | null;
  committed_files: string[];
  warnings: string[];
  failure_reason: string | null;
};

export type RemoteDeliveryArtifact = {
  status: string;
  remote_name: string | null;
  branch_name: string | null;
  commit_sha: string | null;
  base_branch: string | null;
  pushed: boolean;
  push_created: boolean;
  pr_created: boolean;
  pr_number: number | null;
  pr_url: string | null;
  warnings: string[];
  failure_reason: string | null;
};

export type ArtifactMap = {
  engineering_plan?: EngineeringPlan;
  candidate_diff?: CandidateDiffArtifact;
  verification?: VerificationArtifact;
  change_set_review?: ChangeSetReviewArtifact;
  correction_history?: CorrectionHistoryArtifact;
  application_result?: ApplicationResultArtifact;
  local_git_delivery_result?: LocalDeliveryArtifact;
  remote_delivery_result?: RemoteDeliveryArtifact;
};

export type StartRunInput = {
  repository_id: string;
  task: string;
  self_correct: boolean;
  max_correction_rounds: number;
};
