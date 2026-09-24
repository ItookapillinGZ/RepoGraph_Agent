"use client";

import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";

import { API_BASE, createRun, getRepositories, getRuns } from "@/lib/api";
import type { RepositorySummary, StudioRun } from "@/lib/types";

export function NewRunForm() {
  const [repositories, setRepositories] = useState<RepositorySummary[]>([]);
  const [runs, setRuns] = useState<StudioRun[]>([]);
  const [repositoryId, setRepositoryId] = useState("");
  const [task, setTask] = useState("");
  const [selfCorrect, setSelfCorrect] = useState(true);
  const [rounds, setRounds] = useState(2);
  const [loading, setLoading] = useState(true);
  const [runsLoading, setRunsLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [repositoryError, setRepositoryError] = useState("");
  const [runsError, setRunsError] = useState("");

  useEffect(() => {
    getRepositories()
      .then((repositoryData) => {
        setRepositories(repositoryData);
        if (repositoryData.length) setRepositoryId(repositoryData[0].id);
      })
      .catch(() => setRepositoryError(`Could not load repositories from ${API_BASE}. Check the backend connection and refresh.`))
      .finally(() => setLoading(false));
    getRuns()
      .then(setRuns)
      .catch(() => setRunsError("Could not load recent runs. Check the backend connection and refresh."))
      .finally(() => setRunsLoading(false));
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const runId = await createRun({
        repository_id: repositoryId,
        task,
        self_correct: selfCorrect,
        max_correction_rounds: selfCorrect ? rounds : 0,
      });
      window.location.assign(`/runs/${runId}`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Run could not be started.");
      setSubmitting(false);
    }
  }

  return (
    <div className="dashboard-grid">
      <section className="panel new-run-panel">
        <div className="panel-heading">
          <div><p className="section-index">01</p><h2>Start engineering run</h2></div>
          <span className="panel-kicker">Preview only until approval</span>
        </div>
        <form onSubmit={submit}>
          <label>
            Repository
            <select
              value={repositoryId}
              onChange={(event) => setRepositoryId(event.target.value)}
              disabled={loading || submitting || Boolean(repositoryError)}
              required
            >
              {!repositories.length && <option value="">{repositoryError ? "Could not load repositories" : "No repositories available"}</option>}
              {repositories.map((repository) => (
                <option key={repository.id} value={repository.id}>
                  {repository.name}{repository.git_repository ? " · Git" : ""}
                </option>
              ))}
            </select>
          </label>
          {repositoryError && <p className="error-banner" role="alert">{repositoryError}</p>}
          <label>
            Engineering task
            <textarea
              value={task}
              onChange={(event) => setTask(event.target.value)}
              placeholder="Fix user lookup behavior and add regression tests."
              maxLength={10000}
              disabled={submitting}
              required
            />
            <span className="field-meta"><span>Describe the outcome, not the commands.</span><span>{task.length} / 10,000</span></span>
          </label>
          <div className="form-options">
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={selfCorrect}
                onChange={(event) => setSelfCorrect(event.target.checked)}
                disabled={submitting}
              />
              <span><strong>Bounded self-correction</strong><small>Revise failed candidates before approval.</small></span>
            </label>
            <label className="rounds-field">
              Max rounds
              <input
                type="number"
                min={0}
                max={5}
                value={rounds}
                onChange={(event) => setRounds(Number(event.target.value))}
                disabled={!selfCorrect || submitting}
              />
            </label>
          </div>
          {error && <p className="error-banner" role="alert">{error}</p>}
          <button className="primary-button" disabled={submitting || loading || !repositoryId || !task.trim()}>
            {submitting ? "Starting run…" : "Start run"}<span aria-hidden="true">→</span>
          </button>
        </form>
      </section>

      <section className="panel recent-panel">
        <div className="panel-heading">
          <div><p className="section-index">02</p><h2>Recent runs</h2></div>
          <span className="count-badge">{runs.length}</span>
        </div>
        <div className="run-list">
          {runsLoading && <p className="empty-state">Loading workspace runs…</p>}
          {runsError && <p className="empty-state" role="alert">{runsError}</p>}
          {!runsLoading && !runsError && !runs.length && <p className="empty-state">No runs yet. Your first verified change set will appear here.</p>}
          {runs.map((run) => (
            <Link className="run-row" href={`/runs/${run.id}`} key={run.id}>
              <div className="run-row-top"><strong>{run.repository_id}</strong><span className={`status-chip status-${run.status}`}>{run.status}</span></div>
              <p>{run.task}</p>
              <div className="run-row-meta"><span>{run.phase.replaceAll("_", " ")}</span><time>{new Date(run.created_at).toLocaleString()}</time></div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
