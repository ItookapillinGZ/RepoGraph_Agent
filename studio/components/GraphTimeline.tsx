import type { RunEvent } from "@/lib/types";

const stages = [
  ["Explore", ["repository_exploration"]],
  ["Plan", ["planning", "plan_"]],
  ["Execute", ["execution", "candidate_", "temporary_"]],
  ["Verify", ["verification", "preview_verified"]],
  ["Review", ["review"]],
  ["Correct", ["correction"]],
  ["Approval", ["waiting_for_apply", "waiting_for_git", "waiting_for_remote", "apply_approval", "git_approval", "remote_approval"]],
  ["Apply", ["application", "apply_"]],
  ["Git", ["git_delivery"]],
  ["Remote", ["remote_"]],
  ["PR", ["github_pr", "run_completed"]],
] as const;

function stageState(events: RunEvent[], prefixes: readonly string[]) {
  const relevant = events.filter((event) => prefixes.some((prefix) => event.event_type.startsWith(prefix) || event.phase === prefix));
  if (!relevant.length) return "pending";
  if (relevant.some((event) => event.status === "failed")) return "failed";
  if (relevant.some((event) => event.status === "waiting")) return "waiting";
  const latest = relevant.at(-1);
  if (latest?.status === "started") return "running";
  return "success";
}

export function GraphTimeline({ events }: { events: RunEvent[] }) {
  const correctionRounds = Array.from(new Set(events
    .map((event) => event.metadata.correction_round)
    .filter((round): round is number => typeof round === "number" && round > 0)));

  return (
    <section className="panel timeline-panel">
      <div className="panel-heading">
        <div><p className="section-index">Workflow</p><h2>Graph timeline</h2></div>
        <span className="panel-kicker">{events.length} persisted events</span>
      </div>
      <div className="timeline-stage-row">
        {stages.map(([label, prefixes], index) => {
          const state = stageState(events, prefixes);
          return (
            <div className={`timeline-stage state-${state}`} key={label}>
              <div className="stage-node"><span aria-hidden="true">{state === "success" ? "✓" : state === "failed" ? "!" : index + 1}</span></div>
              <strong>{label}</strong>
              <small>{state}</small>
            </div>
          );
        })}
      </div>
      {correctionRounds.length > 0 && (
        <div className="correction-loop"><span>↳ Correction loop</span>{correctionRounds.map((round) => <code key={round}>Round {round}</code>)}</div>
      )}
      <ol className="event-log">
        {events.slice(-12).reverse().map((event) => (
          <li key={event.sequence}>
            <span className={`event-dot event-${event.status}`} aria-hidden="true" />
            <div><strong>{event.title}</strong>{event.message && <p>{event.message}</p>}</div>
            <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
          </li>
        ))}
      </ol>
    </section>
  );
}
