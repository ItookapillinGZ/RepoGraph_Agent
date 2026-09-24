import { API_BASE } from "@/lib/api";
import type { RunEvent } from "@/lib/types";

export function connectRunEvents(
  runId: string,
  after: number,
  onEvent: (event: RunEvent) => void,
  onConnection: (connected: boolean) => void,
): () => void {
  const source = new EventSource(
    `${API_BASE}/api/runs/${encodeURIComponent(runId)}/stream?after=${after}`,
  );
  source.onopen = () => onConnection(true);
  source.onerror = () => onConnection(false);
  source.addEventListener("run_event", (message) => {
    try {
      onEvent(JSON.parse((message as MessageEvent<string>).data) as RunEvent);
    } catch {
      onConnection(false);
    }
  });
  return () => source.close();
}
