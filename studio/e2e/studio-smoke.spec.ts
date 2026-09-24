import { expect, test, type Page } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";

type HarnessState = {
  repository: string;
  initial_sha: string;
  initial_index_hash: string;
};

function git(repository: string, ...args: string[]): string {
  return execFileSync("git", args, {
    cwd: repository,
    encoding: "utf-8",
    timeout: 30_000,
  }).trim();
}

function sha256(pathname: string): string {
  return createHash("sha256").update(readFileSync(pathname)).digest("hex");
}

async function receiveNextSseSequence(
  page: Page,
  runId: string,
  after: number,
): Promise<number> {
  return page.evaluate(
    ({ currentRunId, currentAfter }) => new Promise<number>((resolve, reject) => {
      const source = new EventSource(
        `http://127.0.0.1:8000/api/runs/${encodeURIComponent(currentRunId)}/stream?after=${currentAfter}`,
      );
      const timeout = window.setTimeout(() => {
        source.close();
        reject(new Error("Timed out waiting for a Studio SSE event."));
      }, 10_000);
      source.addEventListener("run_event", (event) => {
        window.clearTimeout(timeout);
        source.close();
        const payload = JSON.parse((event as MessageEvent<string>).data) as {
          sequence: number;
        };
        resolve(payload.sequence);
      });
    }),
    { currentRunId: runId, currentAfter: after },
  );
}

test("Studio smoke reaches a real G5A commit through the browser", async ({ page, request }) => {
  const runtimeRoot = process.env.REPOGRAPH_E2E_RUNTIME;
  expect(runtimeRoot).toBeTruthy();
  const state = JSON.parse(
    readFileSync(path.join(runtimeRoot!, "state.json"), "utf-8"),
  ) as HarnessState;

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Start engineering run" })).toBeVisible();
  await page.getByLabel("Repository").selectOption("calculator");
  await page.getByLabel("Engineering task").fill(
    "Fix the add function so the existing regression test passes.",
  );
  await page.getByRole("button", { name: /Start run/ }).click();
  await expect(page).toHaveURL(/\/runs\/[0-9a-f-]+$/);

  const status = page.locator(".status-chip.large");
  await expect(status).toContainText("verified", { timeout: 60_000 });
  await expect(page.getByRole("heading", { name: "Graph timeline" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Engineering plan" })).toBeVisible();
  await expect(page.getByLabel("Unified candidate diff", { exact: true }))
    .toContainText("return a + b");
  await expect(page.locator("#verification")).toContainText("passed");
  await expect(page.locator("#review")).toContainText("good");

  const runId = new URL(page.url()).pathname.split("/").at(-1)!;
  const disconnectedAfter = await receiveNextSseSequence(page, runId, 0);
  const historyResponse = await request.get(
    `http://127.0.0.1:8000/api/runs/${runId}/events?after=${disconnectedAfter}`,
  );
  expect(historyResponse.ok()).toBeTruthy();
  const history = await historyResponse.json() as {
    events: Array<{ sequence: number }>;
  };
  expect(history.events.length).toBeGreaterThan(0);
  expect(history.events.map((event) => event.sequence)).toEqual(
    Array.from(
      { length: history.events.at(-1)!.sequence - disconnectedAfter },
      (_, index) => disconnectedAfter + index + 1,
    ),
  );
  const reconnectedSequence = await receiveNextSseSequence(
    page,
    runId,
    disconnectedAfter,
  );
  expect(reconnectedSequence).toBe(history.events[0].sequence);

  expect(readFileSync(path.join(state.repository, "src", "calculator.py"), "utf-8"))
    .toContain("return a - b");
  expect(git(state.repository, "rev-parse", "HEAD")).toBe(state.initial_sha);
  expect(git(state.repository, "symbolic-ref", "--short", "HEAD")).toBe("main");
  expect(sha256(path.join(state.repository, ".git", "index"))).toBe(
    state.initial_index_hash,
  );

  await page.getByRole("button", { name: "Approve Apply" }).click();
  const applyDialog = page.getByRole("dialog", { name: "Approve filesystem changes" });
  await expect(applyDialog).toBeVisible();
  await applyDialog.getByRole("button", { name: "Approve Apply" }).click();
  await expect(status).toContainText("applied");
  expect(readFileSync(path.join(state.repository, "src", "calculator.py"), "utf-8"))
    .toContain("return a + b");
  expect(git(state.repository, "rev-parse", "HEAD")).toBe(state.initial_sha);
  expect(sha256(path.join(state.repository, ".git", "index"))).toBe(
    state.initial_index_hash,
  );

  await page.getByRole("button", { name: "Create Local Commit" }).click();
  const gitDialog = page.getByRole("dialog", { name: "Approve local branch and commit" });
  await expect(gitDialog).toBeVisible();
  await gitDialog.getByRole("button", { name: "Create Local Commit" }).click();
  await expect(status).toContainText("git created");

  const artifactResponse = await request.get(
    `http://127.0.0.1:8000/api/runs/${runId}/artifacts/local_git_delivery_result`,
  );
  expect(artifactResponse.ok()).toBeTruthy();
  const artifact = await artifactResponse.json() as {
    payload: { branch_name: string; commit_sha: string; base_sha: string };
  };
  expect(artifact.payload.branch_name).toMatch(/^repograph\/[0-9a-f]{12}$/);
  expect(git(state.repository, "rev-parse", artifact.payload.branch_name))
    .toBe(artifact.payload.commit_sha);
  expect(git(state.repository, "rev-parse", `${artifact.payload.commit_sha}^`))
    .toBe(state.initial_sha);
  expect(git(state.repository, "show", `${artifact.payload.commit_sha}:src/calculator.py`))
    .toContain("return a + b");
  expect(git(state.repository, "rev-parse", "HEAD")).toBe(state.initial_sha);
  expect(git(state.repository, "symbolic-ref", "--short", "HEAD")).toBe("main");
  expect(sha256(path.join(state.repository, ".git", "index"))).toBe(
    state.initial_index_hash,
  );
});
