import { defineConfig } from "@playwright/test";
import os from "node:os";
import path from "node:path";

const projectRoot = path.resolve(__dirname, "..");
const runtimeRoot = process.env.REPOGRAPH_E2E_RUNTIME
  ?? path.join(os.tmpdir(), `repograph-studio-e2e-${process.pid}`);
const python = path.join(projectRoot, ".venv", "Scripts", "python.exe");
process.env.REPOGRAPH_E2E_RUNTIME = runtimeRoot;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 180_000,
  expect: { timeout: 60_000 },
  reporter: "line",
  use: {
    baseURL: "http://127.0.0.1:3000",
    browserName: "chromium",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: [
    {
      command: `"${python}" -m tests.studio_system.e2e_server`,
      cwd: projectRoot,
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: false,
      timeout: 240_000,
      env: {
        REPOGRAPH_E2E_RUNTIME: runtimeRoot,
        PYTHONPATH: projectRoot,
      },
    },
    {
      command: "npm run dev -- --hostname 127.0.0.1 --port 3000",
      cwd: __dirname,
      url: "http://127.0.0.1:3000",
      reuseExistingServer: false,
      timeout: 240_000,
      env: {
        NEXT_PUBLIC_REPOGRAPH_API_URL: "http://127.0.0.1:8000",
      },
    },
  ],
});
