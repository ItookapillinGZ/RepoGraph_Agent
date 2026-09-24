import Link from "next/link";
import type { ReactNode } from "react";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="app-frame">
      <header className="topbar">
        <Link className="brand" href="/" aria-label="RepoGraph Studio home">
          <span className="brand-mark" aria-hidden="true">RG</span>
          <span>
            <strong>RepoGraph</strong>
            <small>Studio</small>
          </span>
        </Link>
        <div className="local-badge"><span /> Trusted local workspace</div>
      </header>
      {children}
    </div>
  );
}
