import { AppShell } from "@/components/AppShell";
import { NewRunForm } from "@/components/NewRunForm";

export default function DashboardPage() {
  return (
    <AppShell>
      <main className="dashboard">
        <section className="hero-panel">
          <div>
            <p className="eyebrow">Repository engineering control plane</p>
            <h1>Move from task to reviewed change set.</h1>
            <p className="hero-copy">
              Plan, verify, review, and deliver repository changes through explicit
              human approvals.
            </p>
          </div>
          <div className="flow-strip" aria-label="Studio workflow">
            <span>Explore</span><i />
            <span>Plan</span><i />
            <span>Execute</span><i />
            <span>Verify</span><i />
            <span>Review</span><i />
            <span>Approve</span>
          </div>
        </section>
        <NewRunForm />
      </main>
    </AppShell>
  );
}
