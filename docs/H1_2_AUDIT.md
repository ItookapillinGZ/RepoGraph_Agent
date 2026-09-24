# Stage H1.2 — Validation Audit & Dependency Security Cleanup

Audit date: 2026-09-01

## 1. Executive result

The H1.1 report was materially inaccurate about one test boundary. As found,
the Studio vertical system test did call the production
publish_local_git_delivery() and did perform real ls-remote, create-only push,
and post-push verification against a local bare repository. It achieved that by
configuring origin as a valid https://github.com/... URL and adding a
repository-local url.<file-uri>.insteadOf rewrite. GitHub HTTP was mocked
through _request_json; structured model output was also injected.

That was real Git plumbing, but it was not an acceptable production-safety
test boundary: the same effective Git configuration could redirect production
Git transport after the configured remote passed GitHub URL parsing.

H1.2 removes that rewrite from the Studio system harness and splits validation:

- Studio system tests use the existing StudioApprovalService.remote_boundary
  injection to validate Approval #3 API/state/service wiring.
- G5B integration tests validate real local-bare Git semantics separately.
- Production G5B rejects local/file/non-GitHub remotes and any effective
  insteadOf or pushInsteadOf entry that matches the validated GitHub target.

No production test-mode flag or host-validation bypass exists.

## 2. Approval #3 actual call chain

The production chain is:

1. POST /api/runs/{run_id}/approve/remote in
   studio_backend/api/approvals.py.
2. FastAPI resolves request.app.state.approvals through
   studio_backend/api/dependencies.py.
3. create_app() stores either the default StudioApprovalService or the
   StudioServices.approval_service_factory result.
4. StudioApprovalService.approve_remote() validates explicit approval,
   approval digest, required G5A artifact, and expected persisted state, then
   claims publishing/remote_delivery with compare-and-set.
5. The default boundary is publish_local_git_delivery(). Tests may inject the
   service's existing remote_boundary callable.
6. publish_local_git_delivery() validates the repository root, remote name,
   G5A delivery through validate_local_git_delivery(), safety snapshot, base
   branch, PR title/body, and repository identity.
7. _resolve_github_repository() reads one pushurl or one url;
   _parse_github_remote_url() accepts only canonical github.com HTTPS/SSH
   forms. H1.2 then rejects matching Git URL rewrites.
8. _verify_github_repository() performs the bounded GitHub repository GET and
   verifies owner/name.
9. _ls_remote_sha() verifies the base and candidate branch.
10. Immediately before push, G5A validation and the local safety snapshot are
    checked again.
11. _push_create_only() pushes the exact commit with a create-only
    force-with-lease and no force overwrite.
12. _ls_remote_sha() verifies the post-push SHA.
13. _lookup_pull_request() performs the bounded PR GET and validates any
    returned PR.
14. _create_pull_request() performs the bounded PR POST when needed and
    _parse_pull_request() validates the returned canonical URL, state, head,
    head SHA, base, and base SHA.

## 3. REAL / MOCKED / INJECTED / BYPASSED matrix

BYPASSED below means the named test does not execute that layer. It does not
mean production disables the layer.

| Layer | H1.1 Studio system test as found | H1.2 Studio system test | G5B integration tests |
|---|---|---|---|
| Studio Approval #3 API | REAL | REAL | BYPASSED |
| SQLite approval state machine | REAL | REAL | BYPASSED |
| StudioApprovalService | REAL | REAL | BYPASSED |
| StudioServices injection | INJECTED: execution/model only | INJECTED: execution/model plus remote adapter | BYPASSED |
| publish_local_git_delivery() | REAL | BYPASSED by injected adapter | REAL |
| validate_local_git_delivery() | REAL | REAL inside injected adapter | REAL |
| GitHub remote parsing | REAL | BYPASSED | BYPASSED via injected resolver; unit/regression tested |
| Git transport target | BYPASSED by Git insteadOf rewrite | BYPASSED | INJECTED local identity |
| git ls-remote | REAL | BYPASSED | REAL |
| git push | REAL | BYPASSED | REAL |
| create-only lease | REAL | BYPASSED | REAL |
| local bare remote | REAL | BYPASSED | REAL |
| post-push SHA verification | REAL | BYPASSED | REAL |
| GitHub repository GET transport | MOCKED via _request_json | BYPASSED | BYPASSED |
| GitHub repository response validation | REAL | BYPASSED | BYPASSED |
| PR lookup transport | MOCKED via _request_json | BYPASSED | MOCKED |
| PR POST transport | MOCKED via _request_json | BYPASSED | MOCKED |
| PR response validation | REAL | BYPASSED | REAL |
| Studio persistence/events/result mapping | REAL | REAL | BYPASSED |

## 4. Production GitHub-only boundary

Production accepts only:

- https://github.com/OWNER/REPO[.git]
- git@github.com:OWNER/REPO[.git]
- ssh://git@github.com/OWNER/REPO[.git]

Production rejects local paths, file://, localhost, HTTP, credentials, custom
ports, queries/fragments, and arbitrary hosts. H1.2 also rejects effective
url.*.insteadOf and url.*.pushInsteadOf entries whose prefix matches the
validated push target before token lookup, network access, or push.

The GitHub API boundary independently permits only HTTPS api.github.com, and PR
URLs must be canonical HTTPS github.com URLs for the validated repository.

## 5. Test-only injection safety

The Studio system suite now uses the already-existing service dependency
injection point. InjectedRemoteBoundary exists only under tests/studio_system;
it has no environment flag and cannot be selected by the production app factory
unless a caller explicitly constructs test StudioServices.

G5B integration tests inject _resolve_github_repository() with a local bare
identity and mock GitHub HTTP/repository verification. This monkeypatch exists
only inside test contexts. The public production publisher has no argument that
accepts an arbitrary transport or host.

## 6. Regression coverage

H1.2 adds explicit production-publisher regressions for:

- a local bare path;
- a file:// bare URL;
- https://localhost/...;
- an arbitrary HTTPS hostname;
- a valid GitHub remote redirected through matching insteadOf.

All cases fail before GitHub HTTP or push. Existing tests separately cover
canonical GitHub forms, create-only exact-SHA refspec, no force overwrite,
idempotent same-SHA retry, races, post-push verification, and local state
preservation.

## 7. npm audit findings

Initial state:

- 3 vulnerable packages reported: 1 moderate, 2 high.
- sharp@0.34.5 was an optional transitive dependency of next@15.5.25.
- postcss@8.4.31 is an exact transitive dependency of next@15.5.25; npm also
  reports next as affected through PostCSS.

After compatible remediation:

- sharp updated to 0.35.4, inside Next 15.5.25's declared
  ^0.34.3 or ^0.35.4 range.
- 2 vulnerable packages remain: 1 moderate, 1 high: next via postcss, plus the
  postcss node.
- npm audit offers only next@16.3.4, a semver-major update, for the remaining
  findings. No force fix or PostCSS override was applied.

## 8. Advisory table

| Package | Severity | Advisory / CVE | Direct | Installed | Patched | Dependency path | Runtime |
|---|---|---|---:|---|---|---|---:|
| postcss | Moderate | GHSA-qx2v-qp2m-jg93 / CVE-2026-41305 | No | 8.4.31 | 8.5.10 | app → next@15.5.25 → postcss | Yes; normally build-time here |
| postcss | High | GHSA-6g55-p6wh-862q / CVE-2026-45623 | No | 8.4.31 | 8.5.12 | app → next@15.5.25 → postcss | Yes; normally build-time here |
| postcss | High | GHSA-r28c-9q8g-f849 / CVE-2026-73646 | No | 8.4.31 | 8.5.18 | app → next@15.5.25 → postcss | Yes; normally build-time here |
| postcss | Moderate | GHSA-fxqj-rqcc-2cmp / CVE-2026-69153 | No | 8.4.31 | 8.5.23 | app → next@15.5.25 → postcss | Yes; normally build-time here |
| sharp | High | GHSA-f88m-g3jw-g9cj / CVE-2026-33327, -33328, -35590, -35591 | No | 0.34.5 → 0.35.4 | 0.35.0 | app → next@15.5.25 → sharp, optional | Yes for image optimization |

next@15.5.25 has no separate direct advisory in this audit output; npm marks it
moderate because it brings in the affected PostCSS node. React, React DOM,
Playwright, ESLint, and the checked ESLint transitive path have no reported
advisories in this audit.

## 9. Dependency impact analysis

PostCSS findings require attacker-influenced CSS, or a malicious CSS-producing
plugin, to reach PostCSS stringification or previous-source-map loading. This
Studio does not accept CSS uploads, expose a CSS processing API, or generate
styles from repository input. Its checked-in styles/globals.css is trusted build
input, and the app is documented as local-only. The vulnerable package is still
in the production dependency tree, so a malicious source/dependency or a future
untrusted-CSS feature could make the issue reachable; it is not dismissed as a
dev-only finding.

The resolved sharp advisory requires processing untrusted image input. The
current Studio has no next/image usage or image-upload surface, but sharp is an
optional Next production component and could become reachable if image
optimization is added. Updating within Next's declared range removes that latent
risk without a Next major upgrade.

## 10. Accepted limitation and remediation decision

PostCSS remains an accepted known limitation for H1.2. Next 15.5.25 pins
PostCSS exactly to 8.4.31, while npm's supported automatic remediation is Next
16.3.4. Overriding Next's exact internal dependency would be an unproven
compatibility change; forcing Next 16 is explicitly outside this audit stage.

Before exposing any untrusted CSS processing or broadening Studio beyond its
trusted-local deployment, upgrade to a Next release that carries PostCSS
8.5.23 or later and rerun the complete frontend/system suite. The same upgrade
should be planned and validated as a dedicated compatibility stage.

## 11. Conclusion

The final validation claim is accurate after H1.2 because the layers are now
separate and named honestly:

- browser/API orchestration through G4/G5A is system-tested;
- Studio Approval #3 to a G5B adapter is system-tested;
- G5B Git push semantics are integration-tested with a real bare remote;
- production GitHub-only parsing and URL-rewrite rejection are regression-tested;
- GitHub HTTP is mocked in offline tests and validated structurally in
  production code/tests.

The remaining PostCSS advisories are constrained, documented, and require a
major framework compatibility decision rather than a blind force fix.
