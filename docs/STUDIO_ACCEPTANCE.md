# RepoGraph Studio H1.1 Acceptance and Recovery

This procedure validates Studio against a real GitHub repository after the
offline system and browser tests pass. Use a dedicated disposable test
repository. Never begin with an important production repository.

## Automated validation first

From the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\bandit.exe -r . -q -x .\.venv,.\tests

cd studio
npm install
npm run lint
npm run typecheck
npm run build
npx playwright install chromium
npm run e2e
```

The Playwright command starts a deterministic FastAPI server and a Next.js
server against a dedicated temporary workspace. It runs Chromium only, with
one worker, and stops at G5A. It never publishes to GitHub.

The offline Studio system suite exercises Approval #3 through an injected G5B
adapter. It verifies the real FastAPI endpoint, SQLite state machine, service
wiring, and G5A validation, but does not call the production G5B publisher.
G5B's separate integration tests use a real local bare remote for ls-remote,
create-only push, race, and post-push verification while injecting GitHub
identity/HTTP boundaries. The exact H1.2 matrix is in
[H1_2_AUDIT.md](H1_2_AUDIT.md).

## Dedicated GitHub repository setup

1. Create a new private or public repository used only for RepoGraph Studio
   acceptance.
2. Clone it beneath `REPOGRAPH_WORKSPACE_ROOT` as a direct child directory.
3. Configure a local Git identity and create a `main` branch with one small
   regression and its failing test.
4. Confirm `origin` is the intended `github.com` repository and that the
   current working tree and index are clean.
5. Create a narrowly scoped token that can push a branch and open a pull
   request only in this test repository. Export it as `GITHUB_TOKEN` in the
   FastAPI process; never use a `NEXT_PUBLIC_*` variable.
6. Start FastAPI and Next.js using the README Studio instructions.

## Manual acceptance flow

Use a task such as: `Fix the add function so the existing regression test passes.`

Before approval #1, independently record:

```powershell
git rev-parse HEAD
git symbolic-ref --short HEAD
git status --porcelain=v1
git diff --cached
```

Start the Studio run and wait for `verified`. Inspect the persisted timeline,
plan, candidate diff, verification, and change-set review. Confirm the real
repository file, current branch, HEAD, and index have not changed.

Approve **Apply**. Confirm only the approved files now match the candidate,
while the checked-out branch, HEAD, and index remain unchanged.

Approve **Create Local Commit**. Confirm Studio reports `git_created`; the
current branch, HEAD, working tree, and index still match the post-G4 state;
and the new `repograph/<digest-prefix>` commit has the original HEAD as its
only parent and contains exactly the approved files.

Approve **Publish GitHub PR**. Confirm the remote branch SHA equals the local
approved commit and the returned pull request has the requested base, the
approved head, and no unrelated file changes.

## Acceptance checklist

- [ ] Backend starts
- [ ] Frontend starts
- [ ] Repository appears
- [ ] Run reaches `verified`
- [ ] Timeline order is correct
- [ ] Plan is visible
- [ ] Diff is visible
- [ ] Verification is visible
- [ ] Review is visible
- [ ] Apply approval succeeds
- [ ] Files exactly match the candidate
- [ ] Local commit approval succeeds
- [ ] Current HEAD is unchanged
- [ ] Local `repograph/...` branch exists
- [ ] Remote approval succeeds
- [ ] Remote branch SHA equals the local approved commit
- [ ] GitHub pull request opens
- [ ] Pull-request head and base are correct
- [ ] Pull-request diff is exact
- [ ] No unrelated files are present in the pull request

## Restart and retry behavior

Startup reconciliation never invokes an LLM, runs tests, applies files, writes
Git state, pushes, or calls GitHub. Ordinary queued/running previews become
`interrupted` and must be started again as a new run.

An interrupted G4 application is not automatically repeated. The same Apply
approval can explicitly retry and calls the unchanged G4 boundary, which
revalidates original and candidate hashes. If the repository still contains
the original bytes, retry can apply safely. If a crash occurred after all
candidate bytes were written but before the result was persisted, current G4
treats the retry as stale. Studio deliberately does not reinterpret that as
success; manual reconciliation is required.

For interrupted G5A delivery, startup performs only read-only validation. An
existing exact branch/commit is reconstructed as `git_created`; an absent
branch allows explicit Git approval retry; a mismatched branch becomes
`conflict` and is never overwritten.

Interrupted G5B delivery always becomes `interrupted` without network access.
Only an explicit Remote approval retries it. G5B then verifies an already
pushed matching SHA and reuses an existing matching pull request when present.

If repository state is ambiguous, preserve the repository and Studio database,
do not delete or force-update refs, and reconcile the exact approved digest,
working-tree bytes, index, local branch, remote branch, and pull request before
continuing.
